"""Replay the synthetic session through each context strategy and measure.

For every request we record the context size. At checkpoints we ask questions
whose answers were planted earlier and check whether the facts needed to answer
are *present in the assembled context* and the request *fits the window*. That
is a necessary condition for a correct answer; ``live_eval.py`` checks the real
model's answers when an API key is available.

A final "new session" checkpoint starts with an empty transcript - only the
persistent note store carries over - to show what memory is actually for.

    python -m benchmark.run_benchmark
    python -m benchmark.run_benchmark --window 16000 --turns 80
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from context_budget import (BudgetPlan, EngineeredStrategy, Message, NaiveTruncationStrategy,
                            NoteStore, UnboundedStrategy, chunk_document, get_tokenizer)

from .scenario import build_scenario

RESULTS = Path(__file__).resolve().parent.parent / "results"
NEW_SESSION = "S2"
SESSION2_QIDS = ["root_cause", "batches", "decision"]


def make_strategies(sc, chunks, plan, tok):
    args, kw = (sc.system_prompt, chunks), dict(plan=plan, tok=tok)
    return [
        UnboundedStrategy(*args, **kw),
        NaiveTruncationStrategy(*args, **kw),
        EngineeredStrategy(*args, **kw),
        EngineeredStrategy(*args, retrieval=False, label="engineered -retrieval", **kw),
        EngineeredStrategy(*args, clearing=False, label="engineered -tool_clearing", **kw),
        EngineeredStrategy(*args, compaction=False, label="engineered -compaction", **kw),
        EngineeredStrategy(*args, memory=False, label="engineered -memory", **kw),
    ]


def _check(checks, contexts, label, q, s, ctx, plan):
    found = [f for f in q.must_contain if f in ctx.text]
    fits = ctx.total_tokens <= plan.input_budget
    checks.append({"checkpoint": label, "qid": q.qid, "kind": q.kind, "strategy": s.name,
                   "facts_present": len(found) / len(q.must_contain),
                   "answerable": fits and len(found) == len(q.must_contain),
                   "tokens": ctx.total_tokens, "within_window": fits})
    contexts[(label, q.qid, s.name)] = ctx


def run(window: int = 8_000, turns: int = 40, tokenizer: str | None = None, seed: int = 7,
        verbose: bool = True, write: bool = True):
    tok = get_tokenizer(tokenizer)
    plan = BudgetPlan(context_window=window)
    sc = build_scenario(seed, turns)
    chunks = [c for d in sc.documents for c in chunk_document(*d, tok)]
    strategies = make_strategies(sc, chunks, plan, tok)

    history: list[Message] = []
    notes = NoteStore()
    per_request, checks, contexts = [], [], {}

    for turn in sc.turns:
        for s in strategies:  # the request the model sees at the start of this turn
            ctx = s.build(history, turn.user, notes)
            per_request.append({"turn": turn.n, "strategy": s.name, "tokens": ctx.total_tokens,
                                "within_window": ctx.total_tokens <= plan.input_budget,
                                **{f"slot_{k}": v for k, v in ctx.breakdown().items()}})
        history.append(Message("user", turn.user, turn.n))
        if turn.tool_call:
            history.append(Message("tool", turn.tool_output, turn.n, tool_call=turn.tool_call))
        history.append(Message("assistant", turn.assistant, turn.n))
        if turn.note:
            notes.add(turn.n, turn.note)
        for q in sc.checkpoints.get(turn.n, []):
            for s in strategies:
                _check(checks, contexts, turn.n, q, s, s.build(history, q.text, notes), plan)

    compactions = {s.name: getattr(getattr(s, "compactor", None), "compactions", 0) for s in strategies}

    # New session: empty transcript, fresh strategy state, same persistent notes.
    finals = {q.qid: q for q in sc.checkpoints[turns]}
    for s in make_strategies(sc, chunks, plan, tok):
        for qid in SESSION2_QIDS:
            _check(checks, contexts, NEW_SESSION, finals[qid], s, s.build([], finals[qid].text, notes), plan)

    summary = summarise(strategies, per_request, checks, compactions)
    if write:
        RESULTS.mkdir(exist_ok=True)
        _write(per_request, checks, summary, plan, tok.name, turns)
    if verbose:
        print_report(summary, checks, plan, tok.name, sc, turns)
    return summary, per_request, checks, contexts


def summarise(strategies, per_request, checks, compactions):
    out = []
    for s in strategies:
        reqs = [r for r in per_request if r["strategy"] == s.name]
        qs = [c for c in checks if c["strategy"] == s.name and c["checkpoint"] != NEW_SESSION]
        s2 = [c for c in checks if c["strategy"] == s.name and c["checkpoint"] == NEW_SESSION]
        out.append({
            "strategy": s.name,
            "requests": len(reqs),
            "total_input_tokens": sum(r["tokens"] for r in reqs),
            "mean_tokens": round(sum(r["tokens"] for r in reqs) / len(reqs)),
            "peak_tokens": max(r["tokens"] for r in reqs),
            "requests_over_window": sum(not r["within_window"] for r in reqs),
            "questions": len(qs),
            "answerable": sum(c["answerable"] for c in qs),
            "answerable_pct": round(100 * sum(c["answerable"] for c in qs) / len(qs), 1),
            "new_session_answerable": f"{sum(c['answerable'] for c in s2)}/{len(s2)}",
            "compactions": compactions[s.name],
        })
    return out


def _write(per_request, checks, summary, plan, tok_name, turns):
    with open(RESULTS / "per_request.csv", "w", newline="") as f:
        keys = ["turn", "strategy", "tokens", "within_window"] + sorted(
            {k for r in per_request for k in r if k.startswith("slot_")})
        w = csv.DictWriter(f, fieldnames=keys, restval=0)
        w.writeheader()
        w.writerows(per_request)
    (RESULTS / "results.json").write_text(json.dumps(
        {"plan": asdict(plan) | {"input_budget": plan.input_budget}, "tokenizer": tok_name,
         "turns": turns, "summary": summary, "checks": checks}, indent=2, default=str))


def print_report(summary, checks, plan, tok_name, sc, turns):
    print(f"\nContext window {plan.context_window:,} -> input budget {plan.input_budget:,} "
          f"({plan.output_reserve:,} reserved for output, {plan.safety_margin:.0%} safety margin) | tokenizer: {tok_name}")
    print(f"Session: {turns} turns, {len(sc.documents)} knowledge-base documents\n")
    hdr = (f"{'strategy':<27}{'total in':>10}{'mean/req':>10}{'peak':>8}{'over':>6}"
           f"{'answerable':>14}{'new sess':>10}{'compact':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in summary:
        print(f"{r['strategy']:<27}{r['total_input_tokens']:>10,}{r['mean_tokens']:>10,}{r['peak_tokens']:>8,}"
              f"{r['requests_over_window']:>6}{r['answerable']:>7}/{r['questions']:<2} ({r['answerable_pct']:>3.0f}%)"
              f"{r['new_session_answerable']:>10}{r['compactions']:>9}")
    print("\nanswerable = all facts needed are in the context AND the request fits the window."
          "\nover = requests that exceed the window (the API would reject them).")

    names = [r["strategy"] for r in summary]
    short = {"unbounded": "unbnd", "naive_truncation": "naive", "engineered": "eng"}
    for label, title in [(turns, f"final checkpoint, turn {turns}"), (NEW_SESSION, "new session, notes only")]:
        rows = [c for c in checks if c["checkpoint"] == label]
        qids = list(dict.fromkeys(c["qid"] for c in rows))
        print(f"\nPer question ({title}):")
        print(f"  {'question':<14}{'kind':<22}" + "".join(f"{short.get(n, n.split()[-1][:8]):>9}" for n in names))
        for qid in qids:
            r = [c for c in rows if c["qid"] == qid]
            cells = []
            for n in names:
                c = next(x for x in r if x["strategy"] == n)
                cells.append("over" if not c["within_window"] else ("yes" if c["answerable"] else "-"))
            print(f"  {qid:<14}{r[0]['kind']:<22}" + "".join(f"{x:>9}" for x in cells))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", type=int, default=8_000, help="context window in tokens")
    ap.add_argument("--turns", type=int, default=40, help="session length (>= 14)")
    ap.add_argument("--tokenizer", choices=["heuristic", "tiktoken", "anthropic"], default=None)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    run(a.window, a.turns, a.tokenizer, a.seed)
