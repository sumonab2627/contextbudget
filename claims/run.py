"""Claims extraction: efficient vs naive context, across models.

    python -m claims.run                         # offline: exact token counts per model tokenizer
    python -m claims.run --mock                  # exercise the live path with no network
    python -m claims.run --live gpt gemini slm   # real calls (needs API keys / running Ollama)
    python -m claims.run --live claude           # Claude; ANTHROPIC_API_KEY also enables exact offline counts

Offline output tokens are the size of the *reference answer* in the format each pipeline
asks for. They exclude any reasoning the model might add, so they understate the naive
pipeline's real output. Live runs report billed usage.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .generate import DATA
from .generate import main as generate
from .ingest import prepare, raw_text, render
from .models import PROFILES, counter_for
from .pipelines import cost_offline, plan_efficient, plan_naive, run_live
from .providers import MockProvider, provider_for
from .scoring import correctness_per_token, score

RESULTS = Path(__file__).resolve().parent.parent / "results"


def load():
    if not (DATA / "claim_pack.pdf").exists():
        generate()
    gt = json.loads((DATA / "ground_truth.json").read_text())
    raw, cleaned, routed = prepare(DATA / "claim_pack.pdf")
    return gt, raw, cleaned, routed


def plans(model, gt, raw, routed):
    c = counter_for(model.tokenizer)
    return c, {"naive": cost_offline(plan_naive(raw, gt, model, c.count), c.count),
               "efficient": cost_offline(plan_efficient(routed, gt, model, c.count), c.count)}


def _final_cover(calls):
    return set(calls[-1].covers) if calls[0].pipeline == "naive" else {k for c in calls for k in c.covers}


def offline(keys: list[str]):
    gt, raw, cleaned, routed = load()
    n_fields = len(gt["evidence"])
    rows = []
    for key in keys:
        m = PROFILES[key]
        counter, p = plans(m, gt, raw, routed)
        for name, calls in p.items():
            tin, tout = sum(c.tokens_in for c in calls), sum(c.tokens_out for c in calls)
            covered = len(_final_cover(calls))
            rows.append({"model": m.label, "key": key, "size_class": m.size_class, "window": m.context_window,
                         "tokenizer": counter.name, "exact_tokens": counter.exact, "pipeline": name,
                         "calls": len(calls), "tokens_in": tin, "tokens_out": tout, "tokens_total": tin + tout,
                         "cost_usd": m.cost(tin, tout), "fields_with_evidence": covered, "fields": n_fields,
                         "evidence_per_1k_tokens": correctness_per_token(covered, tin + tout),
                         "call_detail": [{"task": c.task, "in": c.tokens_in, "out": c.tokens_out} for c in calls]})

    # Tokenizer effect: identical text, different models
    samples = {"raw claim pack": raw_text(raw),
               "cleaned + routed pages": "\n".join(render(v) for v in routed.values()),
               "invoice pages only": render(routed["financial"]),
               "clinical narrative only": render(routed["clinical"])}
    tok_rows = {name: {PROFILES[k].label + ("" if counter_for(PROFILES[k].tokenizer).exact else " ~"): counter_for(PROFILES[k].tokenizer).count(text) for k in keys}
                for name, text in samples.items()}
    # Where the tokens were: raw pack by page type
    ref = counter_for("o200k")
    by_kind: dict[str, dict] = {}
    for rp, cp in zip(raw, cleaned):
        d = by_kind.setdefault(cp.kind, {"pages": 0, "raw_tokens": 0, "sent_tokens": 0})
        d["pages"] += 1
        d["raw_tokens"] += ref.count(rp.text)
        routed_ids = {p.number for v in routed.values() for p in v}
        d["sent_tokens"] += ref.count(cp.text) if cp.number in routed_ids else 0

    RESULTS.mkdir(exist_ok=True)
    out = {"generated": time.strftime("%Y-%m-%d %H:%M"), "rows": rows, "tokenizer_effect": tok_rows,
           "pack_breakdown_o200k": by_kind, "prices": {k: PROFILES[k].price_source for k in keys}}
    (RESULTS / "claims_offline.json").write_text(json.dumps(out, indent=2))
    print_offline(rows, tok_rows, by_kind)
    return out


def print_offline(rows, tok_rows, by_kind):
    print("\nCLAIMS EXTRACTION - naive vs efficient context (offline, per-model tokenizers)\n")
    hdr = f"{'model':<30}{'pipeline':<11}{'calls':>6}{'input':>9}{'output':>8}{'total':>9}{'saving':>8}{'$/100k claims':>15}{'fields':>8}{'CPT*':>7}"
    print(hdr)
    print("-" * len(hdr))
    for i in range(0, len(rows), 2):
        naive, eff = rows[i], rows[i + 1]
        for r in (naive, eff):
            est = "" if r["exact_tokens"] else " ~"
            save = "" if r is naive else f"{100 * (1 - eff['tokens_total'] / naive['tokens_total']):.0f}%"
            cost = "n/a" if r["cost_usd"] is None else f"{r['cost_usd'] * 100_000:,.0f}"
            print(f"{(r['model'] + est) if r is naive else '':<30}{r['pipeline']:<11}{r['calls']:>6}{r['tokens_in']:>9,}"
                  f"{r['tokens_out']:>8,}{r['tokens_total']:>9,}{save:>8}{cost:>15}"
                  f"{r['fields_with_evidence']:>5}/{r['fields']:<2}{r['evidence_per_1k_tokens']:>7.2f}")
    print("\n~ = estimated (no public tokenizer; set ANTHROPIC_API_KEY for exact counts via the free count_tokens API)")
    print("fields = fields whose evidence is fully visible in the call that produces them (necessary, not sufficient)")
    print("CPT* = fields-with-evidence per 1k tokens; live runs report true Correctness Per Token")

    print("\nSame text, different tokenizers (input tokens):")
    models = list(next(iter(tok_rows.values())))
    print(f"  {'text':<26}" + "".join(f"{m.split(' (')[0][:22]:>24}" for m in models))
    for name, counts in tok_rows.items():
        base = next((v for m, v in counts.items() if m.startswith("GPT")), next(iter(counts.values())))
        print(f"  {name:<26}" + "".join(f"{v:>16,} ({v / base:>4.2f}x)" for v in counts.values()))

    print("\nWhere the raw pack's tokens are (o200k) and what the efficient pipeline sends:")
    for kind, d in sorted(by_kind.items(), key=lambda x: -x[1]["raw_tokens"]):
        print(f"  {kind:<24}{d['pages']:>3} pages {d['raw_tokens']:>7,} raw -> {d['sent_tokens']:>6,} sent")


def live(keys: list[str], mock: bool = False):
    gt, raw, cleaned, routed = load()
    RESULTS.mkdir(exist_ok=True)
    rows = []
    for key in keys:
        m = PROFILES[key]
        counter, p = plans(m, gt, raw, routed)
        for name, calls in p.items():
            prov = MockProvider(m, calls, counter.count) if mock else provider_for(m)
            t = time.perf_counter()
            try:
                final, log = run_live(calls, prov)
            except Exception as exc:
                print(f"{m.label} / {name}: FAILED - {type(exc).__name__}: {exc}")
                continue
            s = score(final, gt)
            tin, tout = sum(x["tokens_in"] for x in log), sum(x["tokens_out"] for x in log)
            correct = sum(s.values())
            rows.append({"model": m.label, "pipeline": name, "calls": len(log), "tokens_in": tin, "tokens_out": tout,
                         "reasoning_tokens": sum(x.get("reasoning_tokens", 0) for x in log),
                         "retries": sum(x["attempt"] > 1 for x in log), "correct": correct, "fields": len(s),
                         "cpt": correctness_per_token(correct, tin + tout), "cost_usd": m.cost(tin, tout),
                         "seconds": round(time.perf_counter() - t, 1), "missed": [k for k, v in s.items() if not v],
                         "extraction": final, "log": log})
            r = rows[-1]
            cost = "n/a" if r["cost_usd"] is None else f"${r['cost_usd']:.5f}"
            print(f"{m.label:<30}{name:<10} calls={r['calls']:<3} in={tin:>7,} out={tout:>6,} "
                  f"correct={correct}/{len(s)} CPT={r['cpt']:.2f} retries={r['retries']} cost={cost}")
    tag = "mock" if mock else time.strftime("%Y%m%d-%H%M")
    (RESULTS / f"claims_live_{tag}.json").write_text(json.dumps(rows, indent=2, default=str))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(PROFILES), choices=list(PROFILES))
    ap.add_argument("--live", nargs="*", choices=list(PROFILES), help="run real calls for these models")
    ap.add_argument("--mock", action="store_true", help="run the live path with a replay provider")
    a = ap.parse_args()
    if a.live is not None:
        live(a.live or a.models)
    elif a.mock:
        live(a.models, mock=True)
    else:
        offline(a.models)
