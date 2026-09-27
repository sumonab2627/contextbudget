"""Optional: ask a real Claude model the checkpoint questions using each strategy's context.

The offline benchmark proves the facts are *in* the context and the request
*fits*. This script closes the loop - does the model actually answer correctly -
and compares the real billed input tokens with our local estimate.

    pip install anthropic
    export ANTHROPIC_API_KEY=...
    python -m benchmark.live_eval                       # naive vs engineered, final checkpoint
    CTX_MODEL=claude-sonnet-5 python -m benchmark.live_eval --strategies naive_truncation engineered unbounded

Cost at defaults: ~18 requests of 3-7k input tokens each on Haiku 4.5 - a few cents.
"""
from __future__ import annotations

import argparse
import os
import re

from .run_benchmark import run
from .scenario import build_scenario


def _norm(s: str) -> str:
    return re.sub(r"[\s,]", "", s.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=8_000)
    ap.add_argument("--turns", type=int, default=40)
    ap.add_argument("--strategies", nargs="+", default=["naive_truncation", "engineered"])
    a = ap.parse_args()

    import anthropic

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY to run the live evaluation (the offline benchmark needs no key).")
    client = anthropic.Anthropic()
    model = os.environ.get("CTX_MODEL", "claude-haiku-4-5-20251001")
    _, _, _, contexts = run(a.window, a.turns, verbose=False, write=False)
    questions = {q.qid: q for q in build_scenario(7, a.turns).checkpoints[a.turns]}

    print(f"model={model} window={a.window:,} turns={a.turns}\n")
    print(f"{'strategy':<20}{'question':<14}{'est tok':>9}{'real tok':>10}  correct  answer")
    score: dict[str, list[bool]] = {}
    for (label, qid, strat), ctx in contexts.items():
        if label != a.turns or strat not in a.strategies:
            continue
        system, messages = ctx.to_messages()
        resp = client.messages.create(model=model, max_tokens=200, system=system, messages=messages)
        answer = "".join(b.text for b in resp.content if b.type == "text").strip()
        ok = all(_norm(f) in _norm(answer) for f in questions[qid].must_contain)
        score.setdefault(strat, []).append(ok)
        print(f"{strat:<20}{qid:<14}{ctx.total_tokens:>9,}{resp.usage.input_tokens:>10,}  "
              f"{'yes' if ok else 'no ':<7}  {answer[:70].replace(chr(10), ' ')}")
    print()
    for strat, oks in score.items():
        print(f"{strat:<20} {sum(oks)}/{len(oks)} correct")


if __name__ == "__main__":
    main()
