"""Render the headline chart for the post from results/claims_offline.json.

    python -m claims.chart        # -> results/tokens_per_claim.png (1200x675, LinkedIn-friendly)
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
NAIVE, EFFICIENT = "#eb6834", "#2a78d6"          # validated categorical pair (light surface)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#1f1f1e", "#5f5e58", "#e6e5e0"


def main() -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.loads((RESULTS / "claims_offline.json").read_text())["rows"]
    by = {}
    for r in rows:
        by.setdefault(r["key"], {})[r["pipeline"]] = r
    order = ["gpt", "gemini", "claude", "slm"]           # top to bottom
    labels = {"gpt": "GPT-5 mini", "gemini": "Gemini 3.5 Flash-Lite", "claude": "Claude Sonnet 5 (est.)",
              "slm": "Llama 3.2 3B, local\n(8k window)"}

    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=100)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    h = 0.36
    for y, k in enumerate(order):
        n, e = by[k]["naive"], by[k]["efficient"]
        ax.barh(y - h / 2 - 0.02, n["tokens_total"], h, color=NAIVE)        # naive above (axis inverted)
        ax.barh(y + h / 2 + 0.02, e["tokens_total"], h, color=EFFICIENT)
        calls = f"  ({n['calls']} calls)" if n["calls"] > 1 else ""
        ax.text(n["tokens_total"] + 250, y - h / 2 - 0.02, f"{n['tokens_total']:,}{calls}", va="center",
                fontsize=12, color=INK2)
        save = 100 * (1 - e["tokens_total"] / n["tokens_total"])
        ax.text(e["tokens_total"] + 250, y + h / 2 + 0.02, f"{e['tokens_total']:,}   −{save:.0f}%",
                va="center", fontsize=12, color=INK, fontweight="bold")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([labels[k] for k in order],
                       fontsize=13, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 29_000)
    ax.xaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=INK2, labelsize=11)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k"))
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Tokens per claim (input + output)", color=INK2, fontsize=12)

    fig.text(0.02, 0.94, "Same claim, a quarter of the tokens", fontsize=22, fontweight="bold", color=INK)
    fig.text(0.02, 0.895, "21-page claim pack → 24 structured fields. Naive context vs engineered context, "
             "counted with each model's own tokenizer.", fontsize=12.5, color=INK2)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=NAIVE), plt.Rectangle((0, 0), 1, 1, color=EFFICIENT)],
               labels=["Naive: whole pack, long prompt, verbose output", "Engineered: cleaned, routed, compact"],
               loc="upper left", bbox_to_anchor=(0.02, 0.875), ncol=2, frameon=False, fontsize=12,
               labelcolor=INK, handlelength=1.2)
    fig.text(0.02, 0.025, "Offline token counts; Claude estimated (tokenizer not public). Synthetic, fictional claim pack. "
             "Output = size of the requested answer format.", fontsize=10, color=INK2)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.78, bottom=0.14)
    out = RESULTS / "tokens_per_claim.png"
    fig.savefig(out, facecolor=SURFACE)
    return out


if __name__ == "__main__":
    print(main())
