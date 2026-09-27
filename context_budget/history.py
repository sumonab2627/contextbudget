"""Conversation history management: tool-result clearing and compaction.

Two cheap, complementary moves from Anthropic's context-engineering guidance:

1. **Tool-result clearing** - once a tool result is a few turns old, the raw
   output has usually been digested into the agent's reply. Replace it with a
   stub that records *what* was called, so the agent can re-run it if needed.
2. **Compaction** - when history nears its budget, fold older messages into a
   running summary that keeps decisions, findings, identifiers and open
   questions, while the most recent messages stay verbatim.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace

from .tokenizer import Tokenizer


@dataclass(frozen=True)
class Message:
    role: str                 # "user" | "assistant" | "tool"
    content: str
    turn: int
    tool_call: str | None = None   # e.g. read_logs(service="claims-router") for tool messages

    def render(self) -> str:
        if self.role == "tool":
            return f"[tool result: {self.tool_call}]\n{self.content}"
        return f"{self.role.upper()}: {self.content}"


def clear_old_tool_results(history: list[Message], keep_recent: int, tok: Tokenizer) -> list[Message]:
    tool_idx = [i for i, m in enumerate(history) if m.role == "tool"]
    to_clear = set(tool_idx[:-keep_recent] if keep_recent else tool_idx)
    out = []
    for i, m in enumerate(history):
        if i in to_clear:
            stub = (f"[cleared {tok.count(m.content)}-token result of {m.tool_call}; "
                    f"key findings are in the assistant reply that followed. Re-run the tool if raw output is needed.]")
            out.append(replace(m, content=stub))
        else:
            out.append(m)
    return out


# --- Extractive summarizer (deterministic stand-in for an LLM summarizer) ----------------------

_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
_IDENT = re.compile(r"\b[A-Z]{1,5}-\d{2,}\b|\bv?\d+\.\d+(\.\d+)?\b|\b\d{1,3}(,\d{3})+\b")
_SIGNAL = re.compile(
    r"\b(root cause|decid\w*|agree\w*|confirm\w*|found|finding|next step|action|rollback|roll back|"
    r"must|owner|blocked|todo|open question|resolved|affected)\b",
    re.I,
)


class ExtractiveSummarizer:
    """Scores sentences for salience and keeps the best ones, in original order.

    Production systems use the model itself (or a cheaper model) for this; the
    point here is the *policy*: prefer the agent's and user's own conclusions
    over raw tool output, and prefer sentences carrying identifiers and decisions.
    """

    role_weight = {"user": 2.0, "assistant": 2.0, "tool": 0.3, "summary": 2.5}

    def __init__(self, tok: Tokenizer):
        self.tok = tok

    def summarize(self, previous_summary: str, messages: list[Message], budget: int) -> str:
        candidates: list[tuple[float, int, str]] = []
        order = 0
        sources = [("summary", previous_summary)] + [(m.role, m.content) for m in messages]
        for role, text in sources:
            if text.startswith("[cleared"):
                continue
            for s in _SENT.split(text):
                s = s.strip(" -•")
                if len(s) < 12:
                    continue
                score = self.role_weight.get(role, 1.0) * (
                    1 + 1.5 * len(_IDENT.findall(s)) + 2.0 * len(_SIGNAL.findall(s))
                )
                candidates.append((score, order, s))
                order += 1
        seen, chosen, used = set(), [], 0
        for score, pos, s in sorted(candidates, key=lambda c: (-c[0], c[1])):
            key = s.lower()
            cost = self.tok.count(s) + 2
            if key in seen or used + cost > budget:
                continue
            seen.add(key)
            chosen.append((pos, s))
            used += cost
        return "\n".join(f"- {s}" for _, s in sorted(chosen))


class Compactor:
    """Maintains a running summary; folds messages into it when history gets large."""

    def __init__(self, tok: Tokenizer, summary_budget: int = 600):
        self.tok = tok
        self.summary_budget = summary_budget
        self.summarizer = ExtractiveSummarizer(tok)
        self.summary = ""
        self.compacted_upto = 0          # index into full history
        self.compactions = 0

    def maybe_compact(self, source: list[Message], view: list[Message], history_budget: int,
                      trigger: float, keep_recent: int, render_tokens) -> None:
        """``view`` is what we'd send (possibly with cleared tool results) and decides *when*
        to compact; ``source`` is the original transcript and is what gets summarised - a
        summary built from cleared stubs would silently lose the facts they replaced."""
        live = view[self.compacted_upto:]
        live_tokens = render_tokens(live) + self.tok.count(self.summary)
        if live_tokens <= history_budget * trigger or len(live) <= keep_recent:
            return
        end = len(view) - keep_recent
        fold = source[self.compacted_upto:end]
        self.summary = self.summarizer.summarize(self.summary, fold, self.summary_budget)
        self.compacted_upto = end
        self.compactions += 1
