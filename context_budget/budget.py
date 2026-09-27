"""Token budget planning.

A request's context window is split into *slots*. Fixed slots (system prompt,
tool definitions, output reserve) are paid first. Elastic slots (memory,
retrieved documents) get a ceiling, and whatever they don't use flows down to
conversation history - so a request with no relevant documents doesn't waste
its retrieval allowance.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BudgetPlan:
    context_window: int = 8_000          # hard limit of the model / your cost cap
    output_reserve: int = 1_000          # tokens kept free for the model's answer
    safety_margin: float = 0.05          # headroom for tokenizer estimation error
    memory_max: int = 500                # persistent notes
    retrieval_max: int = 1_800           # just-in-time retrieved documents
    compaction_trigger: float = 0.75     # compact when history uses >75% of its share
    keep_recent_messages: int = 6        # never compact the most recent N messages
    keep_recent_tool_results: int = 2    # older tool results are cleared to stubs

    @property
    def input_budget(self) -> int:
        """Tokens available for everything we send."""
        usable = int(self.context_window * (1 - self.safety_margin))
        return usable - self.output_reserve


@dataclass
class Allocation:
    """Tracks spend against the input budget, slot by slot."""

    total: int
    spent: dict[str, int] = field(default_factory=dict)

    @property
    def remaining(self) -> int:
        return self.total - sum(self.spent.values())

    def grant(self, slot: str, wanted_max: int) -> int:
        """Largest amount a slot may use: its ceiling or what's left, whichever is smaller."""
        return max(0, min(wanted_max, self.remaining))

    def charge(self, slot: str, tokens: int) -> None:
        if tokens > self.remaining:
            raise OverflowError(f"slot '{slot}' needs {tokens}, only {self.remaining} left")
        self.spent[slot] = self.spent.get(slot, 0) + tokens
