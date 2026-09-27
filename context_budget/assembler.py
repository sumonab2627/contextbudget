"""Context assembly strategies.

Each strategy turns (system prompt, knowledge base, history, notes, current query)
into one request payload. They differ only in *what* they choose to include:

* ``UnboundedStrategy``   - send everything. Baseline for cost; breaks past the window.
* ``NaiveTruncationStrategy`` - the common default: stuff the knowledge base up to a
  share of the window, keep history, drop the oldest messages when it overflows.
* ``EngineeredStrategy``  - budgeted slots + just-in-time retrieval + tool-result
  clearing + compaction + persistent notes. Every technique can be switched off
  individually for ablation.

Section order is deliberate: stable content first (system prompt, tool specs) so it
can be prompt-cached; volatile content last; the current question at the very end,
where models attend to it most reliably.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .budget import Allocation, BudgetPlan
from .history import Compactor, Message, clear_old_tool_results
from .memory import NoteStore
from .retrieval import BM25Index, Chunk, pack
from .tokenizer import Tokenizer


@dataclass
class Section:
    name: str
    text: str
    tokens: int


@dataclass
class AssembledContext:
    strategy: str
    sections: list[Section] = field(default_factory=list)
    budget: int | None = None

    @property
    def total_tokens(self) -> int:
        return sum(s.tokens for s in self.sections)

    @property
    def text(self) -> str:
        return "\n\n".join(s.text for s in self.sections if s.text)

    @property
    def within_budget(self) -> bool:
        return self.budget is None or self.total_tokens <= self.budget

    def breakdown(self) -> dict[str, int]:
        return {s.name: s.tokens for s in self.sections}

    def to_messages(self) -> tuple[str, list[dict]]:
        """Render as a Claude Messages API payload: (system, messages)."""
        system = next((s.text for s in self.sections if s.name == "system"), "")
        body = "\n\n".join(s.text for s in self.sections if s.name != "system" and s.text)
        return system, [{"role": "user", "content": body}]


class ContextStrategy(ABC):
    name = "base"

    def __init__(self, system_prompt: str, chunks: list[Chunk], plan: BudgetPlan, tok: Tokenizer):
        self.system_prompt = system_prompt
        self.chunks = chunks
        self.plan = plan
        self.tok = tok

    # helpers ------------------------------------------------------------------------------
    def _section(self, name: str, text: str) -> Section:
        return Section(name, text, self.tok.count(text))

    def _history_tokens(self, msgs: list[Message]) -> int:
        return sum(self.tok.count(m.render()) for m in msgs)

    def _history_section(self, msgs: list[Message], summary: str = "") -> Section:
        parts = []
        if summary:
            parts.append(f"<conversation_summary>\n{summary}\n</conversation_summary>")
        if msgs:
            parts.append("<conversation>\n" + "\n\n".join(m.render() for m in msgs) + "\n</conversation>")
        return self._section("history", "\n\n".join(parts))

    def _query_section(self, query: str) -> Section:
        return self._section("query", f"<current_request>\n{query}\n</current_request>")

    def _stuff_kb(self, budget: int) -> Section:
        """Naive pre-loading: documents in their stored order until the share is full."""
        out, used = [], 0
        for c in self.chunks:
            cost = c.tokens + 12
            if used + cost > budget:
                break
            out.append(c.render())
            used += cost
        return self._section("documents", "\n".join(out))

    @staticmethod
    def _fifo_fit(msgs: list[Message], budget: int, count) -> list[Message]:
        msgs = list(msgs)
        while msgs and count(msgs) > budget:
            msgs.pop(0)
        return msgs

    @abstractmethod
    def build(self, history: list[Message], query: str, notes: NoteStore) -> AssembledContext: ...


class UnboundedStrategy(ContextStrategy):
    name = "unbounded"

    def build(self, history, query, notes):
        ctx = AssembledContext(self.name, budget=None)
        ctx.sections.append(self._section("system", self.system_prompt))
        ctx.sections.append(self._section("documents", "\n".join(c.render() for c in self.chunks)))
        ctx.sections.append(self._history_section(history))
        ctx.sections.append(self._query_section(query))
        return ctx


class NaiveTruncationStrategy(ContextStrategy):
    name = "naive_truncation"
    kb_share = 0.40

    def build(self, history, query, notes):
        budget = self.plan.input_budget
        ctx = AssembledContext(self.name, budget=budget)
        system, q = self._section("system", self.system_prompt), self._query_section(query)
        docs = self._stuff_kb(int(budget * self.kb_share))
        room = budget - system.tokens - q.tokens - docs.tokens - 40  # wrapper tags
        msgs = self._fifo_fit(history, room, self._history_tokens)
        ctx.sections += [system, docs, self._history_section(msgs), q]
        return ctx


class EngineeredStrategy(ContextStrategy):
    def __init__(self, *args, retrieval=True, clearing=True, compaction=True, memory=True,
                 label: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_retrieval, self.use_clearing = retrieval, clearing
        self.use_compaction, self.use_memory = compaction, memory
        self.index = BM25Index(self.chunks)
        self.compactor = Compactor(self.tok)
        self.name = label or "engineered"

    def build(self, history, query, notes):
        plan = self.plan
        alloc = Allocation(plan.input_budget)
        ctx = AssembledContext(self.name, budget=plan.input_budget)

        # 1. Fixed costs first: system prompt and the question itself are non-negotiable.
        system, q = self._section("system", self.system_prompt), self._query_section(query)
        alloc.charge("system", system.tokens)
        alloc.charge("query", q.tokens)

        # 2. Persistent notes, capped.
        mem = self._section("memory", "")
        if self.use_memory:
            mem = self._section("memory", notes.render(query, alloc.grant("memory", plan.memory_max), self.tok))
        alloc.charge("memory", mem.tokens)

        # 3. Just-in-time retrieval, capped; unused allowance flows to history.
        if self.use_retrieval:
            picked = pack(self.index, query, alloc.grant("documents", plan.retrieval_max))
            docs = self._section("documents", "\n".join(c.render() for c in picked))
        else:
            docs = self._stuff_kb(alloc.grant("documents", int(plan.input_budget * NaiveTruncationStrategy.kb_share)))
        alloc.charge("documents", docs.tokens)

        # 4. History gets everything that's left.
        history_budget = alloc.remaining - 40  # wrapper tags
        hist = clear_old_tool_results(history, plan.keep_recent_tool_results, self.tok) if self.use_clearing else list(history)
        summary = ""
        if self.use_compaction:
            self.compactor.maybe_compact(list(history), hist, history_budget, plan.compaction_trigger,
                                         plan.keep_recent_messages, self._history_tokens)
            summary = self.compactor.summary
            hist = hist[self.compactor.compacted_upto:]
        summary_cost = self.tok.count(summary) + (10 if summary else 0)
        # Last resort: FIFO-drop the oldest verbatim messages if still over.
        hist = self._fifo_fit(hist, history_budget - summary_cost, self._history_tokens)
        section = self._history_section(hist, summary)
        while section.tokens > alloc.remaining and summary:  # pathological case: trim summary
            summary = "\n".join(summary.splitlines()[1:])
            section = self._history_section(hist, summary)
        alloc.charge("history", section.tokens)

        ctx.sections += [system, mem, docs, section, q]
        return ctx
