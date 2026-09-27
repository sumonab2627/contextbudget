"""Structured note-taking: persistent memory that lives *outside* the context window.

The agent writes short notes (via a memory tool) as it works. Notes survive
compaction and even new sessions, and only the most relevant/recent ones are
loaded into each request, within a fixed token ceiling.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .retrieval import terms
from .tokenizer import Tokenizer


@dataclass(frozen=True)
class Note:
    turn: int
    text: str


class NoteStore:
    def __init__(self, path: Path | None = None):
        self.path = path
        self.notes: list[Note] = []
        if path and path.exists():
            self.notes = [Note(**n) for n in json.loads(path.read_text())]

    def add(self, turn: int, text: str) -> None:
        self.notes.append(Note(turn, text))
        if self.path:
            self.path.write_text(json.dumps([asdict(n) for n in self.notes], indent=2))

    def render(self, query: str, budget: int, tok: Tokenizer) -> str:
        """Rank by term overlap with the query, then recency; fill the budget."""
        q = set(terms(query))
        ranked = sorted(
            self.notes,
            key=lambda n: (len(q & set(terms(n.text))), n.turn),
            reverse=True,
        )
        lines, used = [], tok.count("<memory>\n</memory>")
        for n in ranked:
            line = f"- (turn {n.turn}) {n.text}"
            cost = tok.count(line) + 1
            if used + cost <= budget:
                lines.append((n.turn, line))
                used += cost
        if not lines:
            return ""
        return "<memory>\n" + "\n".join(l for _, l in sorted(lines)) + "\n</memory>"
