"""Just-in-time retrieval: chunk, rank with BM25, pack greedily into a token budget.

Instead of pre-loading a knowledge base into the prompt, we keep it outside the
window and pull in only the chunks relevant to the current request. BM25 is
used so the demo has no dependencies; swap in embeddings or hybrid search in
production - the packing logic is identical.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .tokenizer import Tokenizer

_TERM = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was "
    "were what which with will who how does do our we you your i".split()
)


def terms(text: str) -> list[str]:
    return [t for t in _TERM.findall(text.lower()) if t not in _STOP]


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    title: str
    text: str
    tokens: int

    def render(self) -> str:
        return f'<document id="{self.doc_id}" title="{self.title}">\n{self.text}\n</document>'


def chunk_document(doc_id: str, title: str, text: str, tok: Tokenizer, max_tokens: int = 350) -> list[Chunk]:
    """Split on paragraph boundaries, keeping each chunk under ``max_tokens``."""
    chunks, buf = [], []
    for para in [p.strip() for p in text.split("\n\n") if p.strip()]:
        candidate = "\n\n".join(buf + [para])
        if buf and tok.count(candidate) > max_tokens:
            body = "\n\n".join(buf)
            chunks.append(Chunk(f"{doc_id}#{len(chunks)}", title, body, tok.count(body)))
            buf = [para]
        else:
            buf.append(para)
    if buf:
        body = "\n\n".join(buf)
        chunks.append(Chunk(f"{doc_id}#{len(chunks)}", title, body, tok.count(body)))
    return chunks


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.4, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self._tf = [Counter(terms(c.title + " " + c.text)) for c in chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = sum(self._len) / max(1, len(self._len))
        df = Counter(t for tf in self._tf for t in tf)
        n = len(chunks)
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str) -> list[tuple[float, Chunk]]:
        q = terms(query)
        scored = []
        for i, tf in enumerate(self._tf):
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if f:
                    norm = f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self._len[i] / self._avg))
                    s += self._idf.get(t, 0.0) * norm
            if s > 0:
                scored.append((s, self.chunks[i]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored


def pack(index: BM25Index, query: str, budget: int, min_relative_score: float = 0.6,
         max_chunks: int = 4, min_score: float = 2.0) -> list[Chunk]:
    """Greedy knapsack: take the best chunks that fit.

    ``min_score`` skips retrieval entirely when nothing is a real match (e.g. a
    question about logs, not policy). ``min_relative_score`` drops weak matches
    (anything below 60% of the best hit). Filling the slot with marginal chunks
    adds noise and cost - the unused allowance goes back to conversation history.
    """
    results = index.search(query)
    if not results or results[0][0] < min_score:
        return []
    top = results[0][0]
    picked, used = [], 0
    for score, chunk in results:
        if score < top * min_relative_score or len(picked) >= max_chunks:
            break
        if used + chunk.tokens + 12 <= budget:  # +12 for the <document> wrapper
            picked.append(chunk)
            used += chunk.tokens + 12
    return picked
