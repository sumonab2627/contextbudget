"""Token counting.

Every budgeting decision depends on counting tokens *before* sending a request.
Three counters are provided, all behind the same tiny interface:

* ``HeuristicTokenizer`` - dependency-free, deterministic, ~within 10-15% of real
  BPE tokenizers for English prose. Default so the demo runs anywhere.
* ``TiktokenTokenizer`` - exact for OpenAI models, a close proxy for others.
* ``AnthropicTokenizer`` - exact for Claude, via the ``count_tokens`` API
  (network call, so cache aggressively and use it to *calibrate*, not per chunk).

The absolute number matters less than using the *same* counter consistently and
keeping a safety margin (see ``BudgetPlan.safety_margin``).
"""
from __future__ import annotations

import functools
import math
import os
import re
from typing import Protocol

_WORD = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class Tokenizer(Protocol):
    name: str

    def count(self, text: str) -> int: ...


class HeuristicTokenizer:
    """Approximates BPE: roughly one token per 4 characters of a word, one per symbol."""

    name = "heuristic"

    @functools.lru_cache(maxsize=65536)
    def count(self, text: str) -> int:  # noqa: D401 - cached pure function
        if not text:
            return 0
        total = 0
        for piece in _WORD.findall(text):
            total += max(1, math.ceil(len(piece) / 4)) if piece[0].isalnum() or piece[0] == "_" else 1
        return total


class TiktokenTokenizer:
    name = "tiktoken"

    def __init__(self, encoding: str = "cl100k_base"):
        import tiktoken  # optional dependency

        self._enc = tiktoken.get_encoding(encoding)

    @functools.lru_cache(maxsize=65536)
    def count(self, text: str) -> int:
        return len(self._enc.encode(text, disallowed_special=()))


class AnthropicTokenizer:
    """Exact Claude token counts via the Messages count_tokens endpoint."""

    name = "anthropic"

    def __init__(self, model: str | None = None):
        import anthropic  # optional dependency

        self._client = anthropic.Anthropic()
        self._model = model or os.environ.get("CTX_MODEL", "claude-haiku-4-5-20251001")

    @functools.lru_cache(maxsize=4096)
    def count(self, text: str) -> int:
        if not text:
            return 0
        resp = self._client.messages.count_tokens(
            model=self._model, messages=[{"role": "user", "content": text}]
        )
        return max(0, resp.input_tokens - 7)  # subtract fixed per-message overhead (approx.)


def get_tokenizer(kind: str | None = None) -> Tokenizer:
    """Return the requested tokenizer, falling back to the heuristic one."""
    kind = (kind or os.environ.get("CTX_TOKENIZER", "heuristic")).lower()
    try:
        if kind == "tiktoken":
            return TiktokenTokenizer()
        if kind == "anthropic":
            return AnthropicTokenizer()
    except Exception as exc:  # missing package, no network, no API key...
        print(f"[tokenizer] '{kind}' unavailable ({type(exc).__name__}); using heuristic")
    return HeuristicTokenizer()
