"""Model profiles: provider, context window, output limits, tokenizer and price.

Model choice changes the token bill in three separate ways, and this project measures
each one:

1. **Tokenizer** - the same text is a different number of tokens per model. Claims
   packs are full of amounts, codes and dates, and tokenizers split digits very
   differently (Gemini's splits every digit; GPT's and Llama's group them).
2. **Context window** - a small local model (an SLM served with an 8k window) cannot
   take the whole pack, so a naive prompt must be chunked and its instructions, schema
   and examples re-sent with every chunk, plus a merge call.
3. **Output behaviour** - reasoning models bill thinking tokens as output; small models
   more often return invalid JSON and need a retry. Only a live run measures this.

Prices were checked on 27 Sep 2026 where a source is given. Everything else is marked
``verify``. Edit PROFILES to match the models you use.
"""
from __future__ import annotations

import functools
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from context_budget.tokenizer import HeuristicTokenizer

TOKENIZER_DIR = Path(__file__).resolve().parent / "tokenizers"
LLAMA3_URL = "https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3/tokenizer.model"


@dataclass(frozen=True)
class ModelProfile:
    key: str
    label: str
    provider: str                 # anthropic | openai | google | ollama
    model_id: str
    size_class: str
    context_window: int
    tokenizer: str                # anthropic | o200k | gemini | llama3
    price_in: float | None = None   # USD per 1M input tokens
    price_out: float | None = None  # USD per 1M output tokens
    price_source: str = "verify"
    options: dict = field(default_factory=dict, hash=False, compare=False)

    def naive_max_output(self) -> int:
        return min(4_096, self.context_window // 4)

    def efficient_max_output(self) -> int:
        return 1_024

    def cost(self, tokens_in: int, tokens_out: int) -> float | None:
        if self.price_in is None or self.price_out is None:
            return None
        return (tokens_in * self.price_in + tokens_out * self.price_out) / 1e6


PROFILES: dict[str, ModelProfile] = {
    "claude": ModelProfile(
        "claude", "Claude Sonnet 5", "anthropic", os.environ.get("CLAUDE_MODEL", "claude-sonnet-5"),
        "frontier LLM", 200_000, "anthropic"),
    "gpt": ModelProfile(
        "gpt", "GPT-5 mini", "openai", os.environ.get("OPENAI_MODEL", "gpt-5-mini"),
        "small frontier LLM", 400_000, "o200k", 0.25, 2.00,
        "developers.openai.com/api/docs/models/gpt-5-mini (27 Sep 2026)",
        {"reasoning_effort": os.environ.get("OPENAI_REASONING_EFFORT", "minimal")}),
    "gemini": ModelProfile(
        "gemini", "Gemini 3.5 Flash-Lite", "google", os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        "small frontier LLM", 1_048_576, "gemini", 0.30, 2.50,
        "ai.google.dev/gemini-api/docs/pricing (27 Sep 2026)"),
    "slm": ModelProfile(
        "slm", "Llama 3.2 3B (Ollama, local)", "ollama", os.environ.get("OLLAMA_MODEL", "llama3.2:3b"),
        "SLM, local", int(os.environ.get("OLLAMA_NUM_CTX", 8_192)), "llama3", 0.0, 0.0,
        "local inference - no per-token fee"),
}


# --- tokenizers --------------------------------------------------------------------------------

@dataclass
class Counter:
    count: Callable[[str], int]
    name: str
    exact: bool


@functools.lru_cache(maxsize=None)
def counter_for(kind: str) -> Counter:
    heuristic = HeuristicTokenizer()
    try:
        if kind == "o200k":
            _use_bundled_tiktoken_cache()
            import tiktoken

            enc = tiktoken.get_encoding("o200k_base")
            return Counter(_cached(lambda s: len(enc.encode(s, disallowed_special=()))), "o200k_base", True)
        if kind == "llama3":
            import tiktoken
            from tiktoken.load import load_tiktoken_bpe

            path = TOKENIZER_DIR / "llama3_tokenizer.model"
            if not path.exists():
                _download(LLAMA3_URL, path)
            ranks = load_tiktoken_bpe(str(path))
            pat = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*"
                   r"|\s*[\r\n]+|\s+(?!\S)|\s+")
            enc = tiktoken.Encoding("llama3", pat_str=pat, mergeable_ranks=ranks, special_tokens={})
            return Counter(_cached(lambda s: len(enc.encode(s, disallowed_special=()))), "llama3 (128k vocab)", True)
        if kind == "gemini":
            warnings.filterwarnings("ignore", message=".*local tokenizer.*")
            from google.genai import local_tokenizer

            tok = local_tokenizer.LocalTokenizer(model_name="gemini-2.5-flash")
            return Counter(_cached(lambda s: tok.count_tokens(s).total_tokens),
                           "Gemini local tokenizer (Gemma 3 vocab)", True)
        if kind == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
            import anthropic

            client = anthropic.Anthropic()
            model = PROFILES["claude"].model_id

            def count(s: str) -> int:  # count_tokens is free; subtract the fixed message overhead
                n = client.messages.count_tokens(model=model, messages=[{"role": "user", "content": s or "."}])
                return max(0, n.input_tokens - 8)

            return Counter(_cached(count), "Anthropic count_tokens API", True)
    except Exception as exc:  # missing package / no network: fall back, but say so
        print(f"[tokenizer] {kind}: {type(exc).__name__}: {exc} - using heuristic estimate")
    return Counter(heuristic.count, "heuristic estimate", False)


def _cached(fn: Callable[[str], int]) -> Callable[[str], int]:
    return functools.lru_cache(maxsize=4096)(fn)


def _use_bundled_tiktoken_cache() -> None:
    """litellm ships the o200k/cl100k files; use them so this works offline."""
    if os.environ.get("TIKTOKEN_CACHE_DIR"):
        return
    try:
        import litellm  # noqa: F401  (only for its bundled tokenizer files)

        os.environ["TIKTOKEN_CACHE_DIR"] = str(Path(litellm.__file__).parent / "litellm_core_utils" / "tokenizers")
    except ImportError:
        pass


def _download(url: str, path: Path) -> None:
    import urllib.request

    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)
