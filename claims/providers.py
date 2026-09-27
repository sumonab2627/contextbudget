"""Thin provider adapters: complete(system, user, max_output) -> (text, usage dict).

Usage is taken from each provider's billing counters, so live numbers are what you pay:
  anthropic: usage.input_tokens / output_tokens
  openai:    usage.prompt_tokens / completion_tokens (includes reasoning tokens)
  google:    usage_metadata.prompt_token_count / candidates_token_count + thoughts_token_count
  ollama:    prompt_eval_count / eval_count  (NB: Ollama silently truncates prompts longer
             than num_ctx - the naive pipeline chunks precisely to avoid that)
  mock:      returns the reference output; tokens counted locally (plumbing test, no network)
"""
from __future__ import annotations

import json
import os
import time
import urllib.request

from .models import ModelProfile


class Provider:
    def __init__(self, model: ModelProfile):
        self.model = model

    def complete(self, system: str, user: str, max_output: int) -> tuple[str, dict]:
        raise NotImplementedError


class AnthropicProvider(Provider):
    def __init__(self, model):
        super().__init__(model)
        import anthropic

        self.client = anthropic.Anthropic()

    def complete(self, system, user, max_output):
        t = time.perf_counter()
        r = self.client.messages.create(model=self.model.model_id, max_tokens=max_output, system=system,
                                        messages=[{"role": "user", "content": user}], temperature=0)
        text = "".join(b.text for b in r.content if b.type == "text")
        return text, {"tokens_in": r.usage.input_tokens, "tokens_out": r.usage.output_tokens,
                      "seconds": round(time.perf_counter() - t, 2)}


class OpenAIProvider(Provider):
    def __init__(self, model):
        super().__init__(model)
        import openai

        self.client = openai.OpenAI()

    def complete(self, system, user, max_output):
        t = time.perf_counter()
        extra = {k: v for k, v in self.model.options.items() if k == "reasoning_effort" and v}
        r = self.client.chat.completions.create(
            model=self.model.model_id, max_completion_tokens=max_output + 2_048,  # headroom for reasoning
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **extra)
        u = r.usage
        reasoning = getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", 0) or 0
        return r.choices[0].message.content or "", {
            "tokens_in": u.prompt_tokens, "tokens_out": u.completion_tokens, "reasoning_tokens": reasoning,
            "seconds": round(time.perf_counter() - t, 2)}


class GoogleProvider(Provider):
    def __init__(self, model):
        super().__init__(model)
        from google import genai

        self.client = genai.Client()

    def complete(self, system, user, max_output):
        from google.genai import types

        t = time.perf_counter()
        r = self.client.models.generate_content(
            model=self.model.model_id, contents=user,
            config=types.GenerateContentConfig(system_instruction=system, max_output_tokens=max_output + 2_048,
                                               temperature=0))
        m = r.usage_metadata
        thoughts = getattr(m, "thoughts_token_count", 0) or 0
        return r.text or "", {"tokens_in": m.prompt_token_count,
                              "tokens_out": (m.candidates_token_count or 0) + thoughts,
                              "reasoning_tokens": thoughts, "seconds": round(time.perf_counter() - t, 2)}


class OllamaProvider(Provider):
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def complete(self, system, user, max_output):
        t = time.perf_counter()
        body = {"model": self.model.model_id, "stream": False,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "options": {"num_ctx": self.model.context_window, "num_predict": max_output, "temperature": 0}}
        req = urllib.request.Request(f"{self.host}/api/chat", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as resp:
            r = json.loads(resp.read())
        return r["message"]["content"], {"tokens_in": r.get("prompt_eval_count", 0),
                                         "tokens_out": r.get("eval_count", 0),
                                         "seconds": round(time.perf_counter() - t, 2)}


class MockProvider(Provider):
    """Replays the reference output for each call - exercises the whole live path offline."""

    def __init__(self, model, calls, count):
        super().__init__(model)
        self._by_user = {c.user: c.reference_output for c in calls}
        self._last = [c.reference_output for c in calls if c.task == "merge"]
        self.count = count

    def complete(self, system, user, max_output):
        text = self._by_user.get(user) or (self._last[0] if self._last else "{}")
        return text, {"tokens_in": self.count(system) + self.count(user) + 24, "tokens_out": self.count(text)}


def provider_for(model: ModelProfile) -> Provider:
    return {"anthropic": AnthropicProvider, "openai": OpenAIProvider,
            "google": GoogleProvider, "ollama": OllamaProvider}[model.provider](model)
