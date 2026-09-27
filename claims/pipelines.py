"""Plan and run the two extraction pipelines for a given model.

``plan_*`` turns (document, model) into the list of API calls that pipeline would make,
respecting the model's context window. Offline, each call is costed with the model's own
tokenizer, and its output is costed from a *reference output* built from ground truth in
the format that pipeline asks for. Live, the same calls are sent to the provider and the
real billed usage is recorded.

Naive:     raw text of the whole pack + long prompt. If it doesn't fit the window, split the
           pack into chunks, re-send the whole prompt with each chunk, then run a merge call.
Efficient: clean -> classify -> route -> one small call per task (member / clinical /
           financial) with a shared short system prompt; results merged in code (no LLM).
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field

from . import prompts
from .ingest import Page, raw_text, render
from .models import ModelProfile

WRAP = 12  # rough per-message framing overhead added by chat formats


@dataclass
class Call:
    pipeline: str
    task: str
    system: str
    user: str
    max_output: int
    reference_output: str
    covers: list[str] = field(default_factory=list)   # evidence keys fully visible in this call
    tokens_in: int = 0
    tokens_out: int = 0

    @property
    def input_text(self) -> str:
        return self.system + "\n" + self.user


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s)


def covered_keys(text: str, evidence: dict[str, list[str]]) -> list[str]:
    t = _norm(text)
    return [k for k, ev in evidence.items() if all(_norm(e) in t for e in ev)]


# --- reference outputs --------------------------------------------------------------------------

def _subset(truth: dict, task: str) -> dict:
    out = {}
    for group, only in prompts.TASKS[task].items():
        name = group.rstrip("[]")
        val = truth[name]
        out[name] = {k: val[k] for k in only} if only else val
    return out


def _wrap_leaves(obj, evidence_lookup):
    if isinstance(obj, dict):
        return {k: _wrap_leaves(v, evidence_lookup) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_wrap_leaves(v, evidence_lookup) for v in obj]
    return {"value": obj, "source_quote": evidence_lookup(obj), "confidence": 0.95}


def naive_reference(truth: dict, evidence: dict, keys: list[str] | None = None) -> str:
    """Pretty-printed value/source_quote/confidence JSON, restricted to fields in ``keys``."""
    data = copy.deepcopy(truth)
    if keys is not None:
        data = _restrict(data, keys)
    quotes = {str(v): ev[0] for k, ev in evidence.items() for v in [k.split(".", 1)[-1]]}
    lookup = lambda v: quotes.get(str(v), str(v))
    body = json.dumps(_wrap_leaves(data, lookup), indent=2)
    return "Located member details on the claim form, clinical details in the discharge summary and " \
           "financial details across the invoices and correspondence.\n```json\n" + body + "\n```"


def _restrict(truth: dict, keys: list[str]) -> dict:
    out: dict = {}
    for k in keys:
        group, _, sub = k.partition(".")
        if group not in truth or sub == "no_superseded":
            continue
        val = truth[group]
        if isinstance(val, list):
            item = next((i for i in val if sub in i.values()), None)
            if item is not None:
                out.setdefault(group, []).append(item)
        elif sub in val:
            out.setdefault(group, {})[sub] = val[sub]
    return out


def efficient_reference(truth: dict, task: str) -> str:
    return json.dumps(_subset(truth, task), separators=(",", ":"))


# --- planning -------------------------------------------------------------------------------------

def plan_naive(raw_pages: list[Page], gt: dict, model: ModelProfile, count) -> list[Call]:
    truth, evidence = gt["expected"], gt["evidence"]
    system = prompts.naive_system()
    doc = raw_text(raw_pages)
    max_out = model.naive_max_output()
    single = prompts.naive_user(doc)
    if count(system) + count(single) + 2 * WRAP + max_out <= model.context_window:
        return [Call("naive", "all", system, single, max_out, naive_reference(truth, evidence),
                     covered_keys(single, evidence))]

    # Too big: chunk by pages; the full prompt travels with every chunk.
    overhead = count(system) + count(prompts.naive_user("", (99, 99))) + 2 * WRAP
    room = model.context_window - max_out - overhead
    if room < 200:
        raise ValueError(f"{model.label}: window too small even for the naive prompt alone")
    chunks, buf = [], []
    for p in raw_pages:
        for piece in _split_to_fit(p.text, room, count):
            if buf and count("\n\n".join(buf + [piece])) > room:
                chunks.append("\n\n".join(buf))
                buf = []
            buf.append(piece)
    if buf:
        chunks.append("\n\n".join(buf))
    calls = []
    for i, c in enumerate(chunks):
        user = prompts.naive_user(c, (i + 1, len(chunks)))
        keys = covered_keys(user, evidence)
        calls.append(Call("naive", f"chunk {i + 1}/{len(chunks)}", system, user, max_out,
                          naive_reference(truth, evidence, keys), keys))
    merge_user = prompts.naive_merge_user([c.reference_output for c in calls])
    # the merge call can only know what some chunk saw in full
    union = sorted({k for c in calls for k in c.covers})
    if count(system) + count(merge_user) + 2 * WRAP + max_out > model.context_window:
        raise ValueError(f"{model.label}: merge call does not fit the window; needs hierarchical merging")
    calls.append(Call("naive", "merge", system, merge_user, max_out,
                      naive_reference(truth, evidence, union), union))
    return calls


def _split_to_fit(text: str, room: int, count) -> list[str]:
    if count(text) <= room:
        return [text]
    out, buf = [], []
    for line in text.splitlines():
        if buf and count("\n".join(buf + [line])) > room:
            out.append("\n".join(buf))
            buf = []
        buf.append(line)
    if buf:
        out.append("\n".join(buf))
    return out


def plan_efficient(routed: dict[str, list[Page]], gt: dict, model: ModelProfile, count) -> list[Call]:
    truth, evidence = gt["expected"], gt["evidence"]
    calls = []
    for task, pages in routed.items():
        user = prompts.efficient_user(task, render(pages))
        need = count(prompts.EFFICIENT_SYSTEM) + count(user) + 2 * WRAP + model.efficient_max_output()
        if need > model.context_window:
            raise ValueError(f"{model.label}: task '{task}' needs {need} tokens > window")
        task_keys = [k for k in evidence if _task_of(k) == task]
        keys = [k for k in covered_keys(user, evidence) if k in task_keys]
        calls.append(Call("efficient", task, prompts.EFFICIENT_SYSTEM, user, model.efficient_max_output(),
                          efficient_reference(truth, task), keys))
    return calls


def _task_of(key: str) -> str:
    group, _, sub = key.partition(".")
    for task, groups in prompts.TASKS.items():
        for g, only in groups.items():
            if g.rstrip("[]") == group and (only is None or sub in only):
                return task
    return "?"


def cost_offline(calls: list[Call], count) -> list[Call]:
    for c in calls:
        c.tokens_in = count(c.system) + count(c.user) + 2 * WRAP
        c.tokens_out = count(c.reference_output)
    return calls


# --- live execution --------------------------------------------------------------------------------

def parse_json(text: str):
    m = re.search(r"```json\s*(.*?)```", text, re.S)
    candidate = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
    return json.loads(candidate)


def unwrap(obj):
    """Turn value/source_quote/confidence leaves back into plain values."""
    if isinstance(obj, dict):
        if set(obj) >= {"value"} and set(obj) <= {"value", "source_quote", "confidence"}:
            return obj.get("value")
        return {k: unwrap(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [unwrap(v) for v in obj]
    return obj


def run_live(calls: list[Call], provider, retries: int = 1) -> tuple[dict, list[dict]]:
    """Execute a planned pipeline. Returns (final extraction, per-call usage log)."""
    log, results = [], {}
    naive_partials = []
    for c in calls:
        user = c.user
        if c.task == "merge":  # rebuild the merge prompt from the *real* partial outputs
            user = prompts.naive_merge_user(naive_partials)
        attempt, parsed, text = 0, None, ""
        while attempt <= retries and parsed is None:
            msg = user if attempt == 0 else user + "\n\nYour previous reply was not valid JSON. Return valid JSON only."
            text, usage = provider.complete(c.system, msg, c.max_output)
            log.append({"pipeline": c.pipeline, "task": c.task, "attempt": attempt + 1, **usage})
            try:
                parsed = unwrap(parse_json(text))
            except (ValueError, json.JSONDecodeError):
                attempt += 1
        if c.pipeline == "naive" and c.task.startswith("chunk"):
            naive_partials.append(text)
        if parsed is not None:
            results[c.task] = parsed
    if calls[0].pipeline == "naive":
        final = results.get("merge") or results.get("all") or {}
    else:
        final = {}
        for part in results.values():  # deterministic merge in code - no extra LLM call
            for k, v in part.items():
                final[k] = {**final.get(k, {}), **v} if isinstance(v, dict) else v
    return final, log
