# context-budget

Two runnable demonstrations that an engineered context window uses far fewer tokens than a
naive one, without losing the information the model needs:

1. **Claims extraction** (`claims/`): turn a messy 21-page claim pack PDF into structured JSON.
   Naive and efficient context are compared across four models: Claude, GPT, Gemini, and a
   local small language model (Llama 3.2 3B on Ollama).
2. **Long-running agent** (`context_budget/`, `benchmark/`): a 40-turn agent session under
   a fixed token budget. Covers retrieval, tool-result clearing, compaction, and memory.

---

## Part 1 - Claims extraction: naive vs efficient context, across models

```bash
pip install -r requirements.txt
python -m claims.generate                    # builds data/claim_pack.pdf + ground_truth.json
python -m claims.run                         # offline: exact token counts with each model's tokenizer
python -m claims.run --mock                  # runs the full live path with a replay provider
python -m claims.run --live gpt gemini slm   # real calls (needs keys / Ollama running)
python -m claims.run --live claude           # ANTHROPIC_API_KEY also makes offline Claude counts exact
python -m claims.chart                       # results/tokens_per_claim.png, for the post
```

Environment variables: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, and optionally
`CLAUDE_MODEL`, `OPENAI_MODEL`, `GEMINI_MODEL`, `OLLAMA_MODEL`, `OLLAMA_NUM_CTX` (default 8192),
`OLLAMA_HOST`. For the SLM, run `ollama pull llama3.2:3b` first.

### The document

`claims/generate.py` builds a fictional international health-insurance claim pack. It is messy
in the ways real packs are:
- a fax cover sheet, plus headers, footers and page numbers on every page
- a claim form whose declared admission date is wrong
- a discharge summary, alongside lab and radiology pages the extraction doesn't need
- an itemised hospital invoice, plus a "DUPLICATE COPY" of its first page
- a surgeon's invoice that was re-issued, where only the email thread says which version is current
- quoted email replies
- a policy schedule, three pages of terms and conditions, and a privacy notice

Ground truth has 24 scored fields, each tied to the exact evidence text it depends on.

### The two pipelines

| | Naive | Efficient |
|---|---|---|
| Input | Raw text of all 21 pages | Headers, footers, duplicates, quoted replies and shared boilerplate removed; pages classified |
| Routing | Everything, every time | Each task gets only its source-of-truth pages: member → claim form; clinical → discharge summary; financial → pre-auth, invoices, pharmacy, email |
| Prompt | ~650-token instructions + full JSON Schema + worked example | ~70-token shared system prompt (cacheable) + compact per-task schema |
| Output | Pretty JSON with value, source quote and confidence per field | Minified JSON, values only |
| Small window | Chunk, re-send the whole prompt with every chunk, then an LLM merge call | Each task fits; results merged in code |

### Results (offline, `results/claims_offline.txt`)

| Model | Pipeline | Calls | Input tokens | Output tokens* | Total | Saving | Fields with evidence |
|---|---|---:|---:|---:|---:|---:|---:|
| Claude Sonnet 5 (estimate) | naive | 1 | 12,814 | 2,000 | 14,814 | | 24/24 |
| | efficient | 3 | 2,937 | 624 | 3,561 | 76% | 24/24 |
| GPT-5 mini | naive | 1 | 10,725 | 1,799 | 12,524 | | 24/24 |
| | efficient | 3 | 2,757 | 386 | 3,143 | 75% | 24/24 |
| Gemini 3.5 Flash-Lite | naive | 1 | 12,226 | 2,242 | 14,468 | | 24/24 |
| | efficient | 3 | 3,200 | 463 | 3,663 | 75% | 24/24 |
| Llama 3.2 3B, 8k window (SLM) | naive | **4** | 20,611 | 3,792 | 24,403 | | **23/24** |
| | efficient | 3 | 2,792 | 387 | 3,179 | **87%** | 24/24 |

\*Offline output is the size of the reference answer in each pipeline's requested format. It
leaves out reasoning and chatter, so it *understates* the naive pipeline's real output.

At list prices (checked 27 Sep 2026), 100,000 claims cost **$628 → $146** on GPT-5 mini and
**$927 → $212** on Gemini 3.5 Flash-Lite.

### How the model choice changes the token count

1. **Tokenizer.** The same cleaned pages are 2,406 tokens for GPT, 2,444 for Llama and 2,810
   for Gemini. On the invoice pages Gemini's count is 1.19× GPT's, because its tokenizer splits
   every digit and claims are full of amounts, codes and dates. On clinical prose the gap is
   smaller. Compare models on your own documents, not on a benchmark's prose.
2. **Context window.** Served locally with an 8k window, the SLM can't take the naive prompt
   in one call. It needs 3 chunks, each re-sending about 3.4k tokens of instructions, schema and
   example, plus a merge call. That's 20.6k input tokens against 10.7k for GPT on the same
   pipeline. Worse, the pre-auth letter and the invoices end up in different chunks, so no call
   can check whether the claim exceeds its pre-authorisation. With the efficient pipeline, the
   same 3B model needs fewer tokens than any of the frontier models.
3. **Output behaviour.** Reasoning models bill thinking tokens as output, and small models more
   often need a JSON retry. Only `--live` measures this. It reports reasoning tokens, retries,
   and true **Correctness Per Token** (correct fields per 1,000 billed tokens).

### Caveats

- "Fields with evidence" means the facts were visible to the call that produces them. That's
  necessary but not sufficient. Accuracy needs `--live`, which was not run when these numbers
  were produced (no API keys in this environment).
- Claude's tokenizer is not public. Offline Claude numbers are estimates until
  `ANTHROPIC_API_KEY` is set, which switches to the free `count_tokens` API. Gemini counts use
  Google's local tokenizer (the Gemini 2.x vocabulary) as a stand-in for 3.x.
- Page classification here is rule-based and tuned to this pack. In production, use a trained
  classifier or an SLM, and monitor misroutes: a page sent to the wrong task is a field the
  model never sees. Routing is where the efficient pipeline can fail silently.
- An 8k window for the SLM is a deployment choice (RAM, speed). Set `OLLAMA_NUM_CTX=32768` and
  the naive pipeline fits in one call. The prompt-overhead lesson still applies to longer packs.
- "$0" for local inference isn't free: prefill time grows with input tokens, so fewer tokens
  still means faster and cheaper hardware. `--live` records seconds per call.

---

## Part 2 - Long-running agent under a token budget

It puts into code the ideas in Anthropic's engineering post
**[Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)**
(Sept 2025): treat context as a finite *attention budget*, and aim for the smallest set of
high-signal tokens that gets you the result you want. The companion
[Claude Cookbook: memory, compaction and tool clearing](https://platform.claude.com/cookbook/tool-use-context-engineering-context-engineering-tools)
shows the same primitives as first-party API features.

### The techniques and where they live

| Technique (from the article) | What it does here | Module |
|---|---|---|
| Count before you send | Pluggable counter: heuristic (default), tiktoken, or Claude's `count_tokens` API | `tokenizer.py` |
| Budget slots | Fixed costs first (system, question, output reserve, safety margin); elastic slots are capped, and anything they don't use goes to history | `budget.py` |
| Just-in-time retrieval | The knowledge base stays *outside* the window. BM25 ranks chunks, a greedy packer fills the slot, and weak matches are skipped | `retrieval.py` |
| Tool-result clearing | Old raw tool output becomes a stub that records the call, so it can be re-fetched | `history.py` |
| Compaction | Older turns are folded into a running summary that keeps decisions and identifiers; recent turns stay verbatim | `history.py` |
| Structured note-taking | Persistent notes outside the window, ranked by relevance and recency, loaded within a cap. They survive compaction and new sessions | `memory.py` |
| Ordering for caching and attention | Stable prefix first (cacheable), volatile content next, question last | `assembler.py` |

### Run it

```bash
python -m benchmark.run_benchmark                        # 8k window, 40-turn session
python -m benchmark.run_benchmark --window 16000 --turns 80
python -m pytest -q                                      # 25 tests across both parts

# Optional: have a real model answer, and compare billed tokens with the estimate
pip install anthropic && export ANTHROPIC_API_KEY=...
python -m benchmark.live_eval
```

Part 2 needs Python 3.10+ and nothing else.

### The experiment

`benchmark/scenario.py` generates a seeded, synthetic 40-turn **incident investigation**. An agent
debugs a stuck claims pipeline and reads ~1,100-token log dumps each turn, next to a knowledge base
of 40 near-identical claims-policy documents (~13k tokens). Facts are planted at known turns: a
root-cause code, a user decision, an agent note, a recent metric, KB limits, and one figure that
only ever appears in raw tool output. At turn 12 and at the end, each strategy builds the context
for a question, and we check whether **every fact needed is in the context and the request fits
the window**.

#### Results (8,000-token window, 40 turns; `results/benchmark_output.txt`)

| strategy | total input tokens | mean / request | requests over window | answerable | new session | compactions |
|---|---:|---:|---:|---:|---:|---:|
| unbounded (send everything) | 1,494,056 | 37,351 | **40 / 40** | 0 / 15 | 0 / 3 | 0 |
| naive truncation | 232,403 | 5,810 | 0 | 3 / 15 | 0 / 3 | 0 |
| **engineered** | **152,585** | **3,815** | 0 | **15 / 15** | **2 / 3** | 3 |
| engineered without retrieval | 235,620 | 5,890 | 0 | 9 / 15 | 2 / 3 | 38 |
| engineered without tool clearing | 147,756 | 3,694 | 0 | 15 / 15 | 2 / 3 | 19 |
| engineered without compaction | 197,864 | 4,947 | 0 | 12 / 15 | 2 / 3 | 0 |
| engineered without memory | 152,500 | 3,812 | 0 | 15 / 15 | 0 / 3 | 3 |

**Compared with naive truncation, the engineered context uses 34% fewer tokens and has every
needed fact for 15 of 15 questions instead of 3.** Sending everything grows past 60k tokens and
would be rejected from the first request.

#### What each ablation shows

- **Retrieval is the biggest single win.** Pre-loading the KB takes 40% of the window and still
  holds only ~10 of 40 documents. Retrieval spends about 330 tokens per KB question, and none at
  all when the question isn't about policy.
- **Compaction protects what the user decided.** Without it, FIFO truncation drops the turn-4
  rollback decision ("v3.8.2, Priya owns it"), which was never written to a note.
- **Tool clearing is about how often you compact.** Recall is the same without it, but history
  fills 6× faster, which means 19 compactions instead of 3. In a real system each compaction is
  an extra LLM call, and each one loses a little detail.
- **Memory is for crossing sessions.** Within a session, compaction covers for it. In a fresh
  session only the note store carries over, so `-memory` scores 0 / 3.

#### The limits, stated plainly

- **Facts that exist only in raw tool output are fragile.** The `p99` figure survives at 8k, but
  in the 16k / 80-turn run the summariser drops it, and at turn 12 it's already a cleared stub.
  The fix is behavioural, not a bigger window: have the agent *state* findings or `remember` them,
  and let it re-run a tool when it needs raw output again (the stub tells it how).
- **Every strategy misses the rollback decision in a new session**, because nobody wrote it down.
  The lesson: note decisions, not just findings.
- **"Answerable" means the facts are present, not that the answer is right.** Having the facts
  is necessary but not sufficient. `live_eval.py` checks real answers from Claude. It was not run
  when these numbers were produced (no API key), so treat that part as unverified.
- **The summariser is extractive and deterministic**, standing in for an LLM summariser so the
  benchmark is reproducible. It scores sentences for identifiers and decision words, and weights
  the user's and agent's statements above raw tool output. A model-based summariser plugs in at
  `Compactor.summarizer`.
- **The heuristic tokenizer** counts roughly one token per four characters of a word. Relative
  comparisons hold because every strategy uses the same counter. Use `--tokenizer anthropic` for
  exact Claude counts.
- The scenario is synthetic and was designed to exercise these techniques. Numbers from a real
  workload will differ; the direction of the effects is the point.

### Using it in your own agent

```python
from context_budget import BudgetPlan, EngineeredStrategy, NoteStore, chunk_document, get_tokenizer

tok = get_tokenizer("anthropic")                       # or "heuristic" / "tiktoken"
plan = BudgetPlan(context_window=32_000, output_reserve=4_000, retrieval_max=6_000)
chunks = [c for doc in my_docs for c in chunk_document(doc.id, doc.title, doc.text, tok)]
strategy = EngineeredStrategy(SYSTEM_PROMPT, chunks, plan=plan, tok=tok)
notes = NoteStore(Path("agent_notes.json"))            # persists across sessions

ctx = strategy.build(history, user_message, notes)     # history: list[Message]
system, messages = ctx.to_messages()                   # ready for client.messages.create(...)
print(ctx.breakdown())                                 # tokens per slot, for logging
```
