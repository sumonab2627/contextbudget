import pytest

from benchmark.run_benchmark import run
from benchmark.scenario import build_scenario
from context_budget import (Allocation, BM25Index, BudgetPlan, Compactor, HeuristicTokenizer, Message,
                            NoteStore, chunk_document, clear_old_tool_results, pack)

tok = HeuristicTokenizer()


@pytest.fixture(scope="module")
def index():
    sc = build_scenario()
    return BM25Index([c for d in sc.documents for c in chunk_document(*d, tok)])


@pytest.mark.parametrize("window,turns", [(4_000, 30), (8_000, 40), (16_000, 80)])
def test_budgeted_strategies_never_exceed_window(window, turns):
    _, per_request, checks, _ = run(window, turns, verbose=False, write=False)
    budget = BudgetPlan(context_window=window).input_budget
    for r in per_request + checks:
        if r["strategy"] != "unbounded":
            assert r["tokens"] <= budget, r


def test_engineered_beats_naive_on_cost_and_recall():
    summary, *_ = run(8_000, 40, verbose=False, write=False)
    by = {s["strategy"]: s for s in summary}
    eng, naive = by["engineered"], by["naive_truncation"]
    assert eng["total_input_tokens"] < 0.75 * naive["total_input_tokens"]
    assert eng["answerable"] >= naive["answerable"] + 8
    assert by["unbounded"]["requests_over_window"] > 0  # sending everything doesn't fit


def test_retrieval_finds_right_document(index):
    hits = pack(index, "auto-approval limit for dental claims submitted in Hong Kong", 1_800)
    assert hits and hits[0].doc_id.startswith("kb-dent-hon")


def test_retrieval_skips_irrelevant_queries(index):
    assert pack(index, "What p99 latency did the payments-gateway logs report?", 1_800) == []


def test_retrieval_respects_budget(index):
    hits = pack(index, "claims limit policy reference", 400)
    assert sum(c.tokens + 12 for c in hits) <= 400


def test_tool_clearing_keeps_only_recent_results():
    h = [Message("tool", "x " * 500, i, tool_call=f"t{i}") for i in range(5)]
    out = clear_old_tool_results(h, 2, tok)
    assert [m.content.startswith("[cleared") for m in out] == [True, True, True, False, False]
    assert "t0" in out[0].content  # the stub says how to re-fetch


def test_compaction_keeps_decisions_and_shrinks_history():
    sc = build_scenario()
    hist = []
    for t in sc.turns[:12]:
        hist.append(Message("user", t.user, t.n))
        if t.tool_call:
            hist.append(Message("tool", t.tool_output, t.n, tool_call=t.tool_call))
        hist.append(Message("assistant", t.assistant, t.n))
    count = lambda ms: sum(tok.count(m.render()) for m in ms)
    c = Compactor(tok, summary_budget=400)
    c.maybe_compact(hist, hist, history_budget=3_000, trigger=0.75, keep_recent=4, render_tokens=count)
    assert c.compactions == 1
    assert tok.count(c.summary) <= 400
    for fact in ["E-4471", "v3.8.2", "Priya", "B-20931"]:
        assert fact in c.summary


def test_allocation_refuses_overspend():
    a = Allocation(100)
    a.charge("system", 80)
    assert a.grant("docs", 50) == 20
    with pytest.raises(OverflowError):
        a.charge("docs", 21)


def test_notes_persist_across_sessions(tmp_path):
    p = tmp_path / "notes.json"
    NoteStore(p).add(3, "Rollback target is v3.8.2")
    rendered = NoteStore(p).render("rollback version", 200, tok)
    assert "v3.8.2" in rendered


def test_notes_respect_budget():
    s = NoteStore()
    for i in range(50):
        s.add(i, f"note number {i} about service-{i} with some detail")
    assert tok.count(s.render("service", 120, tok)) <= 120
