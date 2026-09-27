import copy
import re

import pytest

from claims.models import PROFILES, counter_for
from claims.pipelines import plan_efficient, plan_naive
from claims.providers import MockProvider
from claims.pipelines import run_live, cost_offline
from claims.run import load
from claims.scoring import score


@pytest.fixture(scope="module")
def pack():
    return load()  # (gt, raw, cleaned, routed)


def _n(s):
    return re.sub(r"\s+", " ", s)


def test_evidence_survives_pdf_extraction(pack):
    gt, raw, _, _ = pack
    text = _n("\n".join(p.text for p in raw))
    assert all(_n(e) in text for ev in gt["evidence"].values() for e in ev)


def test_cleaning_drops_noise_but_keeps_sources(pack):
    _, raw, cleaned, routed = pack
    dup = [p for p in cleaned if p.dropped]
    assert len(dup) == 1 and "duplicate" in dup[0].dropped
    routed_kinds = {p.kind for v in routed.values() for p in v}
    assert not routed_kinds & {"terms", "privacy", "lab_report", "radiology", "fax_cover"}
    assert all("Received 22/08/2026" not in p.text for p in cleaned)       # header stripped
    corr = next(p for p in cleaned if p.kind == "correspondence")
    assert "SF-2291 replaces SF-2290" in _n(corr.text) and ">" not in corr.text  # quoted replies stripped


@pytest.mark.parametrize("key", list(PROFILES))
def test_every_call_fits_the_model_window(pack, key):
    gt, raw, _, routed = pack
    m, c = PROFILES[key], counter_for(PROFILES[key].tokenizer).count
    for calls in (plan_naive(raw, gt, m, c), plan_efficient(routed, gt, m, c)):
        for call in cost_offline(calls, c):
            assert call.tokens_in + call.max_output <= m.context_window


@pytest.mark.parametrize("key", list(PROFILES))
def test_efficient_uses_far_fewer_tokens_without_losing_evidence(pack, key):
    gt, raw, _, routed = pack
    m, c = PROFILES[key], counter_for(PROFILES[key].tokenizer).count
    naive = cost_offline(plan_naive(raw, gt, m, c), c)
    eff = cost_offline(plan_efficient(routed, gt, m, c), c)
    total = lambda cs: sum(x.tokens_in + x.tokens_out for x in cs)
    assert total(eff) < 0.35 * total(naive)
    assert {k for x in eff for k in x.covers} == set(gt["evidence"])


def test_small_window_forces_chunking_and_splits_evidence(pack):
    gt, raw, _, routed = pack
    m, c = PROFILES["slm"], counter_for("llama3").count
    naive = plan_naive(raw, gt, m, c)
    assert len(naive) >= 3 and naive[-1].task == "merge"
    # pre-auth letter and invoices land in different chunks: no single call can compare them
    assert "totals.exceeds_pre_authorisation" not in naive[-1].covers


def test_mock_live_run_scores_perfectly_for_efficient(pack):
    gt, raw, _, routed = pack
    m, c = PROFILES["gpt"], counter_for("o200k").count
    calls = plan_efficient(routed, gt, m, c)
    final, log = run_live(calls, MockProvider(m, calls, c))
    assert all(score(final, gt).values()) and len(log) == 3


def test_scoring_penalises_superseded_invoice_and_wrong_date(pack):
    gt = pack[0]
    bad = copy.deepcopy(gt["expected"])
    bad["invoices"].append({"invoice_number": "SF-2290", "amount": 42000})
    bad["treatment"]["admission_date"] = "13/08/2026"     # the member's declared (wrong) date
    s = score(bad, gt)
    assert not s["invoices.no_superseded"] and not s["treatment.admission_date"]
    good = copy.deepcopy(gt["expected"])
    good["treatment"]["discharge_date"] = "18 August 2026"   # format-tolerant
    assert all(score(good, gt).values())
