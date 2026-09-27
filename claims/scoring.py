"""Field-level scoring of an extraction against ground truth, and Correctness Per Token.

Each evidence key in ground_truth.json is one scored field (24 in total). Matching is
lenient on formatting (case, punctuation, date format, thousands separators) and strict
on substance (codes, amounts, which invoice).
"""
from __future__ import annotations

import re
from datetime import datetime


def _s(v) -> str:
    return re.sub(r"[^a-z0-9]", "", str(v).lower().replace("dr ", "").replace("dr.", ""))


def _num(v) -> float | None:
    try:
        return float(str(v).replace(",", "").replace("HKD", "").strip())
    except (TypeError, ValueError):
        return None


def _date(v) -> str | None:
    if v is None:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(str(v).strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _eq(expected, got) -> bool:
    if isinstance(expected, bool):
        return got is expected or str(got).lower() == str(expected).lower()
    if isinstance(expected, (int, float)):
        g = _num(got)
        return g is not None and abs(g - expected) < 0.01
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(expected)):
        return _date(got) == expected
    e, g = _s(expected), _s(got)
    return bool(g) and (e == g or e in g or g in e)


def score(extraction: dict, gt: dict) -> dict[str, bool]:
    truth, results = gt["expected"], {}
    for key in gt["evidence"]:
        group, _, sub = key.partition(".")
        got = extraction.get(group) if isinstance(extraction, dict) else None
        if group == "diagnoses":
            item = next((d for d in got or [] if isinstance(d, dict) and _s(d.get("icd10")) == _s(sub)), None)
            ok = item is not None
            if ok and sub == "K35.80":
                ok = _eq(True, item.get("primary"))
        elif group == "procedures":
            ok = any(isinstance(p, dict) and _s(p.get("code")) in {_s(sub), "44970"} for p in got or [])
        elif group == "invoices" and sub == "no_superseded":
            ok = bool(got) and not any(isinstance(i, dict) and _s(i.get("invoice_number")) == _s("SF-2290") for i in got)
        elif group == "invoices":
            exp = next(i for i in truth["invoices"] if i["invoice_number"] == sub)
            ok = any(isinstance(i, dict) and _s(i.get("invoice_number")) == _s(sub) and _eq(exp["amount"], i.get("amount"))
                     for i in got or [])
        else:
            ok = isinstance(got, dict) and _eq(truth[group][sub], got.get(sub))
        results[key] = bool(ok)
    return results


def correctness_per_token(correct: int, tokens: int) -> float:
    """Correct fields per 1,000 billed tokens (input + output)."""
    return round(1000 * correct / tokens, 3) if tokens else 0.0
