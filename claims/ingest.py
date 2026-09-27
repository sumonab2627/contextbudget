"""PDF ingestion and the *pre-LLM* work that decides most of the token bill.

    read_pdf      -> raw page texts
    clean_pages   -> strip repeated headers/footers, page numbers, quoted email replies,
                     whitespace padding; drop duplicate pages
    classify      -> label each page by document type (rules here; a small classifier
                     or an SLM in production)
    route         -> give each extraction task only the page types that are its source
                     of truth

None of this calls a model. It is ordinary engineering, and it is where the largest
token savings come from.
"""
from __future__ import annotations

import re
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Page:
    number: int
    text: str
    kind: str = "unknown"
    dropped: str | None = None           # reason, if removed
    notes: list[str] = field(default_factory=list)


def read_pdf(path: Path) -> list[Page]:
    warnings.filterwarnings("ignore", module="pypdf")
    from pypdf import PdfReader

    return [Page(i + 1, p.extract_text() or "") for i, p in enumerate(PdfReader(str(path)).pages)]


_PAGE_NO = re.compile(r"^\s*page \d+ of \d+\s*$", re.I)
_DIGITS = re.compile(r"\d")


def _shape(line: str) -> str:
    return _DIGITS.sub("#", line.strip().lower())


def clean_pages(pages: list[Page], repeat_ratio: float = 0.5) -> list[Page]:
    # 1. Lines that recur on most pages (ignoring digits) are headers/footers.
    counts = Counter(s for p in pages for s in {_shape(l) for l in p.text.splitlines() if l.strip()})
    boiler = {s for s, c in counts.items() if c >= repeat_ratio * len(pages)}
    seen: dict[str, int] = {}
    out = []
    for p in pages:
        lines = []
        for l in p.text.splitlines():
            if _shape(l) in boiler or _PAGE_NO.match(l):
                continue
            if l.lstrip().startswith(">"):        # quoted email replies repeat earlier messages
                continue
            lines.append(re.sub(r"[ \t]{2,}", "  ", l.rstrip()))   # collapse table padding
        text = "\n".join(l for l in lines if l.strip())
        # 2. Duplicate pages (e.g. "DUPLICATE COPY" stamps) - compare without stamp lines.
        key = "\n".join(l for l in text.splitlines() if "duplicate" not in l.lower())
        q = Page(p.number, text)
        if key in seen:
            q.dropped = f"duplicate of page {seen[key]}"
        else:
            seen[key] = p.number
        out.append(q)
    return out


RULES = [  # first match wins; checked against the first few lines of the page
    ("fax_cover", r"fax transmission"),
    ("claim_form_declaration", r"claim form - declaration"),
    ("claim_form", r"claim form"),
    ("pre_authorisation", r"pre-authorisation confirmation"),
    ("discharge_summary", r"discharge summary"),
    ("lab_report", r"laboratory report"),
    ("radiology", r"radiology report"),
    ("invoice", r"invoice|tax invoice"),
    ("pharmacy", r"pharmacy receipt"),
    ("correspondence", r"email correspondence|^from: "),
    ("policy_schedule", r"policy schedule"),
    ("terms", r"terms and conditions"),
    ("privacy", r"privacy notice"),
]


def classify(pages: list[Page]) -> list[Page]:
    for p in pages:
        head = "\n".join(p.text.splitlines()[:3]).lower()
        p.kind = next((k for k, pat in RULES if re.search(pat, head, re.M)), "unknown")
    return pages


# Each task reads only its source-of-truth document types.
ROUTES = {
    "member": ["claim_form"],
    "clinical": ["discharge_summary"],
    "financial": ["pre_authorisation", "invoice", "pharmacy", "correspondence"],
}


BOILERPLATE_KINDS = {"terms", "privacy", "claim_form_declaration"}
_SENT = re.compile(r"(?<=[.!?])\s+")


def strip_shared_boilerplate(pages: list[Page]) -> list[Page]:
    """Remove sentences that also appear in T&C/privacy pages (e.g. legal paragraphs
    pasted into letters). Matching is whitespace-insensitive, so line wrapping doesn't matter."""
    flat = " ".join(re.sub(r"\s+", " ", p.text) for p in pages if p.kind in BOILERPLATE_KINDS)
    flat = re.sub(r"\b\d+\.\s", "", flat)
    sentences = {s.strip() for s in _SENT.split(flat) if len(s.strip()) > 40}
    patterns = [re.compile(r"\s+".join(map(re.escape, s.split()))) for s in sentences]
    for p in pages:
        if p.kind in BOILERPLATE_KINDS or p.dropped:
            continue
        text = p.text
        for pat in patterns:
            text = pat.sub("", text)
        lines = [l for l in text.splitlines() if not re.fullmatch(r"\s*(\d+\.)?\s*", l)]
        if len(lines) != len(p.text.splitlines()):
            p.notes.append("shared boilerplate removed")
        p.text = "\n".join(l.rstrip() for l in lines)
    return pages


def route(pages: list[Page]) -> dict[str, list[Page]]:
    live = [p for p in pages if not p.dropped]
    return {task: [p for p in live if p.kind in kinds] for task, kinds in ROUTES.items()}


def prepare(path: Path) -> tuple[list[Page], list[Page], dict[str, list[Page]]]:
    """Full efficient preprocessing: returns (raw pages, cleaned pages, routed pages)."""
    raw = read_pdf(path)
    cleaned = strip_shared_boilerplate(classify(clean_pages(raw)))
    return raw, cleaned, route(cleaned)


def raw_text(pages: list[Page]) -> str:
    return "\n\n".join(p.text for p in pages)


def render(pages: list[Page]) -> str:
    return "\n".join(f'<doc type="{p.kind}" page="{p.number}">\n{p.text}\n</doc>' for p in pages)
