"""Prompts and schemas for the two approaches.

The *naive* prompt is typical of what ends up in production after a few iterations:
long instructions, a full JSON Schema with descriptions, a worked example, and a
request for a source quote and confidence per field. Every one of those choices is
defensible on its own. Together they cost tokens on every call, and again on every
chunk when the document doesn't fit.

The *efficient* prompt keeps one short, stable system prompt (cacheable across
tasks) and gives each task only its slice of the schema, in compact form, asking
for minified JSON.
"""
from __future__ import annotations

import json

# field -> (type, description). Single source of truth for both schema styles.
FIELDS: dict[str, dict[str, tuple[str, str]]] = {
    "member": {
        "name": ("string", "Full name of the insured member exactly as written on the claim form."),
        "member_id": ("string", "The member identifier issued by the insurer, usually in the format XXX-NNNNNNN-NN."),
        "policy_number": ("string", "The policy or contract number under which the member is covered."),
        "plan": ("string", "The commercial name of the plan or product the member holds."),
    },
    "treatment": {
        "country": ("string", "Country where treatment took place, as a full country or territory name."),
        "hospital": ("string", "Name of the hospital or facility where the member was treated."),
        "admission_date": ("string", "Date of admission in ISO 8601 format (YYYY-MM-DD) from the hospital record."),
        "discharge_date": ("string", "Date of discharge in ISO 8601 format (YYYY-MM-DD)."),
        "attending_physician": ("string", "Name of the attending or responsible physician, including title."),
    },
    "diagnoses[]": {
        "icd10": ("string", "ICD-10 diagnosis code including the decimal point, e.g. K35.80."),
        "description": ("string", "The diagnosis description as written in the clinical record."),
        "primary": ("boolean", "True if this is the primary diagnosis for the admission, otherwise false."),
    },
    "procedures[]": {
        "code": ("string", "Procedure code (ICD-10-PCS preferred, otherwise CPT)."),
        "description": ("string", "Description of the procedure performed."),
        "date": ("string", "Date the procedure was performed in ISO 8601 format (YYYY-MM-DD)."),
    },
    "pre_authorisation": {
        "reference": ("string", "The pre-authorisation or guarantee-of-payment reference number."),
        "approved_amount": ("number", "Approved amount as a number without currency symbols or separators."),
        "status": ("string", "Status of the pre-authorisation, e.g. Approved, Declined, Pending."),
    },
    "invoices[]": {
        "provider": ("string", "Name of the provider that issued the invoice."),
        "invoice_number": ("string", "The invoice or receipt number."),
        "date": ("string", "Invoice date in ISO 8601 format (YYYY-MM-DD)."),
        "amount": ("number", "Final amount due on the invoice after discounts, as a number."),
    },
    "totals": {
        "currency": ("string", "ISO 4217 currency code of the billed amounts."),
        "total_billed": ("number", "Sum of the amounts of all current (not superseded or duplicate) invoices."),
        "exceeds_pre_authorisation": ("boolean", "True if total_billed is greater than the pre-authorised amount."),
    },
}

TASKS = {  # which schema groups (and which treatment fields) each efficient task extracts
    "member": {"member": None, "treatment": ["country"]},
    "clinical": {"treatment": ["hospital", "admission_date", "discharge_date", "attending_physician"],
                 "diagnoses[]": None, "procedures[]": None},
    "financial": {"pre_authorisation": None, "invoices[]": None, "totals": None},
}

# ---------------------------------------------------------------------------------------------
# Naive
# ---------------------------------------------------------------------------------------------

NAIVE_INSTRUCTIONS = """You are an expert medical claims adjudication assistant working for a large international health insurer. Your job is to read the complete claim submission provided below and extract all of the information required to register and assess the claim, returning it as structured JSON that conforms exactly to the JSON Schema provided.

Please follow these rules carefully:
1. Read the entire document before extracting anything. Claim packs often contain several documents: claim forms, pre-authorisation letters, clinical summaries, laboratory and radiology reports, invoices, receipts, correspondence, policy schedules and terms and conditions.
2. Extract every field defined in the schema. If a field genuinely cannot be found, set its value to null. Never invent or guess values.
3. For every field, return an object with three properties: "value" (the extracted value), "source_quote" (the exact text from the document that supports the value) and "confidence" (a number between 0 and 1 expressing how confident you are).
4. Dates must be converted to ISO 8601 format (YYYY-MM-DD). Where different documents give different dates, prefer the hospital's clinical record over forms completed by the member.
5. Monetary amounts must be returned as numbers without currency symbols or thousands separators. Record the currency separately.
6. Diagnoses must use ICD-10 codes as written in the clinical documents. Identify which diagnosis is primary.
7. Invoices: include every current invoice or receipt. Exclude duplicate copies and any invoice that has been superseded, replaced or cancelled according to the correspondence.
8. Calculate total_billed as the sum of all current invoices and compare it with the pre-authorised amount.
9. Ignore boilerplate such as terms and conditions, privacy notices and fax cover sheets unless they contain claim-specific information.
10. Before producing the JSON, briefly explain how you located each group of fields and any conflicts you resolved. Then output the JSON inside a ```json code block, pretty-printed with two-space indentation.

Accuracy is critical: the output will be used to make payment decisions."""


def verbose_json_schema() -> str:
    """A reasonably well-factored schema: the value/quote/confidence wrapper is defined once in $defs."""
    defs = {f"{t}_field": {"type": "object", "properties": {
        "value": {"type": [t, "null"]},
        "source_quote": {"type": ["string", "null"], "description": "Exact supporting text from the document."},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1}},
        "required": ["value", "source_quote", "confidence"]} for t in ("string", "number", "boolean")}
    props: dict = {}
    for group, fields in FIELDS.items():
        name = group.rstrip("[]")
        obj = {"type": "object", "properties": {
            k: {"$ref": f"#/$defs/{t}_field", "description": d} for k, (t, d) in fields.items()},
            "required": list(fields)}
        props[name] = {"type": "array", "items": obj} if group.endswith("[]") else obj
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": "InsuranceClaimExtraction",
              "type": "object", "$defs": defs, "properties": props, "required": list(props)}
    return json.dumps(schema, indent=2)


FEW_SHOT = """EXAMPLE INPUT (excerpt from a different claim):
CLAIM FORM - OUTPATIENT
Member name: Sofia Marquez   Member ID: MGH-5510238-02   Policy number: IPMI-ES-7712094
Plan: Global Essential   Country of treatment: Spain
INVOICE  Clinica Dental Sol, Valencia   Invoice number: CDS-10442   Date: 02/03/2026
Root canal treatment, tooth 36 ................ EUR 480.00
Total due: EUR 480.00

EXAMPLE OUTPUT:
```json
{
  "member": {
    "name": {"value": "Sofia Marquez", "source_quote": "Member name: Sofia Marquez", "confidence": 0.99},
    "member_id": {"value": "MGH-5510238-02", "source_quote": "Member ID: MGH-5510238-02", "confidence": 0.99},
    "policy_number": {"value": "IPMI-ES-7712094", "source_quote": "Policy number: IPMI-ES-7712094", "confidence": 0.99},
    "plan": {"value": "Global Essential", "source_quote": "Plan: Global Essential", "confidence": 0.98}
  },
  "invoices": [
    {
      "provider": {"value": "Clinica Dental Sol", "source_quote": "Clinica Dental Sol, Valencia", "confidence": 0.97},
      "invoice_number": {"value": "CDS-10442", "source_quote": "Invoice number: CDS-10442", "confidence": 0.99},
      "date": {"value": "2026-03-02", "source_quote": "Date: 02/03/2026", "confidence": 0.95},
      "amount": {"value": 480.0, "source_quote": "Total due: EUR 480.00", "confidence": 0.99}
    }
  ],
  "totals": {
    "currency": {"value": "EUR", "source_quote": "EUR 480.00", "confidence": 0.99},
    "total_billed": {"value": 480.0, "source_quote": "Total due: EUR 480.00", "confidence": 0.97},
    "exceeds_pre_authorisation": {"value": null, "source_quote": null, "confidence": 0.5}
  }
}
```"""


def naive_system() -> str:
    return NAIVE_INSTRUCTIONS


def naive_user(document: str, part: tuple[int, int] | None = None) -> str:
    head = f"JSON SCHEMA:\n{verbose_json_schema()}\n\n{FEW_SHOT}\n\n"
    if part:
        head += (f"NOTE: The document is too long for a single request. This is PART {part[0]} OF {part[1]}. "
                 "Extract whatever fields appear in this part; use null for anything not present.\n\n")
    return head + f"CLAIM DOCUMENT:\n{document}"


def naive_merge_user(partials: list[str]) -> str:
    body = "\n\n".join(f"PARTIAL RESULT {i + 1}:\n{p}" for i, p in enumerate(partials))
    return (f"JSON SCHEMA:\n{verbose_json_schema()}\n\nThe claim document was processed in {len(partials)} parts. "
            "Merge the partial results below into one final result that conforms to the schema. Resolve "
            f"conflicts using the rules above.\n\n{body}")


# ---------------------------------------------------------------------------------------------
# Efficient
# ---------------------------------------------------------------------------------------------

EFFICIENT_SYSTEM = ("Extract fields from health-insurance claim documents. Return only minified JSON matching the "
                    "schema; no prose. Use null if absent. Dates YYYY-MM-DD. Amounts as numbers. Prefer clinical "
                    "records for clinical dates. Exclude duplicate or superseded invoices.")


def compact_schema(task: str) -> str:
    parts = []
    for group, only in TASKS[task].items():
        fields = FIELDS[group]
        keys = only or list(fields)
        body = ",".join(f"{k}:{fields[k][0][:3] if fields[k][0] != 'boolean' else 'bool'}" for k in keys)
        name = group.rstrip("[]")
        parts.append(f"{name}:[{{{body}}}]" if group.endswith("[]") else f"{name}:{{{body}}}")
    return "{" + ",".join(parts) + "}"


def efficient_user(task: str, documents: str) -> str:
    return f"Schema:{compact_schema(task)}\n{documents}"
