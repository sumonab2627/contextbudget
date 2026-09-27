"""A deterministic, synthetic long-running agent session.

An incident-investigation agent works through N turns (default 40) on a claims-processing
outage. It reads large log dumps with a tool, the user makes decisions along the
way, the agent writes notes, and a 40-document claims-policy knowledge base sits
alongside. Facts the agent will later be asked about are planted at known turns:

    turn 2   root-cause error code      (tool output + agent's reply + note)
    turn 3   p99 latency                (ONLY in raw tool output - a deliberately hard case)
    turn 4   rollback decision + owner  (user message only - no note)
    turn 7   affected batch range       (agent's reply + note)
    turn N-2 peak queue depth           (recent tool output + agent's reply)
    KB       limits / policy references buried among 40 similar documents

Everything is seeded, so every run produces identical numbers.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

SYSTEM_PROMPT = """You are an incident-investigation agent for a health-insurance claims platform.
Work step by step. Use tools to inspect services; after each tool call, state the key findings
in one or two sentences. Record durable facts with the `remember` tool. When answering, cite the
turn or document you relied on, and say plainly if the information is not in your context.

Tools:
- read_logs(service: str, window: str) -> recent log lines for a service
- query_metrics(metric: str, window: str) -> time-series summary
- list_batches(status: str) -> claim batches in a given status
- remember(note: str) -> persist a short note across the session"""

CATEGORIES = ["dental", "optical", "maternity", "physiotherapy", "outpatient",
              "inpatient", "oncology", "chiropractic"]
REGIONS = [("United Kingdom", "GBP"), ("Hong Kong", "HKD"), ("Singapore", "SGD"),
           ("UAE", "AED"), ("Spain", "EUR")]
SERVICES = ["claims-router", "payments-gateway", "member-api", "fraud-scorer",
            "notification-svc", "ledger-sync", "document-ocr", "eligibility-svc"]

_FILLER = [
    "Claims must include an itemised invoice and the treating provider's registration number.",
    "Pre-authorisation is recommended where the provider is outside the direct-settlement network.",
    "Members can track claim status through the member portal or the mobile app.",
    "Duplicate submissions are detected automatically and closed with a reference to the original.",
    "Currency conversion uses the published rate on the date of treatment.",
    "Claims flagged by the fraud-scoring service are held for investigation before settlement.",
    "Turnaround targets are measured from receipt of a complete claim, not first contact.",
    "Supporting documents may be uploaded as PDF or image files up to 10 MB each.",
    "Where a claim is partially approved, the member receives an itemised explanation of benefits.",
    "Providers in the direct-settlement network are paid within the agreed settlement window.",
    "Appeals must be lodged within 60 days of the decision letter.",
    "Exclusions and waiting periods are defined in the member's policy schedule.",
]


@dataclass
class Turn:
    n: int
    user: str
    assistant: str
    tool_call: str | None = None
    tool_output: str | None = None
    note: str | None = None


@dataclass
class Question:
    qid: str
    text: str
    must_contain: list[str]
    kind: str


@dataclass
class Scenario:
    system_prompt: str
    documents: list[tuple[str, str, str]]          # (doc_id, title, text)
    turns: list[Turn]
    checkpoints: dict[int, list[Question]] = field(default_factory=dict)


def _kb(rng: random.Random) -> tuple[list[tuple[str, str, str]], dict]:
    docs, facts = [], {}
    for cat in CATEGORIES:
        for region, cur in REGIONS:
            limit = f"{cur} {rng.randrange(8, 95) * 100:,}"
            ref = f"POL-{rng.randrange(1000, 9999)}"
            turnaround = rng.choice([5, 7, 10, 12, 15])
            paras = [
                " ".join(rng.sample(_FILLER, 3)),
                f"For {cat} claims submitted in {region}, the auto-approval limit is {limit}. "
                f"Claims above this limit are routed to manual review by a clinical assessor. "
                f"Policy reference: {ref}.",
                f"The {region} operations team targets a decision on {cat} claims within "
                f"{turnaround} working days. " + " ".join(rng.sample(_FILLER, 3)),
                " ".join(rng.sample(_FILLER, 3)),
            ]
            doc_id = f"kb-{cat[:4]}-{region.split()[0].lower()[:3]}"
            docs.append((doc_id, f"{cat.title()} claims - {region}", "\n\n".join(paras)))
            facts[(cat, region)] = (limit, ref)
    return docs, facts


def _logs(rng: random.Random, service: str, planted: list[str], lines: int = 38) -> str:
    out = []
    for i in range(lines):
        mm, ss = 38 + i // 10, rng.randrange(60)
        lvl = rng.choices(["INFO", "INFO", "INFO", "WARN"], k=1)[0]
        if lvl == "WARN":
            msg = f"W-{rng.randrange(1000, 9999)} retrying upstream call attempt={rng.randrange(1, 4)}"
        else:
            msg = (f"request_id=R-{rng.randrange(10**6, 10**7)} path=/v2/claims/{rng.randrange(10**5, 10**6)} "
                   f"status={rng.choice([200, 200, 200, 202, 409])} latency_ms={rng.randrange(20, 900)}")
        out.append(f"2026-09-14T09:{mm:02d}:{ss:02d}Z {lvl} {service} {msg}")
    for p in planted:
        out.insert(rng.randrange(len(out)), p)
    return "\n".join(out)


def build_scenario(seed: int = 7, total_turns: int = 40) -> Scenario:
    if total_turns < 14:
        raise ValueError("total_turns must be >= 14")
    rng = random.Random(seed)
    docs, kb = _kb(rng)
    t: list[Turn] = []

    t.append(Turn(1, "Claims have been stuck in 'processing' since about 09:40. Start investigating the claims pipeline.",
                  "Starting with the claims-router, which fronts the pipeline. I'll pull its recent logs."))
    t.append(Turn(2, "Go ahead and check the claims-router logs.",
                  "Root cause candidate: error E-4471 in claims-router - a schema mismatch on member_id "
                  "(integer sent where a string is expected), starting right after the 09:38 deploy.",
                  'read_logs(service="claims-router", window="30m")',
                  _logs(rng, "claims-router", [
                      "2026-09-14T09:38:12Z ERROR claims-router E-4471 schema mismatch on field member_id (expected string, got int)",
                      "2026-09-14T09:41:55Z ERROR claims-router E-4471 schema mismatch on field member_id (expected string, got int)"]),
                  "Root cause candidate: E-4471 member_id schema mismatch in claims-router after the 09:38 deploy."))
    t.append(Turn(3, "Is the payments side affected too? Check payments-gateway.",
                  "payments-gateway looks healthy: no errors, latency within its normal range, so the problem is upstream.",
                  'read_logs(service="payments-gateway", window="30m")',
                  _logs(rng, "payments-gateway", [
                      "2026-09-14T09:45:00Z INFO payments-gateway summary window=30m p99_latency_ms=2,317 error_rate=0.02%"])))
    t.append(Turn(4, "Decision: we agreed with the platform lead to roll back claims-router to v3.8.2 tonight. "
                     "Priya owns the change and will run it at 22:00.",
                  "Understood - rollback of claims-router to v3.8.2 at 22:00, owned by Priya. I'll keep collecting evidence for the change record."))
    for n, svc in [(5, "member-api"), (6, "fraud-scorer")]:
        t.append(Turn(n, f"Check {svc} for anything related.",
                      f"No errors in {svc}; the W-codes are routine upstream retries and not related to the outage.",
                      f'read_logs(service="{svc}", window="30m")', _logs(rng, svc, [])))
    t.append(Turn(7, "Which claim batches are stuck?",
                  "Affected batches are B-20931 through B-20958 (28 batches), all created after 09:38.",
                  'list_batches(status="processing")',
                  "\n".join(f"B-{b} status=processing created=09:{38 + (b - 20931) // 4:02d} claims={rng.randrange(40, 400)}"
                            for b in range(20931, 20959)) + "\n" + _logs(rng, "batch-scheduler", [], 20),
                  "Affected batches: B-20931 to B-20958 (28 batches)."))
    noise = SERVICES[2:]
    for n in range(8, total_turns - 2):
        svc = noise[n % len(noise)]
        if n % 2:
            t.append(Turn(n, f"Look at {svc} again with a wider window.",
                          f"{svc} shows only routine retries; nothing new beyond what we already know.",
                          f'read_logs(service="{svc}", window="2h")', _logs(rng, svc, [])))
        else:
            t.append(Turn(n, f"Anything unusual in {svc} metrics?",
                          f"{svc} metrics are within normal bounds.",
                          f'query_metrics(metric="{svc}.error_rate", window="2h")', _logs(rng, svc, [], 24)))
    t.append(Turn(total_turns - 2, "How bad did the backlog get?",
                  "The claims queue depth peaked at 18,442 at 10:12 and is now draining slowly.",
                  'query_metrics(metric="claims.queue_depth", window="2h")',
                  _logs(rng, "queue-monitor", ["2026-09-14T10:12:00Z INFO queue-monitor claims.queue_depth peak=18,442"], 24)))
    t.append(Turn(total_turns - 1, "Check eligibility-svc once more.", "eligibility-svc is healthy.",
                  'read_logs(service="eligibility-svc", window="30m")', _logs(rng, "eligibility-svc", [])))
    t.append(Turn(total_turns, "Good. Let's prepare the change record.", "Ready - ask me for any detail you need for the record."))

    d_lim, _ = kb[("dental", "Hong Kong")]
    _, m_ref = kb[("maternity", "UAE")]
    p_lim, _ = kb[("physiotherapy", "Singapore")]
    o_lim, o_ref = kb[("oncology", "Spain")]
    q = {
        "root_cause": Question("root_cause", "What error code did we identify as the root cause candidate in claims-router?", ["E-4471"], "early finding"),
        "decision": Question("decision", "Which version are we rolling claims-router back to, and who owns the change?", ["v3.8.2", "Priya"], "user decision"),
        "batches": Question("batches", "Which claim batch IDs are affected by the outage?", ["B-20931", "B-20958"], "agent note"),
        "queue": Question("queue", "What was the peak claims queue depth?", ["18,442"], "recent tool result"),
        "kb_dental": Question("kb_dental", "What is the auto-approval limit for dental claims submitted in Hong Kong?", [d_lim], "knowledge base"),
        "kb_maternity": Question("kb_maternity", "What is the policy reference for maternity claims in the UAE?", [m_ref], "knowledge base"),
        "kb_physio": Question("kb_physio", "What is the auto-approval limit for physiotherapy claims in Singapore?", [p_lim], "knowledge base"),
        "kb_oncology": Question("kb_oncology", "For oncology claims in Spain, what are the auto-approval limit and policy reference?", [o_lim, o_ref], "knowledge base"),
        "p99": Question("p99", "What p99 latency did the payments-gateway logs report?", ["2,317"], "old raw tool output"),
    }
    mid = [q[k] for k in ["root_cause", "decision", "batches", "kb_dental", "kb_maternity", "p99"]]
    return Scenario(SYSTEM_PROMPT, docs, t, {12: mid, total_turns: list(q.values())})
