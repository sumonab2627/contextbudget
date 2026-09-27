"""Generate a synthetic, fictional international health-insurance claim pack as a PDF.

The pack is deliberately messy in the ways real claim packs are:

* a fax cover sheet, repeated headers/footers and page numbers on every page
* a claim form whose declared admission date disagrees with the hospital's records
* clinical pages (discharge summary) alongside noise (lab tables, radiology)
* a two-page itemised hospital invoice *and* a "DUPLICATE COPY" of its first page
* a surgeon's invoice that was re-issued - the original is still in the pack, and only
  the email thread says which one is current
* an email thread with quoted replies repeating earlier messages
* a policy schedule, three pages of terms & conditions, and a privacy notice

All names, organisations and identifiers are invented. Outputs:
``data/claim_pack.pdf`` and ``data/ground_truth.json`` (expected extraction plus the
evidence strings each field depends on).

    python -m claims.generate
"""
from __future__ import annotations

import json
import random
import textwrap
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

INSURER = "Meridian Global Health (fictional)"
HEADER = f"{INSURER} - International Claims Unit | Fax +44 20 7946 0000 | Received 22/08/2026 09:14"
FOOTER = "CONFIDENTIAL - contains personal health information. Handle in line with data protection policy."

MEMBER = "Daniel Okafor"
MEMBER_ID = "MGH-7730192-01"
POLICY = "IPMI-GB-2291847"
PLAN = "Global Prestige Plus"
HOSPITAL = "Lantern Bay Medical Centre"
PREAUTH = "PA-2026-118734"


def _money(x: float) -> str:
    return f"{x:,.2f}"


def _wrap(text: str, width: int = 100) -> list[str]:
    out = []
    for para in text.split("\n"):
        out += textwrap.wrap(para, width) or [""]
    return out


def _boilerplate(rng: random.Random, n_paras: int, topic: str) -> list[str]:
    s = [
        f"The insurer may request any further information it reasonably requires to assess a claim under this {topic}.",
        "Benefits are payable only for treatment that is medically necessary and provided by a recognised medical practitioner.",
        "Where the member holds other insurance covering the same event, the insurer will pay only its rateable proportion.",
        "Any claim that is fraudulent or exaggerated in any respect will be declined and the policy may be cancelled from inception.",
        "Claims must be submitted within six months of the date of treatment unless the insurer agrees otherwise in writing.",
        "The insurer's liability is limited to the benefit limits shown in the policy schedule and table of benefits.",
        "Treatment received in a country subject to international sanctions is excluded from cover.",
        "The member must take all reasonable steps to minimise the cost of treatment, including using network providers where available.",
        "Amounts in currencies other than the policy currency are converted at the rate applicable on the date of treatment.",
        "Nothing in these terms affects the member's statutory rights under the law governing this contract.",
        "Personal data is processed for the purposes of underwriting, claims handling, fraud prevention and regulatory reporting.",
        "Data may be transferred to service providers outside the member's country of residence subject to appropriate safeguards.",
        "Complaints should be raised first with the insurer's customer relations team before referral to an ombudsman service.",
        "Waiting periods apply to certain benefits as shown in the table of benefits and are not waived on renewal.",
        "Pre-existing conditions are excluded unless declared at application and accepted in writing by the insurer.",
    ]
    lines = []
    for i in range(n_paras):
        lines += _wrap(f"{i + 1}. " + " ".join(rng.sample(s, 4)))
        lines.append("")
    return lines


def build(seed: int = 11) -> tuple[list[list[str]], dict]:
    rng = random.Random(seed)
    pages: list[list[str]] = []

    # 1 - fax cover
    pages.append([
        "FAX TRANSMISSION", "", "To: Meridian Global Health - International Claims Unit",
        f"From: {HOSPITAL}, Patient Accounts", "Date: 21/08/2026   Pages (including cover): 21", "",
        f"Re: Claim submission for patient {MEMBER}", "",
        *_wrap("Please find enclosed the completed claim form, pre-authorisation letter, discharge summary, "
               "investigation reports, itemised invoices and supporting correspondence. If any pages are "
               "missing or illegible please contact our patient accounts team on +852 5550 1234."),
        "", *_wrap("This transmission is intended only for the addressee. If you have received it in error, "
                   "please notify the sender immediately and destroy all copies."),
    ])

    # 2-3 - claim form
    pages.append([
        "CLAIM FORM - INPATIENT AND DAY-CASE TREATMENT (Section A-C)", "",
        "SECTION A - MEMBER DETAILS",
        f"Member name: {MEMBER}", f"Member ID: {MEMBER_ID}", f"Policy number: {POLICY}",
        f"Plan: {PLAN}", "Date of birth: 03/11/1984", "Country of residence: United Kingdom",
        "Correspondence address: 18 Alder Row, Leeds LS6 2QP", "",
        "SECTION B - TREATMENT DETAILS",
        "Country of treatment: Hong Kong", f"Hospital / clinic: {HOSPITAL}",
        "Date of admission (as declared by member): 13/08/2026", "Date of discharge: 18/08/2026",
        "Reason for treatment: severe abdominal pain, admitted via emergency department",
        "Was the treatment pre-authorised? Yes - reference PA-2026-118734",
        "Is the claim related to an accident or injury? No", "",
        "SECTION C - PAYMENT",
        "Pay: provider directly (direct settlement)", "Claim currency: HKD",
        "Bank details: not applicable - direct settlement to provider",
    ])
    pages.append([
        "CLAIM FORM - DECLARATION AND AUTHORITY (Section D)", "",
        *_wrap("I declare that the information given in this form is true and complete to the best of my "
               "knowledge. I understand that providing false or misleading information may result in my claim "
               "being declined and my policy being cancelled. I authorise any doctor, hospital or other provider "
               "to release to the insurer any information it requests about my medical history and treatment."),
        "", *_boilerplate(rng, 3, "declaration"),
        "Signed: D. Okafor        Date: 20/08/2026",
    ])

    # 4 - pre-authorisation letter
    pages.append([
        "PRE-AUTHORISATION CONFIRMATION", "",
        f"Pre-authorisation reference: {PREAUTH}", "Status: Approved",
        f"Member: {MEMBER} ({MEMBER_ID})", f"Provider: {HOSPITAL}",
        "Approved procedure: Laparoscopic appendicectomy with general anaesthesia",
        "Approved amount: HKD 120,000.00 (inclusive of hospital, surgeon and anaesthetist fees)",
        "Approved length of stay: up to 4 nights", "",
        *_wrap("This authorisation is subject to the terms and conditions of the member's policy and to the "
               "benefits remaining available at the date of treatment. Any costs above the approved amount must "
               "be referred to the insurer before they are incurred, otherwise they may not be covered."),
        "", *_boilerplate(rng, 2, "authorisation"),
    ])

    # 5-6 - discharge summary
    pages.append([
        "DISCHARGE SUMMARY", "", f"Hospital: {HOSPITAL}, 88 Harbour Crescent, Kowloon, Hong Kong",
        f"Patient: {MEMBER}   Hospital number: LB-0048812", "Date of admission: 14 August 2026",
        "Date of discharge: 18 August 2026", "Attending physician: Dr Helen Cheung (General Surgery)", "",
        "Presenting complaint:",
        *_wrap("41-year-old man presented to the emergency department at 23:40 on 14 August 2026 with 18 hours "
               "of periumbilical pain migrating to the right iliac fossa, anorexia and low-grade fever. "
               "Background of type 2 diabetes, diet and metformin controlled."),
        "", "Diagnoses (ICD-10):",
        "Primary: K35.80 Unspecified acute appendicitis",
        "Secondary: K66.0 Peritoneal adhesions",
        "Secondary: E11.9 Type 2 diabetes mellitus without complications", "",
        "Hospital course:",
        *_wrap("CT abdomen confirmed an inflamed, non-perforated appendix. The patient was taken to theatre on "
               "15 August 2026. Intra-operative findings of mild adhesions, divided. Post-operative recovery was "
               "uncomplicated; capillary glucose was monitored four-hourly and remained within target."),
    ])
    pages.append([
        "DISCHARGE SUMMARY (continued)", "", "Procedures:",
        "15 August 2026 - 0DTJ4ZZ Laparoscopic appendicectomy (CPT 44970) - Surgeon: Dr Marcus Lee",
        "15 August 2026 - General anaesthesia - Anaesthetist: Dr Priya Nair", "",
        "Medication on discharge:",
        "Co-amoxiclav 625 mg three times daily for 5 days", "Paracetamol 1 g four times daily as required",
        "Metformin 500 mg twice daily (unchanged)", "",
        "Follow-up:",
        *_wrap("Wound review with practice nurse at 7-10 days. Outpatient surgical review in 6 weeks. Return "
               "advice given for fever, increasing pain or wound discharge."),
        "", "Signed: Dr Helen Cheung, Consultant General Surgeon, 18 August 2026",
    ])

    # 7 - lab report (noise)
    lab = ["LABORATORY REPORT", "", f"Patient: {MEMBER}   Specimen collected: 14/08/2026 23:58", "",
           "Test                         Result     Units       Reference range"]
    for name, unit, lo, hi in [("Haemoglobin", "g/L", 130, 170), ("White cell count", "x10^9/L", 4, 11),
                               ("Neutrophils", "x10^9/L", 2, 7.5), ("Platelets", "x10^9/L", 150, 400),
                               ("Sodium", "mmol/L", 135, 145), ("Potassium", "mmol/L", 3.5, 5.1),
                               ("Urea", "mmol/L", 2.5, 7.8), ("Creatinine", "umol/L", 60, 110),
                               ("C-reactive protein", "mg/L", 0, 5), ("Glucose (random)", "mmol/L", 4, 7.8),
                               ("HbA1c", "mmol/mol", 20, 42), ("Bilirubin", "umol/L", 0, 21),
                               ("ALT", "U/L", 0, 45), ("ALP", "U/L", 30, 130), ("Amylase", "U/L", 28, 100),
                               ("Lactate", "mmol/L", 0.5, 2.2)]:
        for when in ["14/08", "16/08"]:
            v = round(rng.uniform(lo * 0.8, hi * 1.6), 1)
            lab.append(f"{name + ' (' + when + ')':<29}{v:<11}{unit:<12}{lo}-{hi}")
    pages.append(lab)

    # 8 - radiology (noise)
    pages.append([
        "RADIOLOGY REPORT - CT ABDOMEN AND PELVIS WITH CONTRAST", "",
        f"Patient: {MEMBER}   Examination date: 15/08/2026 01:12", "",
        *_wrap("Technique: Axial images acquired through the abdomen and pelvis following intravenous contrast, "
               "with coronal and sagittal reformats. Findings: The appendix is dilated to 11 mm with wall "
               "thickening, mucosal hyperenhancement and periappendiceal fat stranding. No appendicolith. No "
               "extraluminal gas or organised collection to suggest perforation. Small volume of free fluid in "
               "the pelvis. Liver, spleen, pancreas, adrenals and kidneys are unremarkable. No lymphadenopathy. "
               "Lung bases clear. Impression: Acute non-perforated appendicitis."),
    ])

    # 9-10 - hospital invoice (+ duplicate of page 9 later)
    items = [("Accommodation - private room (per night)", 4, 3_200), ("Nursing care (per day)", 4, 1_450),
             ("Emergency department attendance", 1, 2_800), ("Operating theatre - first hour", 1, 14_500),
             ("Operating theatre - additional 30 min", 1, 5_200), ("Recovery room", 1, 2_600),
             ("CT abdomen and pelvis with contrast", 1, 7_900), ("Laboratory - haematology panel", 2, 680),
             ("Laboratory - biochemistry panel", 2, 940), ("Laboratory - CRP", 2, 260),
             ("Laboratory - HbA1c", 1, 420), ("Capillary glucose monitoring (per day)", 4, 180),
             ("Laparoscopic consumables pack", 1, 9_650), ("Surgical stapler reload", 2, 1_880),
             ("Specimen histopathology", 1, 2_300), ("IV fluids and giving sets", 6, 145.5),
             ("IV antibiotics - co-amoxiclav 1.2 g", 9, 96.25), ("Analgesia - IV paracetamol 1 g", 8, 62),
             ("Anti-emetic - ondansetron 4 mg", 4, 58), ("Enoxaparin 40 mg", 3, 135),
             ("Dressings and wound care", 1, 540), ("Physiotherapy review", 1, 950),
             ("Dietitian review (diabetes)", 1, 880), ("Medical records and reports", 1, 350),
             ("Patient meals (per day)", 4, 285), ("Medical supplies - miscellaneous", 1, 1_127.5)]
    rows = [(f"{i + 1:04d}", d, q, p, q * p) for i, (d, q, p) in enumerate(items)]
    subtotal = sum(r[4] for r in rows)
    discount = round(subtotal * 0.05, 2)
    hosp_total = round(subtotal - discount, 2)
    head = ["TAX INVOICE", "", f"{HOSPITAL} - Patient Accounts", "Invoice number: INV-LBMC-558201",
            "Invoice date: 18/08/2026", f"Patient: {MEMBER}   Insurer: Meridian Global Health   Ref: {PREAUTH}", "",
            "Code  Description                                   Qty    Unit price      Amount (HKD)"]
    fmt = lambda r: f"{r[0]}  {r[1]:<44}{r[2]:>4}  {_money(r[3]):>12}  {_money(r[4]):>16}"
    page9 = head + [fmt(r) for r in rows[:15]] + ["", "Continued on next page"]
    page10 = ["TAX INVOICE (continued) - INV-LBMC-558201", "",
              "Code  Description                                   Qty    Unit price      Amount (HKD)"] + \
             [fmt(r) for r in rows[15:]] + ["",
              f"Subtotal: HKD {_money(subtotal)}", f"Network discount (5%): -HKD {_money(discount)}",
              f"Invoice total: HKD {_money(hosp_total)}", "Payment terms: 30 days. Direct settlement agreed with insurer."]
    pages += [page9, page10]
    pages.append(["*** DUPLICATE COPY - NOT A NEW INVOICE ***"] + page9)

    # surgeon invoices: original (superseded) and re-issued
    def surgeon(num: str, date: str, fee: float, assist: float) -> list[str]:
        return ["INVOICE - SURGEON'S FEES", "", "Dr Marcus Lee, FRCS - Suite 1204, Nathan Medical Tower, Kowloon",
                f"Invoice number: {num}", f"Invoice date: {date}", f"Patient: {MEMBER}   Policy: {POLICY}", "",
                f"15/08/2026  Laparoscopic appendicectomy (CPT 44970)            HKD {_money(fee)}",
                f"15/08/2026  Division of adhesions                              HKD {_money(assist)}",
                "", f"Total due: HKD {_money(fee + assist)}", "", *_wrap(
                    "Fees are charged in accordance with the practice's published schedule. Please quote the "
                    "invoice number with any payment or query.")]
    pages.append(surgeon("SF-2290", "18/08/2026", 36_000, 6_000))
    pages.append(surgeon("SF-2291", "20/08/2026", 34_000, 4_500))
    surgeon_total = 38_500.0

    pages.append(["INVOICE - ANAESTHETIC SERVICES", "", "Dr Priya Nair, FANZCA - Kowloon Anaesthesia Partners",
                  "Invoice number: AN-7719", "Invoice date: 18/08/2026", f"Patient: {MEMBER}", "",
                  "15/08/2026  General anaesthesia, laparoscopic procedure, 95 minutes   HKD 11,400.00",
                  "15/08/2026  Pre-operative anaesthetic assessment                      HKD 1,400.00",
                  "", "Total due: HKD 12,800.00"])
    anaes_total = 12_800.0

    rx = [("Co-amoxiclav 625 mg x 15", 386.4), ("Paracetamol 500 mg x 32", 48.0),
          ("Wound dressing pack", 212.0), ("Blood glucose test strips x 50", 640.0)]
    rx_total = round(sum(p for _, p in rx), 2)
    pages.append(["PHARMACY RECEIPT", "", f"{HOSPITAL} Outpatient Pharmacy", "Receipt number: RX-33019",
                  "Date: 18/08/2026", f"Patient: {MEMBER}", ""] +
                 [f"{d:<40}HKD {_money(p):>10}" for d, p in rx] + ["", f"Total paid: HKD {_money(rx_total)}"])

    # email thread with quoted replies
    pages.append([
        "EMAIL CORRESPONDENCE", "",
        "From: accounts@lanternbay.example   To: claims@meridian.example", "Date: 21/08/2026 16:05",
        f"Subject: RE: RE: Claim {PREAUTH} - {MEMBER} - surgeon invoice",
        *_wrap("Dear Claims Team, as requested we have re-issued Dr Lee's invoice after applying the network "
               "fee schedule. Invoice SF-2291 replaces SF-2290 in full; please disregard SF-2290. All other "
               "invoices in the pack are unchanged. Kind regards, Patient Accounts."), "",
        "> From: claims@meridian.example   Sent: 21/08/2026 10:22",
        *["> " + l for l in _wrap("Thank you for the pack. Dr Lee's invoice SF-2290 exceeds the network fee schedule "
                                  "for CPT 44970. Could you ask the practice to review and re-issue if appropriate? "
                                  "We will hold the claim until we hear from you.", 96)],
        "> ",
        "> > From: accounts@lanternbay.example   Sent: 20/08/2026 17:48",
        *["> > " + l for l in _wrap(f"Please find attached the claim pack for {MEMBER}, pre-authorisation "
                                     f"{PREAUTH}. Direct settlement has been agreed. Invoices enclosed: "
                                     "INV-LBMC-558201, SF-2290, AN-7719 and RX-33019.", 94)],
    ])

    pages.append([
        "POLICY SCHEDULE - EXTRACT", "", f"Policyholder: {MEMBER}   Policy number: {POLICY}   Plan: {PLAN}",
        "Period of cover: 01/01/2026 to 31/12/2026   Area of cover: Worldwide excluding USA", "",
        "Benefit                                   Limit                 Excess / co-insurance",
        "Overall annual maximum                    GBP 3,000,000         GBP 250 excess per year",
        "Hospital accommodation                    Paid in full          None",
        "Surgeons' and anaesthetists' fees         Paid in full          None",
        "Diagnostic tests and imaging              Paid in full          None",
        "Outpatient consultations                  GBP 5,000             10% co-insurance",
        "Prescribed drugs (outpatient)             GBP 2,500             10% co-insurance",
        "Physiotherapy                             20 sessions           None",
        "Emergency evacuation                      Paid in full          None", "",
        *_wrap("This extract is provided for convenience. The full table of benefits and policy wording govern cover."),
    ])
    for i in range(3):
        pages.append([f"POLICY TERMS AND CONDITIONS - PART {i + 1} OF 3", ""] + _boilerplate(rng, 6, "policy"))
    pages.append(["PRIVACY NOTICE", ""] + _boilerplate(rng, 5, "privacy notice"))

    n = len(pages)
    pages = [[HEADER, ""] + p + ["", FOOTER, f"Page {i + 1} of {n}"] for i, p in enumerate(pages)]

    total_billed = round(hosp_total + surgeon_total + anaes_total + rx_total, 2)
    truth = {
        "member": {"name": MEMBER, "member_id": MEMBER_ID, "policy_number": POLICY, "plan": PLAN},
        "treatment": {"country": "Hong Kong", "hospital": HOSPITAL, "admission_date": "2026-08-14",
                      "discharge_date": "2026-08-18", "attending_physician": "Dr Helen Cheung"},
        "diagnoses": [{"icd10": "K35.80", "description": "Unspecified acute appendicitis", "primary": True},
                      {"icd10": "K66.0", "description": "Peritoneal adhesions", "primary": False},
                      {"icd10": "E11.9", "description": "Type 2 diabetes mellitus without complications", "primary": False}],
        "procedures": [{"code": "0DTJ4ZZ", "description": "Laparoscopic appendicectomy", "date": "2026-08-15"}],
        "pre_authorisation": {"reference": PREAUTH, "approved_amount": 120000.0, "status": "Approved"},
        "invoices": [
            {"provider": HOSPITAL, "invoice_number": "INV-LBMC-558201", "date": "2026-08-18", "amount": hosp_total},
            {"provider": "Dr Marcus Lee", "invoice_number": "SF-2291", "date": "2026-08-20", "amount": surgeon_total},
            {"provider": "Dr Priya Nair", "invoice_number": "AN-7719", "date": "2026-08-18", "amount": anaes_total},
            {"provider": f"{HOSPITAL} Outpatient Pharmacy", "invoice_number": "RX-33019", "date": "2026-08-18", "amount": rx_total}],
        "totals": {"currency": "HKD", "total_billed": total_billed,
                   "exceeds_pre_authorisation": total_billed > 120000},
    }
    # Evidence: strings that must be visible *together in one call* to get each field right.
    ev = {
        "member.name": ["Member name: Daniel Okafor"], "member.member_id": [MEMBER_ID],
        "member.policy_number": [POLICY], "member.plan": [PLAN],
        "treatment.country": ["Country of treatment: Hong Kong"],
        "treatment.hospital": [HOSPITAL],
        "treatment.admission_date": ["Date of admission: 14 August 2026"],
        "treatment.discharge_date": ["Date of discharge: 18 August 2026"],
        "treatment.attending_physician": ["Attending physician: Dr Helen Cheung"],
        "diagnoses.K35.80": ["Primary: K35.80"], "diagnoses.K66.0": ["K66.0"], "diagnoses.E11.9": ["E11.9"],
        "procedures.0DTJ4ZZ": ["0DTJ4ZZ"],
        "pre_authorisation.reference": [f"Pre-authorisation reference: {PREAUTH}"],
        "pre_authorisation.approved_amount": ["Approved amount: HKD 120,000.00"],
        "pre_authorisation.status": ["Status: Approved"],
        "invoices.INV-LBMC-558201": [f"Invoice total: HKD {_money(hosp_total)}"],
        "invoices.SF-2291": ["Invoice number: SF-2291", "Invoice SF-2291 replaces SF-2290"],
        "invoices.AN-7719": ["Total due: HKD 12,800.00"],
        "invoices.RX-33019": [f"Total paid: HKD {_money(rx_total)}"],
        "invoices.no_superseded": ["Invoice SF-2291 replaces SF-2290"],
        "totals.currency": ["HKD"],
        "totals.total_billed": [f"Invoice total: HKD {_money(hosp_total)}", "Invoice SF-2291 replaces SF-2290",
                                "Total due: HKD 12,800.00", f"Total paid: HKD {_money(rx_total)}", "Total due: HKD 38,500.00"],
        "totals.exceeds_pre_authorisation": ["Approved amount: HKD 120,000.00", f"Invoice total: HKD {_money(hosp_total)}"],
    }
    return pages, {"expected": truth, "evidence": ev}


def write_pdf(pages: list[list[str]], path: Path) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=A4)
    c.setTitle("Synthetic claim pack (fictional)")
    w, h = A4
    for page in pages:
        y = h - 40
        for i, line in enumerate(page):
            size = 10 if i == 2 else 8
            c.setFont("Courier-Bold" if i == 2 else "Courier", size)
            c.drawString(36, y, line)
            y -= size + 3.2
            if y < 30:
                raise ValueError(f"page overflow: {page[2]}")
        c.showPage()
    c.save()


def main() -> None:
    DATA.mkdir(exist_ok=True)
    pages, truth = build()
    write_pdf(pages, DATA / "claim_pack.pdf")
    (DATA / "ground_truth.json").write_text(json.dumps(truth, indent=2))
    print(f"Wrote {DATA / 'claim_pack.pdf'} ({len(pages)} pages) and ground_truth.json")


if __name__ == "__main__":
    main()
