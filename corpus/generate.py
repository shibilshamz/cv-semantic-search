"""Generate a synthetic CV corpus.

Why synthetic: the real corpus this service indexes is other people's personal
data. It cannot be committed, screenshotted or demoed. A generated corpus keeps
the repo reproducible for anyone who clones it, and lets the eval set in
eval/briefs.yaml assert on specific candidates by name.

The corpus is deliberately built to break the chunker rather than to flatter it:

  * three heading vocabularies -- "Work Experience" / "Employment History" /
    "Professional Experience" -- plus CVs with no headings at all
  * a near-duplicate pair (same person, two recruitment agencies, different
    formatting) to exercise collapse-by-candidate at retrieval
  * one candidate whose relevant skill is described in words no recruiter would
    ever type, which is the case that separates semantic search from grep

Deterministic: seeded, so regenerating produces byte-identical files and the
eval numbers stay comparable across runs.

    python corpus/generate.py
"""

import json
import random
import re
from pathlib import Path

SEED = 20260905
OUT = Path(__file__).parent / "synthetic"

# --------------------------------------------------------------------------
# Planted candidates. Hand-written, because the eval asserts on them by email.
# Each one exists to test something specific.
# --------------------------------------------------------------------------

PLANTED = [
    # The case for semantic retrieval. A recruiter searching for "payments" or
    # "APIs" finds nothing here -- neither word appears in the CV. The meaning
    # does. If this candidate is not returned for the payments brief, the
    # embedding step is not earning its place.
    {
        "email": "ahmed.rashid@example.com",
        "name": "Ahmed Al-Rashid",
        "file": "ahmed-al-rashid.txt",
        "text": """Ahmed Al-Rashid
ahmed.rashid@example.com | +971 50 411 8823 | Dubai, UAE
Emirati national | No visa sponsorship required

Professional Summary
Nine years building high-throughput transaction processing for regional banks.
Comfortable owning a service end to end, from schema design through to the
on-call rotation that wakes you at 3am when it misbehaves.

Work Experience

Senior Software Engineer, Emirates NBD
Feb 2021 - Present
Designed and built the settlement reconciliation service that matches inbound
card movements against issuer statements overnight. Reduced manual investigation
from four hours a day to under twenty minutes.
Introduced idempotency keys across the money movement path after a duplicate
posting incident; the class of bug has not recurred in three years.
Ran the migration of the nightly clearing batch from a monolith cron job onto
Kafka, cutting the settlement window from six hours to ninety minutes.

Software Engineer, Network International
Aug 2016 - Jan 2021
Maintained the card authorisation gateway, roughly 900 requests per second at
peak, with a 99.98% availability target.
Wrote the retry and reversal logic for declined transactions, including the
timeout handling that decides whether a customer gets charged twice.

Technical Skills
Python, Go, PostgreSQL, Kafka, Redis, Kubernetes, Terraform, gRPC

Education
BSc Computer Science, American University of Sharjah, 2016
""",
    },
    # Near-duplicate pair, file 1 of 2. Same person submitted by two agencies,
    # different layout and heading vocabulary. Both index under the same email,
    # so retrieval must return her once, not twice.
    {
        "email": "elena.petrova@example.com",
        "name": "Elena Petrova",
        "file": "elena-petrova-agency-a.txt",
        "text": """Elena Petrova
elena.petrova@example.com
+971 55 902 4471 | Dubai Marina, UAE
Russian national | UAE residence visa, transferable

Profile
Property consultant with seven years selling off-plan and secondary market
residential units in Dubai. Fluent Russian, English and conversational Arabic.
Consistently in the top three of a forty-agent floor.

Employment History

Senior Property Consultant, Allsopp Property
Mar 2022 - Present
Closed AED 94 million in residential sales across 2024, the second highest on
the team.
Built the Russian-speaking client desk from scratch; it now accounts for around
40% of the office's off-plan volume.

Property Consultant, Betterhomes
Jun 2019 - Feb 2022
Handled secondary market listings across Dubai Marina and JLT.

Languages
Russian (native), English (fluent), Arabic (conversational)

Education
Diploma in Business Administration, Moscow State University, 2018
""",
    },
    {
        "email": "elena.petrova@example.com",
        "name": "Elena Petrova",
        "file": "elena-petrova-agency-b.txt",
        "text": """CURRICULUM VITAE

Name: Elena Petrova
Email: elena.petrova@example.com
Mobile: +971 55 902 4471
Location: Dubai, United Arab Emirates
Nationality: Russian
Visa: Residence visa (transferable)
Notice Period: 30 days

Career Summary
Seven years in Dubai residential real estate. Off-plan and resale. Strong
Russian-speaking client network. Top-three performer on a large sales floor.

Professional Experience

Allsopp Property - Senior Property Consultant
2022 - Present
AED 94m residential sales in 2024. Established and grew the Russian client desk
to roughly 40% of office off-plan volume.

Betterhomes - Property Consultant
2019 - 2022
Secondary market listings, Dubai Marina and Jumeirah Lake Towers.

Key Skills
Off-plan sales, client relationship management, RERA certified, CRM (Property
Finder, Bayut), negotiation

Languages
Russian, English, Arabic (conversational)
""",
    },
    # No headings at all -- one unstructured block. Exercises the chunker's
    # fallback path, and must still be retrievable.
    {
        "email": "maria.santos@example.com",
        "name": "Maria Santos",
        "file": "maria-santos.txt",
        "text": """Maria Santos, mariasantos@example.com, +971 52 338 1190, Jebel Ali, Dubai.
Filipino national, employment visa, 60 days notice. Warehouse supervisor with
eleven years moving palletised freight through third-party logistics sites in
Jebel Ali and Dubai South. Currently at Aramex where I run a night shift of
fourteen pickers across a 22,000 square metre facility, responsible for inbound
receiving, putaway and outbound despatch accuracy. Brought picking errors down
from 1.8% to 0.6% over two years by reorganising slotting around velocity and
retraining the team on scan discipline. Before Aramex I spent six years at
Agility in a similar role covering inbound only. Comfortable with SAP EWM and
Manhattan WMS, forklift licensed, and I have run the annual stock count for the
last four years. Diploma in Supply Chain Management, University of Santo Tomas.
""",
    },
    {
        "email": "rajesh.menon@example.com",
        "name": "Rajesh Menon",
        "file": "rajesh-menon.txt",
        "text": """Rajesh Menon
rajesh.menon@example.com | +971 56 771 2204 | Abu Dhabi, UAE
Indian national | Residence visa | 2 months notice

Summary
Data engineer, six years. Builds and operates streaming and batch pipelines.
Most recently responsible for the ingestion layer feeding a group-wide analytics
warehouse.

Professional Experience

Senior Data Engineer, ADNOC Distribution
Apr 2022 - Present
Built the real-time ingestion pipeline carrying fuel station telemetry into
Snowflake, roughly 40 million events a day, on Kafka and Spark Structured
Streaming.
Cut warehouse compute spend 38% by rewriting the heaviest transformations in
dbt and moving them off hourly full refreshes onto incremental models.

Data Engineer, Majid Al Futtaim
Jul 2019 - Mar 2022
Owned the nightly batch loading point-of-sale data from 340 retail sites.
Introduced data quality checks that caught silent schema drift from upstream.

Technical Skills
Python, SQL, Spark, Kafka, Airflow, dbt, Snowflake, AWS (S3, Glue, EMR)

Education
BTech Information Technology, NIT Calicut, 2019
""",
    },
    {
        "email": "fatima.zahra@example.com",
        "name": "Fatima Zahra Bennani",
        "file": "fatima-zahra-bennani.txt",
        "text": """Fatima Zahra Bennani
fatima.zahra@example.com | +971 54 620 3378 | Dubai, UAE
Moroccan national | Residence visa | Immediately available

Career Objective
Platform engineer looking for infrastructure ownership at a product company.

Work Experience

DevOps Engineer, Careem
Sep 2021 - Present
Run the Kubernetes platform, 14 clusters across three AWS regions, serving
around 200 microservices.
Replaced hand-maintained CloudFormation with Terraform modules; environment
provisioning went from a two-day ticket to a fifteen-minute pipeline run.
Cut the mean time to restore on the checkout path from 47 minutes to 9 by
rebuilding the alerting around user-visible symptoms rather than host metrics.

Systems Engineer, Injazat
Jan 2018 - Aug 2021
Managed on-premise VMware estate and the migration of 60 workloads to Azure.

Technical Skills
Kubernetes, Terraform, AWS, Docker, ArgoCD, Prometheus, Grafana, Python, Bash

Certifications
Certified Kubernetes Administrator (CKA), 2022
AWS Solutions Architect Associate, 2021

Languages
Arabic (native), French (fluent), English (fluent)
""",
    },
    {
        "email": "grace.okonkwo@example.com",
        "name": "Grace Okonkwo",
        "file": "grace-okonkwo.txt",
        "text": """Grace Okonkwo
grace.okonkwo@example.com | +971 50 118 7742 | Sharjah, UAE
Nigerian national | Residence visa | 1 month notice

Professional Summary
Registered nurse with nine years in acute care, six of them in intensive care.
DHA licensed.

Employment History

Staff Nurse - Intensive Care Unit, NMC Royal Hospital
Nov 2019 - Present
Twelve-bed adult ICU. Ventilator management, continuous renal replacement
therapy, post-operative cardiac recovery.
Precept new ICU nurses through their first three months on the unit.

Staff Nurse - Medical Ward, Lagos University Teaching Hospital
Feb 2015 - Sep 2019
General medical ward, 32 beds.

Certifications
DHA License (active), ACLS, BLS, Critical Care Nursing certificate

Education
BSc Nursing, University of Lagos, 2014
""",
    },
]

# --------------------------------------------------------------------------
# Filler. Volume so that retrieval has something to be wrong about -- with only
# the planted CVs, every brief would match trivially.
# --------------------------------------------------------------------------

FIRST = ["Omar", "Sara", "Vikram", "Aisha", "Daniel", "Nour", "Hassan", "Leila",
         "Arjun", "Mei", "Tariq", "Yasmin", "Kwame", "Ana", "Bilal", "Divya",
         "Samir", "Rania", "Joseph", "Chen", "Zara", "Marco", "Nadia", "Karim",
         "Ivan", "Priya", "Ahmed", "Lucia", "Farah", "Sunil", "Layla", "Peter"]
LAST = ["Haddad", "Kapoor", "Fernandes", "Nakamura", "Osei", "Rahman", "Silva",
        "Aziz", "Iyer", "Costa", "Khalifa", "Mbeki", "Ivanov", "Cheng", "Farouk",
        "Reyes", "Sharma", "Mansour", "Dube", "Bakr", "Lopes", "Nasser"]
NATIONALITY = ["Indian", "Pakistani", "Egyptian", "Filipino", "Lebanese",
               "Jordanian", "British", "South African", "Kenyan", "Sri Lankan",
               "Bangladeshi", "Syrian", "Tunisian", "Portuguese"]
CITY = ["Dubai, UAE", "Abu Dhabi, UAE", "Sharjah, UAE", "Doha, Qatar",
        "Riyadh, Saudi Arabia", "Manama, Bahrain", "Muscat, Oman", "Kuwait City, Kuwait"]

DOMAINS = {
    "frontend": {
        "titles": ["Frontend Developer", "Senior Frontend Engineer", "UI Engineer"],
        "firms": ["Noon", "Talabat", "Property Finder", "Tabby", "Bayut"],
        "skills": "JavaScript, TypeScript, React, Next.js, CSS, Webpack, Jest",
        "lines": [
            "Rebuilt the checkout funnel in React, lifting completion by 12%.",
            "Owned the component library used across four product teams.",
            "Cut first contentful paint from 4.1s to 1.3s on mid-range Android.",
        ],
        "edu": "BSc Computer Science",
    },
    "mobile": {
        "titles": ["Mobile Developer", "Senior React Native Developer", "iOS Engineer"],
        "firms": ["Careem", "Anghami", "Kitopi", "Fetchr"],
        "skills": "React Native, Swift, Kotlin, TypeScript, Firebase, Fastlane",
        "lines": [
            "Shipped the React Native rewrite of the customer app to 2.4m installs.",
            "Reduced crash-free session rate regressions by adding release gating.",
            "Built the offline-first sync layer for areas with poor connectivity.",
        ],
        "edu": "BEng Software Engineering",
    },
    "ml": {
        "titles": ["Machine Learning Engineer", "Data Scientist", "Applied Scientist"],
        "firms": ["G42", "Careem", "Emirates Group", "Mubadala"],
        "skills": "Python, PyTorch, scikit-learn, MLflow, SQL, Airflow",
        "lines": [
            "Built the demand forecasting model now driving daily fleet positioning.",
            "Deployed a churn model that lifted retention campaign precision 2.3x.",
            "Set up experiment tracking and offline evaluation for six model teams.",
        ],
        "edu": "MSc Data Science",
    },
    "qa": {
        "titles": ["QA Engineer", "Test Automation Engineer", "SDET"],
        "firms": ["Emaar", "du", "Etisalat", "Aramex"],
        "skills": "Selenium, Playwright, Python, Postman, JMeter, CI/CD",
        "lines": [
            "Automated the regression suite, cutting release testing from 5 days to 6 hours.",
            "Introduced contract testing between the two highest-churn services.",
        ],
        "edu": "BSc Information Technology",
    },
    "accounting": {
        "titles": ["Senior Accountant", "Financial Analyst", "Accounts Manager"],
        "firms": ["Al Futtaim", "Chalhoub Group", "Landmark Group", "Alshaya"],
        "skills": "SAP FICO, Oracle Financials, Excel, IFRS, VAT filing",
        "lines": [
            "Closed monthly books for six entities across three jurisdictions.",
            "Led the VAT implementation across the retail division in 2018.",
            "Reduced the month-end close from 11 working days to 6.",
        ],
        "edu": "BCom Accounting, ACCA (part qualified)",
    },
    "hr": {
        "titles": ["HR Generalist", "Talent Acquisition Specialist", "HR Manager"],
        "firms": ["Majid Al Futtaim", "Emirates NBD", "Jumeirah Group"],
        "skills": "SuccessFactors, Workday, UAE Labour Law, onboarding, ER casework",
        "lines": [
            "Ran end-to-end recruitment for 90+ roles a year across retail and head office.",
            "Rewrote the onboarding programme; 90-day attrition fell from 18% to 7%.",
        ],
        "edu": "BA Human Resource Management, CIPD Level 5",
    },
    "marketing": {
        "titles": ["Digital Marketing Manager", "Performance Marketing Specialist"],
        "firms": ["Namshi", "Ounass", "Mumzworld", "Sivvi"],
        "skills": "Google Ads, Meta Ads, GA4, SEO, HubSpot, Klaviyo",
        "lines": [
            "Managed AED 4.2m annual paid media budget at a blended 3.1 ROAS.",
            "Grew organic sessions 140% in eighteen months through content and technical SEO.",
        ],
        "edu": "BA Marketing",
    },
    "mechanical": {
        "titles": ["Mechanical Engineer", "HVAC Engineer", "Maintenance Engineer"],
        "firms": ["Drake & Scull", "ALEC", "Khansaheer", "Emrill"],
        "skills": "AutoCAD, Revit MEP, HAP, chiller plant optimisation, planned maintenance",
        "lines": [
            "Commissioned HVAC for a 42-storey residential tower in Business Bay.",
            "Cut chiller plant energy consumption 19% through sequencing changes.",
        ],
        "edu": "BEng Mechanical Engineering",
    },
    "security": {
        "titles": ["Security Analyst", "SOC Analyst", "Cybersecurity Engineer"],
        "firms": ["DarkMatter", "Help AG", "Injazat", "spiderSilk"],
        "skills": "Splunk, SIEM, incident response, MITRE ATT&CK, Python, Nessus",
        "lines": [
            "Tier 2 SOC analyst on a 24/7 rotation covering 40 client environments.",
            "Built detection rules that cut false positive volume by roughly half.",
        ],
        "edu": "BSc Cybersecurity",
    },
    "teaching": {
        "titles": ["Secondary Mathematics Teacher", "Head of Science", "Primary Teacher"],
        "firms": ["GEMS Education", "Taaleem", "Repton Dubai", "Dubai College"],
        "skills": "British curriculum, IGCSE, A-Level, differentiated instruction",
        "lines": [
            "Taught IGCSE and A-Level mathematics; 78% A*-A at A-Level in 2024.",
            "Led the department through a KHDA inspection rated Very Good.",
        ],
        "edu": "BEd Mathematics, PGCE",
    },
}

HEADING_STYLES = [
    {"summary": "Professional Summary", "exp": "Work Experience",
     "skills": "Technical Skills", "edu": "Education"},
    {"summary": "Profile", "exp": "Employment History",
     "skills": "Core Competencies", "edu": "Academic Qualifications"},
    {"summary": "Career Summary", "exp": "Professional Experience",
     "skills": "Key Skills", "edu": "Qualifications"},
]


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def make_filler(rng: random.Random, used_emails: set) -> dict:
    name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    email = f"{slug(name).replace('-', '.')}@example.com"
    if email in used_emails:
        return None
    used_emails.add(email)

    domain = rng.choice(list(DOMAINS))
    d = DOMAINS[domain]
    h = rng.choice(HEADING_STYLES)

    end = rng.randint(2023, 2026)
    mid = end - rng.randint(2, 4)
    start = mid - rng.randint(2, 5)
    firms = rng.sample(d["firms"], 2)
    titles = rng.sample(d["titles"], min(2, len(d["titles"])))
    if len(titles) == 1:
        titles *= 2
    lines = rng.sample(d["lines"], min(2, len(d["lines"])))

    years = end - start
    text = f"""{name}
{email} | +971 5{rng.randint(0, 6)} {rng.randint(100, 999)} {rng.randint(1000, 9999)} | {rng.choice(CITY)}
{rng.choice(NATIONALITY)} national | Residence visa | {rng.choice(['30 days', '60 days', '1 month', 'Immediate'])} notice

{h['summary']}
{titles[0]} with {years} years of experience. {lines[0]}

{h['exp']}

{titles[0]}, {firms[0]}
{rng.choice(['Jan', 'Mar', 'Jun', 'Sep'])} {mid} - Present
{lines[0]}
{lines[-1]}

{titles[1]}, {firms[1]}
{rng.choice(['Feb', 'Apr', 'Aug', 'Nov'])} {start} - {mid}
{lines[-1]}

{h['skills']}
{d['skills']}

{h['edu']}
{d['edu']}, {start - 1}
"""
    return {"email": email, "name": name, "file": f"{slug(name)}.txt",
            "text": text, "domain": domain}


def main(count: int = 40) -> None:
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.txt"):
        old.unlink()

    records = [dict(p, domain="planted") for p in PLANTED]
    used = {p["email"] for p in PLANTED}

    guard = 0
    while len(records) < count and guard < count * 20:
        guard += 1
        cv = make_filler(rng, used)
        if cv:
            records.append(cv)

    for r in records:
        (OUT / r["file"]).write_text(r["text"], encoding="utf-8")

    # The manifest is the authoritative candidate_id -> file mapping. On the live
    # VPS path n8n supplies email and name directly from the Claude extraction;
    # this file plays that role for the synthetic corpus.
    manifest = [{"file": r["file"], "candidate_id": r["email"], "name": r["name"],
                 "domain": r["domain"]} for r in records]
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    people = len({r["email"] for r in records})
    print(f"wrote {len(records)} CV files for {people} distinct candidates -> {OUT}")
    print(f"  planted: {len(PLANTED)} (incl. 1 near-duplicate pair)")
    print(f"  filler:  {len(records) - len(PLANTED)}")


if __name__ == "__main__":
    main()
