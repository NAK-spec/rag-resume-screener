"""Generate a reproducible synthetic resume set with rubric-based human scores.

The 3 hand-written resumes in this folder are a smoke test: Spearman/Kendall on
n=3 is meaningless and the RAG-vs-full-context compression comparison needs resumes
long enough to contain text that retrieval can drop. This script writes ~30 more
plus a labels.csv, seeded so anyone can regenerate the exact same set.

HUMAN SCORE RUBRIC (this is what `human_score` encodes)
-------------------------------------------------------
Each of the six job requirements gets a credit level judged from the candidate's
qualification facts, never from name, school prestige, age or gender:

    proven   1.0  the requirement is backed by a concrete role or project
                   ("built a churn model that lifted recall by 12%")
    claimed  0.5  the skill only appears in the Skills line, no supporting work
    absent   0.0  no evidence anywhere in the document

Core requirements are weighted double:

    python_pandas 2   ml_sklearn 2   sql 2   visualization 1   nlp_llm 1   communication 1

    human_score = round(100 * sum(weight * credit) / sum(weight))

Labels therefore reflect intent (what the candidate actually has done), and the
model has to recover that from prose that paraphrases the skill instead of always
naming it. A model that only keyword-matches the Skills line will disagree with
these labels on the "claimed" cases, which is exactly the behaviour worth measuring.

Usage:  python data/generate_dataset.py [--n 30] [--seed 7] [--out data]
"""
import argparse
import random
import re
from pathlib import Path

# ---------------------------------------------------------------- rubric

SKILLS = ["python_pandas", "ml_sklearn", "sql", "visualization", "nlp_llm", "communication"]
WEIGHTS = {"python_pandas": 2, "ml_sklearn": 2, "sql": 2,
           "visualization": 1, "nlp_llm": 1, "communication": 1}
CREDIT = {"proven": 1.0, "claimed": 0.5, "absent": 0.0}

# ---------------------------------------------------------------- phrase pools
# `evidence` sentences demonstrate a skill in a role or project WITHOUT necessarily
# naming the tool, so retrieval and the LLM have to connect paraphrase to requirement.
EVIDENCE = {
    "python_pandas": [
        "wrote Python scripts with pandas to clean and merge {n} rows of messy operational data",
        "automated a manual spreadsheet report in Python, cutting the weekly job from {h} hours to {h2} minutes",
        "built a data-cleaning pipeline in Python that validated {n} records before loading them",
        "used pandas groupby and pivot tables to summarise {n} transactions by region and month",
        "replaced a fragile Excel macro with a Python script the whole team could rerun",
        "profiled {n} rows of survey responses in pandas to find missing-value patterns",
        "handled date parsing, merges and duplicate removal on {n} records using pandas",
        "did exploratory analysis in Python and shared the notebook so colleagues could reproduce it",
    ],
    "ml_sklearn": [
        "trained a churn prediction model with scikit-learn that improved recall by {pct} percent",
        "built a regression model in scikit-learn to forecast {domain} demand {pct} percent closer than the old rule",
        "ran a logistic classifier on {n} labelled cases and compared it against a random-forest baseline",
        "set up cross-validation and hyper-parameter search for a gradient-boosting model",
        "clustered {n} customers with k-means to help the {domain} team segment its outreach",
        "deployed a trained classifier as a small API and monitored its accuracy for a quarter",
        "did feature engineering on {domain} logs and improved the model's ROC-AUC by {pct} points",
        "built and evaluated a text classifier, then wrote up why the simpler model won",
    ],
    "sql": [
        "queried the {db} warehouse daily to prepare training and reporting data",
        "wrote multi-table joins and window functions to answer {domain} questions without an analyst",
        "optimised a slow report query, reducing its runtime from {h} minutes to {h2} seconds",
        "built views and scheduled extracts that {n} downstream users relied on",
        "reconciled data quality issues by auditing SQL aggregations against source tables",
        "described schema changes and wrote migration scripts for the {db} database",
        "pulled cohort metrics with SQL for weekly {domain} reviews",
        "used SQL plus indexes to make the dashboards load under {h2} seconds",
    ],
    "visualization": [
        "published interactive charts in Tableau and Streamlit that management used in {domain} reviews",
        "built a Power BI dashboard tracking {n} live accounts with drill-downs by region",
        "designed the visual language for a reporting pack: consistent scales, annotated thresholds",
        "turned a confusing weekly report into a one-page visual summary with matplotlib",
        "created dashboards that surfaced a {pct} percent drop in {domain} volumes the team had missed",
        "shared Plotly charts with non-technical colleagues and iterated on their feedback",
        "made a Streamlit app so operations staff could query results themselves",
        "visualised model errors in a confusion-matrix chart to guide the next iteration",
    ],
    "nlp_llm": [
        "fine-tuned a small language model for text classification and measured the accuracy lift",
        "built a retrieval-augmented chatbot over internal documents using sentence embeddings",
        "cleaned and vectorised {n} support tickets, then clustered them into issue categories",
        "ran sentiment analysis on {domain} feedback and reported the top recurring complaints",
        "compared embedding models for semantic search on an internal {n}-document corpus",
        "prompted and evaluated an LLM for draft summaries, logging the failure cases",
        "built a keyword-plus-embedding matcher that routed {n} requests to the right team",
        "evaluated two open-source language models on a labelled {domain} dataset",
    ],
    "communication": [
        "presented findings to non-technical stakeholders in {domain} and answered their questions live",
        "wrote the documentation and runbooks that new team members still use",
        "translated model limitations into plain language for the {domain} leadership team",
        "ran weekly demos of analytics work for a mixed technical and business audience",
        "co-authored the proposal that won internal funding for the {domain} project",
        "mentored {n2} interns on how to structure an analysis and report it honestly",
        "produced written handovers for cross-team projects with clear next steps",
        "explained data caveats in a report that changed a product decision",
    ],
}

# Only ever appears in the Skills line: the candidate claims it, work says nothing.
CLAIMED = {
    "python_pandas": "Python",
    "ml_sklearn": "machine learning",
    "sql": "SQL",
    "visualization": "data visualization",
    "nlp_llm": "NLP and large language models",
    "communication": "communication",
}

# How a proven skill shows up in the Skills line (real candidates list their tools).
SKILL_TOOL = {
    "python_pandas": "pandas",
    "ml_sklearn": "scikit-learn",
    "sql": "PostgreSQL",
    "visualization": "dashboards",
    "nlp_llm": "LLM APIs",
    "communication": "technical writing",
}

DOMAINS = ["fintech", "healthcare", "retail", "logistics", "energy", "telecom",
           "education", "agriculture", "public-sector", "e-commerce"]
# Titles and degrees track the qualification pattern: nobody with three data-science
# projects has a hospitality certificate, and nobody without technical evidence has
# held an analyst title. Uncorrelated metadata would be noise the model cannot recover.
ANALYST_ROLES = ["Data Analyst Intern", "Junior Data Engineer", "BI Analyst",
                 "Research Assistant", "Operations Analyst", "Marketing Analyst",
                 "Reporting Analyst", "Graduate Data Trainee"]
SUPPORT_ROLES = ["Receptionist", "Sales Associate", "Barista", "Warehouse Assistant",
                 "Front Desk Officer", "Customer Care Agent", "Teaching Assistant",
                 "Data Entry Clerk", "Administrative Assistant", "Event Coordinator"]
TECH_DEGREES = [
    "BSc Computer Science and Data Science", "BSc Statistics",
    "Diploma in Information Technology", "BSc Information Systems",
    "BEng Software Engineering", "BSc Mathematics",
    "MSc Applied Machine Learning", "Certificate in Business Analytics",
]
NONTECH_DEGREES = [
    "Certificate in Hospitality Management", "BA Business Administration",
    "BSc Biochemistry", "Diploma in Business Studies",
    "Uganda Advanced Certificate in Education", "BSc Economics",
    "High School Diploma", "Diploma in Graphic Design",
]
DBS = ["PostgreSQL", "MySQL", "SQL Server", "BigQuery", "SQLite"]
PRESENT = 2026          # this study's "now"; experience length is derived from it

# Irrelevant work text: the material RAG should be able to leave out of the prompt.
# {c} counts stay small (these are human-scale numbers), {n} stays large (data volumes).
FILLER = [
    "greeted visitors and handled {c} inbound calls per week at the front desk",
    "managed rosters and shift swaps for a team of {c2} staff",
    "reconciled the petty-cash float and prepared banking paperwork",
    "trained new joiners on the internal ticketing system",
    "coordinated a company event for {c3} attendees, including catering and venue",
    "handled escalations from dissatisfied customers and documented each outcome",
    "imaged and set up {c2} laptops for new starters",
    "kept the physical filing system and archive rooms in order",
    "supported the sales team with quotation paperwork and follow-up calls",
    "assisted with inventory counts in the warehouse {c2} times a year",
    "wrote meeting minutes and circulated them to {c2} participants",
    "volunteered at a community literacy club on weekends",
    "played in a local football league and captained the side",
    "ran the university chess club's social media page",
    "took a customer-service course covering tone, escalation and follow-up",
    "helped organise the annual staff retreat, travel and logistics included",
]
TAILS = [
    "References available on request.",
    "Hobbies: running, chess, cooking, reading science fiction.",
    "Objective: to secure a challenging position where I can grow my skills and contribute to organisational goals.",
    "Declaration: I certify that the information above is true to the best of my knowledge.",
    "Additional: available immediately, willing to relocate, driving licence class B.",
    "Courses: online certificate in project management, weekend workshop on spreadsheet basics.",
]
TYPOS = {"analysis": "analyis", "management": "managment", "received": "recieved",
         "separate": "seperate", "their": "thier", "successful": "succesful"}


def role_years(cursor: int, grad_year: int, span: int) -> tuple[int, int]:
    """Inclusive (start, end) years for one role, walking backwards from `cursor`.

    The old inline clamp was just max(grad_year, cursor - span). Once the cursor had
    stepped below graduation year, that returned a start *later* than its own end and the
    CV printed "2024-2023". Clamping to the cursor too keeps start <= end always; a role
    that can no longer fit above graduation year collapses to the single year it occupies.
    """
    return min(cursor, max(grad_year, cursor - span)), cursor


def render(rid: int, rng: random.Random, credits: dict, provenance: list | None = None):
    """Compose one resume from the candidate's qualification facts.

    `provenance`, when given, collects (skill, sentence) for every bullet that is written
    into the document - the rubric-relevant evidence tagged by the skill it demonstrates,
    and irrelevant filler tagged "filler". It costs nothing in the RNG stream, so the text
    is byte-identical with or without it, and `retrieval_quality.py` can then ask the
    pipeline whether a *known* piece of evidence reached the prompt instead of guessing at
    relevance from the surface text.
    """
    def record(tag, t):
        if provenance is not None:
            provenance.append((tag, t))

    domain = rng.choice(DOMAINS)
    proven = [s for s in SKILLS if credits[s] == "proven"]
    claimed = [s for s in SKILLS if credits[s] == "claimed"]
    tech = bool({"python_pandas", "ml_sklearn", "sql"} & set(proven))
    role_pool = ANALYST_ROLES if tech else SUPPORT_ROLES
    degree_pool = TECH_DEGREES if tech else (
        rng.choice([TECH_DEGREES, NONTECH_DEGREES]) if claimed else NONTECH_DEGREES)
    role = rng.choice(role_pool)
    degree = rng.choice(degree_pool)
    grad_year = rng.choice([2016, 2018, 2019, 2021, 2022, 2023, 2024, 2025])
    years = max(0, PRESENT - grad_year)
    # verbosity varies per document so RAG has irrelevant text it can drop
    bullets_per_skill = rng.choice([1, 2, 2, 3, 3, 4])
    n_filler = rng.randint(2, 6) if years <= 1 else rng.randint(6, 16)

    def fill(s):
        return (s.replace("{n}", str(rng.choice([500, 2000, 8000, 15000, 40000])))
                 .replace("{c}", str(rng.choice([40, 70, 120, 250])))
                 .replace("{c2}", str(rng.choice([3, 4, 6, 8, 12])))
                 .replace("{c3}", str(rng.choice([30, 60, 120, 200])))
                 .replace("{pct}", str(rng.choice([6, 9, 11, 12, 15, 18, 22])))
                 .replace("{h}", str(rng.choice([2, 4, 6])))
                 .replace("{h2}", str(rng.choice([8, 15, 30, 45])))
                 .replace("{db}", rng.choice(DBS))
                 .replace("{domain}", domain))

    # Experience: proven evidence, several phrasings per skill, plus irrelevant filler
    seen, exp = set(), []
    for s in proven:
        for ph in rng.sample(EVIDENCE[s], k=min(len(EVIDENCE[s]), bullets_per_skill)):
            t = fill(ph)
            key = re.sub(r"\d+", "#", t).lower()
            if key not in seen:
                seen.add(key)
                exp.append(t)
                record(s, t)
    for _ in range(n_filler):
        t = fill(rng.choice(FILLER))
        key = re.sub(r"\d+", "#", t).lower()
        if key not in seen:
            seen.add(key)
            exp.append(t)
            record("filler", t)
    rng.shuffle(exp)

    # Projects reuse skills in a different setting, never repeating an experience line
    proj = []
    for s in rng.sample(proven, k=min(len(proven), rng.randint(1, 3))) if proven else []:
        for ph in rng.sample(EVIDENCE[s], k=2):
            t = fill(ph)
            key = re.sub(r"\d+", "#", t).lower()
            if key not in seen:
                seen.add(key)
                proj.append(t)
                record(s, t)
                break

    lines = [f"Candidate R{rid:02d}", ""]
    opening = (f"{years} year{'s' if years != 1 else ''} of work experience, "
               f"most recently in a {domain} setting." if years else
               f"Recent graduate seeking a first role in a {domain} team.")
    lines.append(f"{degree}, {grad_year}. {opening}")

    # Real CVs list several roles; distributing the bullets across them is what makes
    # the document long enough that retrieval has irrelevant text to leave out.
    n_roles = 1 if years <= 2 else rng.choice([1, 2, 2, 3, 3, 4])
    per_role = [[] for _ in range(n_roles)]
    for i, t in enumerate(exp):
        per_role[i % n_roles].append(t)
    # most recent role first, walking backwards from today
    cursor = PRESENT
    span = max(1, round(years / n_roles)) if years else 1
    for r, bullets in enumerate(per_role):
        if not bullets:
            continue
        start, end = role_years(cursor, grad_year, span)
        cursor = start - 1
        title = role if r == 0 else rng.choice(role_pool)
        where = domain if r == 0 else rng.choice(DOMAINS)
        dates = f"{start}-{end}" if start != end else str(start)
        lines.append(f"Experience: {title} ({where}, {dates}).")
        lines += [f"  - {b}." for b in bullets]

    if proj:
        lines.append("Projects:")
        lines += [f"  - {p}." for p in proj]
    if rng.random() < 0.6:
        lines.append("Education details: " + rng.choice([
            f"coursework in {rng.choice(['databases, statistics and operating systems', 'linear algebra, probability and web development', 'statistics, research methods and economics'])}",
            f"final year project supervised by the {domain} analytics group",
            "graduated part-time while working"]) + ".")
    if rng.random() < 0.4:
        lines.append("Certifications: " + rng.choice([
            "cloud fundamentals badge", "spreadsheet associate certificate",
            "internal analytics programme, 2 modules", "data literacy workshop"]) + ".")
    skill_tokens = [CLAIMED[s] for s in claimed] + [SKILL_TOOL[s] for s in proven]
    skill_tokens += rng.sample(["Microsoft Excel", "Git", "Jira", "reporting",
                                "teamwork", "time management"], k=rng.randint(1, 3))
    lines.append("Skills: " + ", ".join(dict.fromkeys(skill_tokens)) + ".")
    # Personal details sit in the document but must never move the score
    lines.append(f"Languages: English, {rng.choice(['Luganda', 'Swahili', 'French', 'Arabic'])}.")
    if rng.random() < 0.5:
        lines.append(rng.choice(TAILS))
    text = "\n".join(lines)

    # light noise: real CVs are not proofread
    present = [w for w in TYPOS if w in text]
    for w in rng.sample(present, k=min(len(present), rng.randint(0, 2))):
        text = text.replace(w, TYPOS[w], 1)
    return text


def rubric_score(credits: dict) -> int:
    total = sum(WEIGHTS[s] * CREDIT[c] for s, c in credits.items())
    return round(100 * total / sum(WEIGHTS.values()))


def make_credits(rng: random.Random, tier: str) -> dict:
    """Assign a qualification pattern per tier, weighting the core skills."""
    core = ["python_pandas", "ml_sklearn", "sql"]
    soft = ["visualization", "nlp_llm", "communication"]
    credits = {s: "absent" for s in SKILLS}
    if tier == "strong":
        for s in core:
            credits[s] = "proven"
        for s in soft:
            credits[s] = rng.choice(["proven", "proven", "claimed"])
    elif tier == "medium":
        for s in rng.sample(core, 2):
            credits[s] = "proven"
        for s in [c for c in core if credits[c] == "absent"]:
            credits[s] = rng.choice(["claimed", "absent"])
        for s in rng.sample(soft, rng.randint(1, 2)):
            credits[s] = rng.choice(["proven", "claimed"])
    else:  # weak
        for s in rng.sample(core, 1):
            credits[s] = rng.choice(["claimed", "proven"])
        for s in rng.sample(soft, rng.randint(0, 1)):
            credits[s] = rng.choice(["claimed", "proven"])
    return credits


def build(n: int = 30, seed: int = 7):
    """Generate the whole set in memory: (name, tier, credits, text, provenance) per resume.

    This is the single source of the corpus - `main()` only writes what it returns, and
    `retrieval_quality.py` imports it to get ground-truth provenance for the same text
    the study scored. One construction path, so the two cannot disagree about which
    sentence was written to demonstrate which skill.
    """
    rng = random.Random(seed)
    # keep the score distribution wide enough for rank correlation to mean anything
    tiers = (["strong"] * round(n * 0.27) + ["medium"] * round(n * 0.4)
             + ["weak"] * (n - round(n * 0.27) - round(n * 0.4)))
    rng.shuffle(tiers)
    rows = []
    for i, tier in enumerate(tiers, 1):
        credits = make_credits(rng, tier)
        provenance = []
        text = render(i, rng, credits, provenance)
        rows.append((f"r{i:02d}_{tier}.txt", tier, credits, text, provenance))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()

    out = Path(args.out)
    resumes = out / "resumes"
    resumes.mkdir(parents=True, exist_ok=True)

    rows = build(args.n, args.seed)
    labels, words = [], []
    for name, tier, credits, text, _ in rows:
        (resumes / name).write_text(text, encoding="utf-8")
        labels.append((name, rubric_score(credits), tier))
        words.append(len(text.split()))

    existing = {}
    for old in ["strong_candidate.txt", "medium_candidate.txt", "weak_candidate.txt"]:
        if (resumes / old).exists():
            existing[old] = {"strong_candidate.txt": 90, "medium_candidate.txt": 50,
                             "weak_candidate.txt": 10}[old]
    csv = "resume_id,human_score\n" + "\n".join(
        f"{n},{s}" for n, s, _ in labels) + "".join(
        f"\n{k},{v}" for k, v in existing.items())
    (out / "labels.csv").write_text(csv + "\n", encoding="utf-8")

    scores = [s for _, s, _ in labels]
    print(f"wrote {len(labels)} resumes to {resumes} (seed={args.seed})")
    print(f"length: min {min(words)} / median {sorted(words)[len(words)//2]} / max {max(words)} words")
    print(f"rubric score: min {min(scores)} / mean {round(sum(scores)/len(scores))} / max {max(scores)}")
    for tier in ("strong", "medium", "weak"):
        ts = [s for _, s, t in labels if t == tier]
        print(f"  {tier:7s} n={len(ts):2d}  scores {sorted(ts)}")
    print(f"labels.csv -> {len(labels) + len(existing)} rows (incl. {len(existing)} hand-written)")


if __name__ == "__main__":
    main()
