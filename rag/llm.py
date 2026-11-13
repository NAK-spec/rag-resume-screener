"""Quantized LLM wrapper (llama.cpp / GGUF) with a retrieval-only fallback."""
import json
import os
import re
import time
from dataclasses import dataclass

import psutil

from .config import MAX_NEW_TOKENS, N_CTX, N_THREADS


@dataclass
class GenResult:
    text: str
    latency_s: float
    new_tokens: int
    rss_mb: float

    @property
    def tok_per_s(self) -> float:
        return self.new_tokens / self.latency_s if self.latency_s > 0 else 0.0


def _rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


class QuantizedLLM:
    """Loads a GGUF model (any quantization level) with llama-cpp-python."""

    def __init__(self, path: str, name: str = ""):
        from llama_cpp import Llama
        self.name = name or os.path.basename(str(path))
        self.size_mb = os.path.getsize(path) / 1e6
        t0 = time.perf_counter()
        self.llm = Llama(model_path=str(path), n_ctx=N_CTX,
                         n_threads=N_THREADS, verbose=False)
        self.load_s = time.perf_counter() - t0

    def generate(self, system: str, user: str, meta=None) -> GenResult:
        t0 = time.perf_counter()
        out = self.llm.create_chat_completion(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_tokens=MAX_NEW_TOKENS, temperature=0.0)
        dt = time.perf_counter() - t0
        text = out["choices"][0]["message"]["content"]
        ntok = out.get("usage", {}).get("completion_tokens", len(text.split()))
        return GenResult(text, dt, ntok, _rss_mb())


# Similarity anchors used to turn a raw cosine score into a 0-1 fit score.
# Dense MiniLM and TF-IDF cosine live on very different scales, so a single
# threshold made the fallback score a strong candidate 10/100.
SIM_ANCHORS = {
    "dense": (0.20, 0.55),
    "tfidf": (0.05, 0.30),
}


class RetrievalOnlyLLM:
    """Fallback used when no GGUF model is available (demo / tests).

    Produces the same JSON schema as the real LLM, but derives the score from
    the retrieval similarities passed in `meta`. Lets the whole app and the
    evaluation harness run end to end without a model download.
    """
    name = "Retrieval-only (no LLM)"
    size_mb = 0.0
    load_s = 0.0

    def generate(self, system: str, user: str, meta=None) -> GenResult:
        t0 = time.perf_counter()
        meta = meta or {}
        reqs = meta.get("requirements", [])
        sims = meta.get("req_scores", [])
        lo, hi = SIM_ANCHORS.get(meta.get("embedder", "dense"), SIM_ANCHORS["dense"])
        fit = [min(1.0, max(0.0, (s - lo) / (hi - lo))) for s in sims]
        matched = [r for r, f in zip(reqs, fit) if f >= 0.5]
        missing = [r for r, f in zip(reqs, fit) if f < 0.5]
        score = round(100 * sum(fit) / max(1, len(fit)))
        text = json.dumps({
            "score": int(max(0, min(100, score))),
            "matched": matched, "missing": missing,
            "rationale": "Score derived from semantic similarity between job "
                         "requirements and the most relevant resume passages."})
        return GenResult(text, time.perf_counter() - t0, len(text.split()), _rss_mb())


# A requirement is a handful of asks; a CV is a document. Coverage asks how many of
# those appear anywhere in it, which is the question a no-model scorer can answer
# honestly. 0.6 is the fraction above which the requirement counts as matched: at 0.5 a
# requirement like "Knowledge of SQL and relational databases" is called matched on
# "databases" alone, which is not SQL evidence at all.
KEYWORD_MATCH_FRACTION = 0.6

# JD boilerplate: function words, quantifiers, and the verbs every requisition uses for
# "can do". Counting them would let a CV pass on "experience" alone, and would make a
# two-sentence company description look like a requirement. The set is stored as stems,
# because that is what both sides of the comparison are reduced to before the lookup.
_JD_FILLER_WORDS = {
    "a", "able", "across", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "candidate", "can", "capable", "comfortable", "could", "done", "demonstrated",
    "demonstrates", "desirable", "eg", "etc", "excellent", "experience", "experienced",
    "familiar", "familiarity", "for", "from", "good", "had", "has", "have", "in",
    "include", "included", "including", "into", "is", "it", "its", "knowledge",
    "knowledgeable", "least", "more", "most", "must", "of", "on", "or", "our", "plus",
    "preferred", "proficiency", "proficient", "required", "skill", "skills", "solid",
    "should", "such", "strong", "than", "that", "the", "their", "them", "these",
    "this", "those", "to", "understanding", "use", "used", "using", "was", "we",
    "well", "were", "will", "with", "work", "working", "would", "you", "your",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9+#.\-]*|[A-Za-z]")
PAREN_RE = re.compile(r"\(([^()]*)\)")
# Commas, semicolons and "or" introduce alternatives: "Power BI, Tableau, Plotly" is one
# ask satisfied by any of them, not three asks. "and" is deliberately not here -
# "Python and SQL" asks for both.
ITEM_SPLIT_RE = re.compile(r"[,;]|\bor\b", re.I)
# The words that hand a list over. In "Bachelor's degree in Computer Science, Data
# Science, ..." the degree itself is the ask; the names after "in" are alternatives.
LIST_LEAD = {"a", "an", "at", "by", "for", "in", "of", "on", "the", "to", "with"}

# Applied in order, first match only, to the CV and the requirement alike. Folding cannot
# invent a match; it can only stop "data visualisation" from being a different word than
# "visualization", or "Trained" from a different word than "training".
SPELLING_FOLDS = (("isations", "ization"), ("isation", "ization"),
                  ("ised", "ize"), ("ises", "ize"), ("ising", "ize"), ("ise", "ize"),
                  ("ysed", "yze"), ("yses", "yze"), ("ysing", "yze"), ("yse", "yze"))
ENDING_FOLDS = (("ying", "y"), ("ied", "y"), ("ies", "y"),
                ("ing", ""), ("edly", ""), ("ed", ""), ("es", ""), ("s", ""))


def spelling_key(token: str) -> str:
    """The stem a word stands for: punctuation, British spelling and inflection removed.

    Two words match only when this returns the same string for both, so the fold is the
    whole story: "training" and "trained" both become "train", "queries" and "querying"
    both become "query", "databases" and "database" both become "databas".
    """
    t = token.strip(".-+#").lower()
    for tail in ("'s", "'"):
        if t.endswith(tail):
            t = t[:-len(tail)]
            break
    for suffix, repl in SPELLING_FOLDS:
        if t.endswith(suffix) and len(t) - len(suffix) >= 4:
            t = t[:len(t) - len(suffix)] + repl
            break
    else:
        for suffix, repl in ENDING_FOLDS:
            if (t.endswith(suffix) and not (suffix == "s" and t.endswith("ss"))
                    and len(t) - len(suffix) >= 3):
                stem = t[:len(t) - len(suffix)] + repl
                if (suffix in ("ing", "ed") and len(stem) >= 4
                        and stem[-1] == stem[-2] and stem[-1].isalpha()):
                    stem = stem[:-1]          # running -> run, planning -> plan
                t = stem
                break
    if len(t) > 4 and t.endswith("e"):
        t = t[:-1]
    return t


# Folded once, at import, so the lookup sees the same reduced form the words do.
JD_FILLER = {spelling_key(w) for w in _JD_FILLER_WORDS}


def _fragments(text: str) -> list:
    """The words of a text, with hyphenated compounds split into their parts.

    "Hands-on" and "hands on" have to yield the same words, or a CV that writes the
    phrase without the hyphen would read as a document that never said it.
    """
    out = []
    for token in TOKEN_RE.findall(text):
        if not any(ch.isalpha() for ch in token):
            continue                       # bare numbers are not evidence of a skill
        for part in re.split(r"[.\-]+", token):
            if len(part) > 1 and any(ch.isalpha() for ch in part):
                out.append(part)
    return out


def document_terms(text: str) -> set:
    """The stems a document contains. A stem missing from here is a word it never said."""
    return {spelling_key(p) for p in _fragments(text)}


def _asks(fragment: str) -> list:
    """(stem, as-spelled) for the content words of one requirement fragment."""
    out = []
    for part in _fragments(fragment):
        k = spelling_key(part)
        if k not in JD_FILLER and k not in [x for x, _ in out]:
            out.append((k, part))
    return out


def _split_list_lead(fragment: str) -> tuple:
    """(subject, first name) for the text in front of a comma list.

    "Bachelor's degree in Computer Science" asks for a degree and offers Computer Science
    as one name it would accept. Without the split the subject's words are welded to the
    first name, so no CV could satisfy either of them.
    """
    words = fragment.split()
    cut = -1
    for i, w in enumerate(words):
        if w.endswith(":") or spelling_key(w) in LIST_LEAD:
            cut = i
    if cut < 0:
        return "", fragment
    return " ".join(words[:cut + 1]), " ".join(words[cut + 1:])


def requirement_groups(requirement: str) -> list:
    """One group per ask: each group is a list of alternatives, each alternative a word set.

    A group with more than one alternative is an any-of - a parenthetical example list
    ("(e.g. Power BI, Tableau, Plotly)") or a comma/"or" run of names. Collapsing those
    runs is what stops a six-name degree list from reading as six required words. An
    alternative is met only when all of its own words are present, so "Data Science"
    satisfies the degree list and a CV that merely says "science" does not.
    """
    def one(word):                     # a group that asks for exactly one word
        return [[word]]

    groups = []
    head = PAREN_RE.sub(" ", requirement)
    for span in PAREN_RE.findall(requirement):
        items = [it for it in (_asks(x) for x in ITEM_SPLIT_RE.split(span)) if it]
        if len(items) > 1:
            groups.append(items)           # examples: any one of them is evidence
        else:
            groups.extend(one(w) for w in (items[0] if items else []))
    parts = [p for p in ITEM_SPLIT_RE.split(head) if p.strip()]
    if len(parts) > 1:
        subject, first = _split_list_lead(parts[0])
        items = [it for it in (_asks(x) for x in [first] + parts[1:]) if it]
        if items and max(len(it) for it in items) <= 3:
            groups.extend(one(w) for w in _asks(subject))
            groups.append(items)
        else:
            groups.extend(one(w) for w in _asks(head))
    else:
        groups.extend(one(w) for w in _asks(head))
    return [g for g in groups if g]


def keyword_terms(requirement: str) -> list:
    """Every stem the requirement asks for, alternatives included, in first-seen order."""
    seen = []
    for group in requirement_groups(requirement):
        for alt in group:
            for k, _ in alt:
                if k not in seen:
                    seen.append(k)
    return seen


def keyword_coverage(requirement: str, doc_terms: set) -> tuple:
    """(fraction of the requirement's asks the document meets, words it could not find).

    Absent words come back the way the requirement spelled them, because they are quoted
    to a person: "NLP", not "nlp". For an any-of ask the report names the closest
    alternative - the one whose words were nearly all there.
    """
    groups = requirement_groups(requirement)
    if not groups:
        return 0.0, []
    met, absent = 0, []
    for group in groups:
        best = None
        for alt in group:
            missing = [(k, t) for k, t in alt if k not in doc_terms]
            if not missing:
                best = None
                break
            if best is None or len(missing) < len(best):
                best = missing
        if best is None:
            met += 1
        elif len(group) > 1:
            absent.append(" ".join(t for _, t in best))
        else:
            absent.extend(t for _, t in best)
    return met / len(groups), absent


class KeywordCoverageLLM:
    """Whole-document keyword coverage, selectable wherever no LLM should be loaded.

    This is deliberately NOT the retrieval score: chunk cosine answers "how much do
    these two blobs of text overlap", and a single keyword inside a 90-word chunk
    scores ~0.06 no matter how exactly it matches. Coverage answers the question a
    no-model screener is actually asked - is this term in the CV - and it is
    retriever-independent, so it does not change with the TF-IDF/dense choice. What it
    cannot do is tell a skill named in passing from one demonstrated in work; that is
    the sentence the LLM is in the loop for.
    """
    name = "Keyword coverage (no LLM)"
    size_mb = 0.0
    load_s = 0.0

    def generate(self, system: str, user: str, meta=None) -> GenResult:
        t0 = time.perf_counter()
        meta = meta or {}
        reqs = meta.get("requirements", [])
        doc_terms = document_terms(meta.get("full_text", ""))
        coverages, matched, missing, absent_terms = [], [], [], []
        for r in reqs:
            cov, absent = keyword_coverage(r, doc_terms)
            coverages.append(cov)
            (matched if cov >= KEYWORD_MATCH_FRACTION else missing).append(r)
            absent_terms.extend(absent)
        score = round(100 * sum(coverages) / max(1, len(coverages)))
        text = json.dumps({
            "score": int(max(0, min(100, score))),
            "matched": matched, "missing": missing,
            "rationale": (f"Keyword coverage: {len(matched)} of {max(1, len(reqs))} "
                          "requirements had at least "
                          f"{int(KEYWORD_MATCH_FRACTION * 100)}% of their asks present "
                          "in the CV. Absent words: "
                          + (", ".join(dict.fromkeys(absent_terms)) or "none") + ".")})
        return GenResult(text, time.perf_counter() - t0, len(text.split()), _rss_mb())


def load_llm(path):
    """Return a QuantizedLLM if the file + runtime exist, else the fallback."""
    try:
        if path and os.path.exists(str(path)):
            return QuantizedLLM(str(path))
    except Exception as e:  # missing llama_cpp, bad file, etc.
        print(f"[llm] could not load {path}: {e}")
    return RetrievalOnlyLLM()
