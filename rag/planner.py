"""Classical-planning arm: choose the evidence by searching over actions, not by ranking.

rag/screen.py decides what goes into the prompt with a ranking: top-k chunks per
requirement by cosine similarity, de-duplicated, truncated to a word budget
(`select_evidence`). Nothing in that choice depends on what the passages actually say. A
planner makes it depend on the content: it keeps a state, takes actions that change the
state, and decides the order of its own steps until it reaches its goal.

Action model (STRIPS-style; one goal per requirement, satisfied at the shipped rule
`KEYWORD_MATCH_FRACTION` from rag/llm.py, reused here unchanged):

    explore(requirement, k, floor)
        precondition  this query has not been asked yet
        effect        the retriever's passages for it become readable - the planner
                      learns which passages they are and how similar, not what words
        cost          no prompt words: the requirement line is in the prompt either way

    relax(requirement, k, floor/2)
        precondition  a plan over the readable passages has been produced and this
                      requirement is still not satisfied
        effect        the same query with the similarity floor halved, so passages that
                      sat just below the floor become readable too
        cost          no prompt words

    read(passage)
        precondition  the passage is readable, not already read, and its words fit the
                      remaining budget (the first passage is exempt, as in
                      select_evidence, so a plan is never empty)
        effect        the passage's terms join the prompt's term set - coverage can only
                      rise, so state is monotone in the reads
        cost          the passage's word count against the same budget screen_resume
                      computes (full chunk words, because the prompt prints each one)

Goal: maximise the mean ask coverage of the requirements within that word budget.

Search: greedy best-first forward search with a beam. Nodes are ranked lexicographically
by (requirements left unsatisfied, then graded coverage, then prompt words spent, then
passage identity), which is a total order, so the plan is reproducible byte for byte.
Beam width 1 is pure greedy - take the single best next action. Sweeping 1 / 8 / 24 is
the test of whether looking ahead buys anything over that.

What this model assumes, stated before the numbers: the effect of `read` is known before
the action runs, because the index holds every chunk's text (`ResumeIndex.build` gets it
from `chunk_text`). So this is a budgeted maximum-coverage problem solved by forward
search, not information gathering under uncertainty, and the objective it optimises is
the objective it is graded on - which makes the planner the ceiling of what a
coverage-optimal selection can do, and an unusually favourable test of it rather than an
unbiased estimate of a screener that had to learn its goal (rag/learn.py and
trained_eval.log are the attempt at the unbiased version). Its honest advantage has to
come from *which* passages it spends the budget on and *when it stops*. At deployment
scale, where the CV lives in an external store and the fetch is what costs, `explore`
becomes a true sensing action - that part is not measurable on 33 synthetic CVs and is
not claimed here.
"""
import json
from dataclasses import dataclass, field

from .config import SIM_FLOOR, TOP_K
from .ingest import Chunk, split_requirements
from .llm import KEYWORD_MATCH_FRACTION, document_terms, keyword_coverage
from .screen import Result, build_prompt, evidence_budget

BEAM = 8                 # nodes kept per search ply
MAX_RELAX_ROUNDS = 3     # the floor is halved at most this many times per CV
MIN_RELAX_FLOOR = 1e-4   # below this the floor is filtering nothing anyway


@dataclass
class Step:
    """One action of the plan, with the state it left behind."""
    kind: str                 # "explore" | "relax" | "read"
    target: str               # requirement text or passage label
    detail: str
    words_spent: int
    unsatisfied_after: int
    score_after: int


@dataclass
class Plan:
    resume_id: str
    requirements: list
    budget: int
    beam: int
    steps: list = field(default_factory=list)
    passages: list = field(default_factory=list)      # [(label, Chunk)], document order
    coverages: list = field(default_factory=list)     # graded, one per requirement
    search_covs: list = field(default_factory=list)   # what the search believed
    matched: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    absent_terms: list = field(default_factory=list)
    hits_by_req: dict = field(default_factory=dict)   # requirement index -> retriever hits
    pool: tuple = ()           # the readable Chunks the winning plan was chosen from
    path: tuple = ()                                  # chunk idx, in read order
    terms: frozenset = frozenset()
    words_spent: int = 0
    reads: int = 0
    nodes_expanded: int = 0
    relax_rounds: int = 0
    stop_reason: str = ""

    @property
    def score(self) -> int:
        return round(100 * sum(self.coverages) / max(1, len(self.coverages)))

    @property
    def unsatisfied(self) -> int:
        return sum(1 for c in self.coverages if c < KEYWORD_MATCH_FRACTION)


@dataclass(frozen=True)
class _Node:
    """A state the search can be in. Immutable, so beam members never alias one another."""
    unsat: int
    mean_cov: float
    words: int
    read: frozenset          # chunk idx
    path: tuple              # chunk idx, in read order
    terms: frozenset
    covs: tuple


def _rank(node: _Node, order: str = "goal"):
    """Total order over states, and the one choice the planner has to make.

    `goal` is the classical one: distance to the goal first (requirements still
    unsatisfied), then the graded coverage, then economy. It treats a requirement as a
    predicate that is either true or not, which is what the action model says, and it can
    therefore prefer a passage that satisfies one requirement outright over a passage
    that spreads more total coverage across several. `coverage` drops the predicate and
    maximises the graded mean - which is the number the evaluation reports, so this
    ablation is also the honest price of planning against a goal instead of a score.
    Both are tie-broken by words spent and then by passage identity, so each is total and
    reproducible.
    """
    if order == "coverage":
        return (-node.mean_cov, node.words, tuple(sorted(node.read)))
    return (node.unsat, -node.mean_cov, node.words, tuple(sorted(node.read)))


def chunk_words(chunk: Chunk) -> int:
    return len(chunk.text.split())


def _ask(index, resume_id: str, requirement: str, k: int, floor: float):
    return index.search(requirement, resume_id, k=k, min_score=floor)


def _coverages(reqs, terms) -> tuple:
    return tuple(keyword_coverage(r, terms)[0] for r in reqs)


def _node(reqs, terms, path, words, covs) -> _Node:
    return _Node(sum(1 for c in covs if c < KEYWORD_MATCH_FRACTION),
                 sum(covs) / max(1, len(covs)), words, frozenset(path), tuple(path),
                 frozenset(terms), covs)


def _noisy_or(a, b):
    """Combine two per-requirement coverage estimates the way a union of evidence does."""
    return tuple(1 - (1 - x) * (1 - y) for x, y in zip(a, b))


def _search(reqs, pool: dict, terms_of: dict, budget: int, beam: int, values=None,
            order: str = "goal"):
    """Forward search over `read` actions; pool is the readable set {idx: Chunk}.

    Returns (best node, nodes expanded). Every prefix of a plan is itself a feasible
    plan, so the best node seen at any depth is returned, not only a leaf.

    With `values` (chunk idx -> per-requirement estimate) the search ranks states by
    what a *model* predicts a passage is worth and never consults the term rule; the
    passages it chose are still scored by the rule afterwards, so the comparison with the
    other arms stays fair. Without it, the search ranks on the rule - the untrained arm,
    which is what rag/learn.py exists to replace. `order` chooses between the goal
    distance and the graded score, as _rank documents.
    """
    start = _node(reqs, (), (), 0, _coverages(reqs, ()) if values is None
                  else tuple(0.0 for _ in reqs))
    best, frontier, expanded = start, [start], 0
    for _ in range(len(pool)):
        succ: dict = {}
        for state in frontier:
            expanded += 1
            unread = [i for i in pool if i not in state.read]
            if not unread:
                continue
            cheapest = min(chunk_words(pool[i]) for i in unread)
            for i in sorted(unread):
                chunk, w = pool[i], chunk_words(pool[i])
                # the first read is exempt from the budget, as in select_evidence
                if state.words + w > budget and (state.path or w > cheapest):
                    continue
                terms = state.terms | terms_of[i]
                covs = (_noisy_or(state.covs, values[i]) if values is not None
                        else _coverages(reqs, terms))
                nxt = _node(reqs, terms, state.path + (i,), state.words + w, covs)
                current = succ.get(nxt.read)
                if current is None or _rank(nxt, order) < _rank(current, order):
                    succ[nxt.read] = nxt
        if not succ:
            break
        ranked = sorted(succ.values(), key=lambda n: _rank(n, order))
        frontier = ranked[:max(1, beam)]
        if _rank(frontier[0], order) < _rank(best, order):
            best = frontier[0]
    return best, expanded


def _record(plan: Plan, node: _Node, pool: dict) -> None:
    """Write the winning state into the plan: passages, coverages, matched lists.

    The coverages are always the term rule over the passages actually read, whether the
    search was driven by the rule or by a model's estimate of it - so an arm that planned
    on predictions is still graded on what its prompt really contained.
    """
    chosen = sorted((pool[i] for i in node.path), key=lambda c: c.idx)
    plan.passages = [(f"E{i + 1}", c) for i, c in enumerate(chosen)]
    # The readable set the plan was chosen from, which a later `relax` round can widen:
    # plan.hits_by_req is the last round's asking, not this plan's pool.
    plan.pool = tuple(pool[i] for i in sorted(pool))
    plan.coverages = [keyword_coverage(r, node.terms)[0] for r in plan.requirements]
    plan.search_covs = list(node.covs)
    plan.path, plan.terms = node.path, node.terms
    plan.words_spent, plan.reads = node.words, len(node.path)
    plan.matched, plan.missing, plan.absent_terms = [], [], []
    for r in plan.requirements:
        frac, absent = keyword_coverage(r, node.terms)
        (plan.matched if frac >= KEYWORD_MATCH_FRACTION
         else plan.missing).append(r)
        plan.absent_terms.extend(absent)


def plan_evidence(resume_id: str, jd: str, index, full_text: str, *,
                  beam: int = BEAM, budget: int = None, top_k: int = TOP_K,
                  floor: float = None, relax_rounds: int = MAX_RELAX_ROUNDS,
                  order: str = "goal", values: dict = None) -> Plan:
    """Search for the set of passages to put in the prompt, under a word budget.

    `budget` defaults to the same `evidence_budget(resume words)` the shipped RAG prompt
    gets, so the planner and `select_evidence` are handed the same number of words and
    differ only in how they spend them. `values` swaps the search's notion of what a
    passage is worth from the term rule to a trained estimate (rag/learn.py); the plan
    produced is still graded by the rule.
    """
    reqs = split_requirements(jd) or [jd]
    if budget is None:
        budget = evidence_budget(len(full_text.split()))
    if floor is None:
        floor = SIM_FLOOR.get(index.embedder.kind, 0.0)

    plan = Plan(resume_id=resume_id, requirements=reqs, budget=budget, beam=beam)
    candidates: dict = {}                       # chunk idx -> (best similarity, Chunk)
    for i, r in enumerate(reqs):
        hits = _ask(index, resume_id, r, top_k, floor)
        plan.hits_by_req[i] = hits
        for s, c in hits:
            if c.idx not in candidates or s > candidates[c.idx][0]:
                candidates[c.idx] = (s, c)

    best_rank, node, pool = None, None, {}
    for round_i in range(max(0, relax_rounds) + 1):
        if not candidates:
            # the sensing step came back empty - this is the CV that used to be scored
            # 9/100 with every tool named - so widen the ask before giving up on it
            relax_floor = floor / (2 ** (round_i + 1))
            if (round_i == relax_rounds or relax_floor < MIN_RELAX_FLOOR
                    or relax_rounds <= 0):
                plan.stop_reason = ("nothing to read: no passage of this CV cleared the "
                                    "similarity floor")
                break
            for i in range(len(reqs)):
                for s, c in _ask(index, resume_id, reqs[i], top_k, relax_floor):
                    candidates[c.idx] = (max(s, candidates.get(c.idx, (0.0,))[0]), c)
                plan.hits_by_req[i] = _ask(index, resume_id, reqs[i], top_k, relax_floor)
            plan.relax_rounds = round_i + 1
            continue
        pool = {idx: c for idx, (_, c) in candidates.items()}
        terms_of = {idx: frozenset(document_terms(c.text)) for idx, c in pool.items()}
        vals = None if values is None else {i: values[i] for i in pool if i in values}
        node, expanded = _search(reqs, pool, terms_of, budget, beam, values=vals, order=order)
        plan.nodes_expanded += expanded
        rank = _rank(node, order)
        if best_rank is None or rank < best_rank:
            best_rank = rank
            _record(plan, node, pool)
        if round_i == relax_rounds:
            break
        relax_floor = floor / (2 ** (round_i + 1))
        if relax_floor < MIN_RELAX_FLOOR:
            break
        widened = 0
        for i in (j for j, cov in enumerate(node.covs) if cov < KEYWORD_MATCH_FRACTION):
            for s, c in _ask(index, resume_id, reqs[i], top_k, relax_floor):
                if c.idx not in candidates:
                    widened += 1
                candidates[c.idx] = (max(s, candidates.get(c.idx, (0.0,))[0]), c)
        if not widened:
            break
        plan.relax_rounds = round_i + 1
        for i in range(len(reqs)):
            plan.hits_by_req[i] = _ask(index, resume_id, reqs[i], top_k, relax_floor)

    if not plan.coverages:
        plan.coverages = [0.0] * len(reqs)
    plan.stop_reason = plan.stop_reason or _stop_reason(plan, pool)
    plan.steps = _steps(plan, candidates, floor, top_k)
    return plan


def _stop_reason(plan: Plan, pool: dict) -> str:
    """Why the plan ends where it does, read off the state it ended in."""
    if not pool:
        return "nothing to read: no passage cleared the similarity floor"
    if plan.unsatisfied == 0:
        return "goal reached: every requirement satisfied"
    left = plan.budget - plan.words_spent
    affordable = [i for i in pool if i not in plan.path and chunk_words(pool[i]) <= left]
    if not plan.reads:
        return "budget smaller than any passage: the first read stands alone"
    if not affordable:
        return "budget spent: no unread passage fits the words left"
    return "search exhausted: no further read raises the coverage"


def _steps(plan: Plan, candidates: dict, floor: float, top_k: int) -> list:
    """The plan as an ordered step list, replayable action by action.

    The reads are replayed from the empty state, so every words_spent and score_after in
    here is produced by the same transition function the search used - the test suite
    re-plays this list and requires each step to be legal in the state before it.
    """
    steps = [Step("explore", r, f"top_k={top_k} floor={floor}", 0,
                  len(plan.requirements), 0) for r in plan.requirements]
    for rd in range(plan.relax_rounds):
        steps.append(Step("relax", f"{plan.unsatisfied} unsatisfied requirement(s)",
                          f"floor halved to {floor / 2 ** (rd + 1)}", plan.words_spent,
                          plan.unsatisfied, plan.score))
    terms: set = set()
    words = 0
    label_of = {c.idx: label for label, c in plan.passages}
    for idx in plan.path:
        chunk, w = candidates[idx][1], chunk_words(candidates[idx][1])
        terms |= document_terms(chunk.text)
        words += w
        covs = _coverages(plan.requirements, terms)
        steps.append(Step("read", label_of[idx],
                          f"{w} words (sim {candidates[idx][0]:.3f}) into the prompt",
                          words, sum(1 for c in covs if c < KEYWORD_MATCH_FRACTION),
                          round(100 * sum(covs) / max(1, len(covs)))))
    return steps


def screen_resume_planned(resume_id: str, jd: str, index, full_text: str, *,
                          beam: int = BEAM, budget: int = None, top_k: int = TOP_K,
                          floor: float = None, relax_rounds: int = MAX_RELAX_ROUNDS,
                          order: str = "goal", values: dict = None) -> Result:
    """Plan the prompt, score it with the keyword-coverage rule, hand back a Result.

    The same Result record the LLM and the two no-model scorers produce, so the shared
    `metric_row` and the app read any arm without knowing which one ran.
    """
    plan = plan_evidence(resume_id, jd, index, full_text, beam=beam, budget=budget,
                         top_k=top_k, floor=floor, relax_rounds=relax_rounds,
                         order=order, values=values)
    reqs = plan.requirements
    hits_per_req = [plan.hits_by_req.get(i, []) for i in range(len(reqs))]
    prompt = build_prompt(jd, reqs, hits_per_req, plan.passages)
    kept = {c.idx: label for label, c in plan.passages}
    evidence, evidence_refs = {}, {}
    for i, r in enumerate(reqs):
        refs = [kept[c.idx] for _, c in hits_per_req[i] if c.idx in kept]
        evidence_refs[r] = list(dict.fromkeys(refs))
        evidence[r] = [c.text for _, c in hits_per_req[i] if c.idx in kept]
    n_readable = len({c.idx for hits in hits_per_req for _, c in hits})
    rationale = (f"Planned evidence: {len(plan.passages)} of {n_readable} readable "
                 f"passages read, {plan.words_spent} of {plan.budget} words spent; "
                 f"{len(plan.matched)} of {len(reqs)} requirements satisfied. Absent "
                 "words: " + (", ".join(dict.fromkeys(plan.absent_terms)) or "none") + ".")
    raw = json.dumps({"score": plan.score, "matched": plan.matched,
                      "missing": plan.missing, "rationale": rationale})
    return Result(resume_id=resume_id, score=plan.score,
                  score_coverage=round(100 * len(plan.matched) / max(1, len(reqs))),
                  matched=plan.matched, missing=plan.missing, rationale=rationale,
                  evidence=evidence, evidence_refs=evidence_refs,
                  prompt_words=len(prompt.split()), raw=raw)
