# Presentation outline

Slide-by-slide plan for defending *"An Efficient Retrieval-Augmented Generation Framework with
Quantized Large Language Models for Automated Resume Screening"*.

**Every number here is already computed and committed.** Nothing in this deck needs a model run,
so it can be assembled and rehearsed on a laptop that should not be asked to decode tokens for
four hours. Each figure carries the artifact it comes from; if a slide cannot cite one, it is a
claim and does not belong in the deck. [ARTIFACTS.md](ARTIFACTS.md) is the index that says which
file is authoritative for which number - check it before you quote any `results_*.csv`.

**Read the "Do not say" column before you build the slides.** Half of what makes this work
defensible is that it does not overclaim, and a slide that overclaims hands the examiner the
rebuttal for free.

---

## Run of show (target 20 minutes, 20 slides + backup)

| # | slide | the one thing it says |
|---|---|---|
| 1 | Title | the framework, the constraint (quantized, CPU-only), the task (screen against a JD) |
| 2 | Problem | screening is slow, inconsistent and privacy-constrained; the CVs cannot leave the machine |
| 3 | Research question | can a 4-bit 1.5B model, fed retrieved evidence rather than the whole resume, rank candidates as well as it can read? |
| 4 | System | five stages, one number each (slide 4 diagram) |
| 5 | Corpus and labels | 30 generated + 3 hand-written CVs, rubric-derived labels, and the circularity that follows |
| 6 | Headline result | RAG rho 0.753; full context does not rank at all |
| 7 | Efficiency | 43.4% shorter prompt, 33.6% lower latency, same memory - and where it stops holding |
| 8 | The floor | the single design choice that turned a negative rho into 0.753 |
| 9 | Significance | permutation p, bootstrap CI, leave-one-out - and the paired test the intervals never gave |
| 10 | Main negative result | the LLM adds +0.447 of rho on TF-IDF and -0.062 on dense |
| 11 | Power | what this design could have detected, including the things it could not |
| 12 | Degeneracy | the scorer emits 5 distinct values; rho is band agreement, not fine ranking |
| 13 | What the retriever delivers | 1.03 passages, 25.3% of the CV's proof, 3.80 of 6 requirements marked `none` |
| 14 | The ceiling | +0.732 achievable on that block vs +0.721 measured: 0.268 of the gap is evidence, 0.011 is model |
| 15 | Over-claiming | on 24 of 30 CVs the model reported more matched requirements than the prompt carried |
| 16 | Geometry frontier | top-k is inert, the knee is in the chunking, and it costs 13.1% more prompt |
| 17 | Fairness | the audit found no identity effect separable from its own placebo - unresolved, not answered |
| 18 | Reproducibility | byte-identical repeats, self-checking scripts, indexed artifacts |
| 19 | Limitations | one slide, said out loud, before anyone asks |
| 20 | Contributions and next work | three contributions, one cheap next experiment |

---

## Slide by slide

### 1. Title

Project title, your name, the two constraints that define the whole study: **quantized (Q4_K_M)
and CPU-only**. Say the sentence the rest of the deck earns: *a 1.1 GB 4-bit model on a laptop,
guided by retrieval, ranks candidates at rho 0.753 against a rubric.*

### 2. Why this problem

- Screening is slow, inconsistent, and the input is personal data - so the interesting system is
  one that never sends CVs anywhere.
- Full-context LLM screening is the obvious baseline, and it is expensive per candidate and
  (as slide 6 shows) it does not discriminate.
- RAG promises cheaper prompts; quantization promises a cheaper model. **Nobody had shown what
  happens when you stack both on a task where the score has to be right.**

### 3. Research question, stated as a falsifiable pair

1. Does retrieval-augmented prompting with a 4-bit model rank candidates at least as well as
   feeding it the whole resume, for a smaller prompt?
2. If not, where is the loss: the retriever's evidence, or the quantized reader?

Question 2 is the one this deck answers best (slides 13-16), and it is worth saying so up front:
the answer is *the evidence*, and that answer came from measurement, not from intuition.

### 4. System

Five boxes, one number each, left to right:

```
CV text ──chunk_text──► 111 chunks (90 words, 20 overlap, 33 CVs)
   ──TF-IDF / MiniLM──►  embeddings (headline run: TF-IDF, ngram 1-2, sublinear)
   ──index.search──►     top-3 per requirement, then floor 0.024
   ──select_evidence──►  word budget = min(400, doc, max(120, 0.4·doc)), dedup by best match
   ──build_prompt──►     172 words mean  →  Qwen2.5-1.5B-Instruct Q4_K_M, temperature 0
```

Say what the floor does in plain terms: it is the only stage allowed to report *"nothing
supports this requirement"*, and that turns `missing` from a guess into an observation.

### 5. Corpus and labels - and the caveat that comes with them

- 30 generated CVs (weak/medium/strong tiers, one JD, 6 requirements) + 3 hand-written
  illustrations = 33. Labels span 10-100, mean 58.5, from a stated rubric over credited skills.
- The generator records which sentence it wrote to prove which skill. That provenance is what
  makes slides 13-16 measurable instead of argued.
- **Say this out loud:** labels are rubric-derived, so agreement means the pipeline reproduces
  the rubric, not that it makes good hiring judgements. Do not let this slide be discovered in Q&A.

### 6. Headline result

| mode | Spearman rho | MAE | within ±10 | prompt words |
|---|---|---|---|---|
| RAG | **0.753** | 23.9 | 0.333 | **172** |
| full context | *undefined* | 31.2 | 0.273 | 304 |

(`results_v2.csv`, 33 candidates, TF-IDF, floor 0.024.)

**Undefined, not bad:** full context gave all 33 candidates exactly 85 - including the
hospitality CV labelled 10 - so it has no rank. `evaluate.py` returns NaN rather than a
tie-swollen number. The honest sentence is "full context does not rank", not "full context ranks
badly", and its MAE is still worse (31.2 vs 23.9).

### 7. The efficiency claim, priced

- Prompt 304 → 172 words (**-43.4%**); median 297 → 177. Latency 20.1 → **13.3 s/candidate**
  (-33.6%), throughput 9.5 → 10.0 tok/s, model load 1.71 s, peak RSS 2,079 vs 2,119 MB.
- **It embedder-independently holds**: the dense run reproduces -41.5% on the same corpus,
  measured twice per embedder.
- **And it does not always hold:** on the 3 CVs under 90 words (39, 56, 85 words) RAG is
  *longer* by a constant 21 words, because there is nothing for retrieval to leave out. Claim:
  "compresses any resume with more than one chunk."

Do not say: "cheaper" without naming the machine. 4 threads, one CPU laptop; the timing columns
are a property of that box, and the repo says so.

### 8. The one design choice that mattered most

Corpus-wide RAG Spearman by floor: **-0.083** (no floor) → **0.560** (0.03) → **0.753** (0.024).
(`results_tfidf_q4.csv`, `results_floor030.csv`, `results_v2.csv`; grids in `sweep_tfidf_v2.log`.)

- With no floor, top-k always returns *something*: at 0.000 every CV gets evidence for all 6
  requirements and strong-minus-weak separation is exactly 0.00. A budget that must fill is a
  budget that fabricates.
- The floor costs some true evidence (51.5% of requirements end up unevidenced) and no input
  tokens (median 178 → 177 words), but it costs ~2 s/candidate of output time, because the model
  spends tokens listing what is missing. Frame it as bought, not free.
- The same pathology in prompt form is slide 6's full-context row: hospitality CV (10), medium
  (50), strong (90) → 85, 85, 85. Under RAG: 0, 20, 85.

### 9. Significance

(`significance.py` → `significance.log`; 20,000 permutation draws, seed 7.)

| claim | result |
|---|---|
| RAG rho is not chance | p < 1.0e-04 (tfidf), 1.0e-04 (dense); bootstrap 95% [+0.565, +0.858] |
| not carried by one CV | leave-one-out moves rho by at most 0.040 |
| TF-IDF vs dense embedder | Δrho +0.083, CI [-0.175, +0.351], paired p = 0.514 - **nothing** |
| LLM vs its own no-LLM baseline | tfidf Δrho **+0.447** CI [+0.098, +0.805], p = 0.0095 |

Two things to say here because they are the difference between a statistics slide and a credible
one: the embedder comparison was **paired** (same 33 candidates, so two overlapping intervals
were never the right test), and bootstrap CIs are used instead of Fisher-z because the scorers
emit 4-24 distinct values.

### 10. Main negative result

The model's contribution depends on the retriever, and only one direction survives:

| retriever | no-LLM ranking | Q4_K_M RAG | paired Δrho |
|---|---|---|---|
| TF-IDF | 0.306 (p = 0.087, CI covers 0) | 0.753 | **+0.447** |
| dense MiniLM | **0.732** | 0.670 | **-0.062** (CI [-0.312, +0.191]) |

With a good enough retriever and a threshold, **the quantized LLM adds nothing to the ranking at
n = 33**. Report this as a finding, not a failure: it localises where the model earns its cost.

### 11. Power - what this design could have detected

(`power_analysis.py` → `power_analysis.log`, tie-aware simulator calibrated against the same
permutation rule as slide 9.)

- Smallest true rho detectable at 80% power: **0.456-0.473**.
- Power against the embedder difference: **15.1%**; against the dense LLM-vs-baseline difference:
  **9.6%**. Reaching 80% needs **>250 candidates**.
- Power against the effects actually found: 100.0% / 99.9% / 99.8% - except the TF-IDF no-LLM row,
  **41.6%**, which is why "not significant" there is not "no effect".
- The scorer's own granularity caps rho at **0.864** (4-value) / 0.857 (5-value), so part of the
  distance from 0.670 to a perfect ranking is the scale, not the model.

This slide is what turns "n = 33 is small" from an accusation into a quantified answer.

### 12. Read the correlations as band agreement

RAG mode produced **5** distinct scores across 33 candidates (61% on one value); full context
produced **1**. Every rho reported is between a 5-level score and a 0-100 rubric. `evaluate.py`
prints `n_distinct_scores` and `modal_share` rather than hiding it. The tool is a **shortlisting**
device (rho 0.753), not a scoring device (MAE 23.9, ±10 hit rate 0.333).

### 13. What the retriever actually delivers

(`retrieval_quality.py` → `retrieval_quality.log`, no LLM, measured against the generator's own
proof-to-skill record.)

| shipped config | tfidf | dense |
|---|---|---|
| passages kept per CV | **1.03** | 1.13 |
| proven bullets delivered whole | **25.3%** | 34.1% |
| requirements printed `none` | **3.80 of 6** | 2.27 |
| pasted words per CV | 68.9 | 77.0 |

The compression that makes this "RAG" is exactly what removes the evidence: three quarters of the
sentences the CV contains as proof never reach the model, and it takes one passage per CV to do
it. Self-checked four ways, including a replay that reproduces 30/30 shipped resumes word-for-word.

### 14. The gap is the evidence's, not the quantization's

| | ceiling (perfect reader of that block) | Q4_K_M measured |
|---|---|---|
| tfidf | rho **+0.732**, MAE 16.03 | rho **+0.721**, MAE 24.80 |
| dense | rho **+0.804**, MAE 11.00 | rho +0.648, MAE 16.53 |

**Splitting the shortfall on the same 30 CVs: +0.268 of rho belongs to the evidence block,
+0.011 to the model.** The 4-bit reader is essentially at the ceiling of what it was handed;
the headroom is in retrieval. (On MAE the model does give away more: 16.03 achievable vs 24.80.)

Say the discipline out loud: a ceiling is the rubric applied to what arrived. It bounds any
reader. It is not a score the pipeline got.

### 15. The model contradicts its own prompt

The prompt prints `[i] none` for every requirement it has no passage for. On **24 of 30** CVs
(tfidf) the model reported more requirements matched than that leaves room for - mean 3.87
claimed against 2.20 actually evidenced. On the stricter reading (a requirement counts only if
its own proof bullet arrived whole), **24 of 30** tfidf and **27 of 30** dense.

All 116 `matched` entries are canonical requirement strings, so it is not a parse artifact: an
instruction printed in the prompt is not being followed. `r01_weak.txt` matched 5 of 6 while 5
of its requirements were printed as `none`.

Scope it honestly: this documents an unfollowed citation instruction, **not** a candidate who
fabricated a skill.

### 16. The geometry frontier, measured

(`geometry_sweep.py` → `geometry_sweep.log`, 18 cells per embedder, no model anywhere in it.)

- **Top-k is inert at this budget:** 1 → 8 moves the ceiling by **0.000** of rho and the prompt
  by 0.1 words, and the pasted text is identical (68.9 words). What limits the shipped run is the
  word budget, not `TOP_K = 3`.
- **The knee is in the chunking:** 45/10 reaches ceiling rho **+0.916** at 196 prompt words
  (**+13.1%** prompt for **+0.184** ceiling); reaching 0.90 by widening the budget costs 247.
- **Composition beats volume:** 45/10 delivers *less* proof whole (24.0% vs 25.3%) and one more
  proven skill-instance (44 vs 43 of 86), yet ranks 0.184 of rho better - because it is 2.27
  passages spread across requirements instead of 1.03 concentrated in one.
- **The floor is claim discipline at every geometry:** at 90/20 it costs 0.5 points of proof and
  adds 3.43 requirements per CV marked `none`; its largest cost anywhere (10.1 points) is at one
  chunk per CV, where **5 of 30 CVs are left with no passage at all**.

Then the sentence that keeps this slide honest: **no model was run at any of these cells**, so
these are ceilings and a priced next experiment, not results.

### 17. Fairness - unresolved, not answered

`bias_audit.py`: 6 base resumes × 19 variants = 108 comparisons against a baseline, with a
**placebo** line that carries no identity, and signed deltas so a model that docks 5 points for
one name and adds 5 for another cannot hide behind absolute values.

Result: **no identity effect separable from its placebo**; the only identity signal came from a
single candidate (`n_effective = 1`) and the instrument's own noise is 20-60 points.

Do not say "the model is unbiased." Say the audit could not separate identity from noise, so the
identity question is open, and the instrument is too coarse to close it.

### 18. Reproducibility

- `temperature = 0`; two independent processes per configuration agree on **all 33** scores, all
  `matched` lists and every quality figure (both embedders). Only the timings move (~4%).
- The analysis scripts self-check and refuse to print otherwise: the delivery replay reproduces
  30/30 resumes word-for-word, `rubric(credits) == labels.csv` for 30/30, and the geometry sweep
  must reproduce the committed prompt lengths and passage counts before it reports a single cell.
- A dataset defect (10 of 30 CVs had experience ranges running backwards) was found by
  `data/check_dataset.py`, fixed, and re-measured: **0 of 33** tfidf scores changed - the fix was
  cosmetic, and that is a measurement rather than an assumption.
- `run_study.ps1` reproduces every artifact; every artifact is indexed in `ARTIFACTS.md`,
  including which files must not be quoted.

### 19. Limitations (read them, do not skip them)

- Labels are synthetic and rubric-derived; agreement is rubric reproduction, not hiring judgement.
- **n = 33**: MDE ~0.46, so the small effects (embedder, dense LLM-vs-baseline) were undetectable
  by construction, not disproven.
- The scorer emits 5 distinct values; part of the rho deficit is scale (ceiling 0.864).
- Latency and throughput are from one 4-thread CPU laptop and will not transfer; `avg_tok_per_s`
  is end-to-end (prefill included), so it is lower than decode-only numbers llama.cpp reports.
- Only **Q4_K_M** was benchmarked: no fp16, no other quant, no other size - so this measures a
  point on the quantization curve, not the curve.
- Full context degenerates to one score, so RAG-vs-full rests on the ordered-vs-constant contrast
  plus MAE (23.9 vs 31.2), not on two overlapping intervals.
- The fairness question is unresolved. `peak_rss_mb` is process-wide, so RAG-vs-full memory within
  a run is not separable.

### 20. Contributions and what I would do next

Contributions, each backed by a committed artifact:

1. A working end-to-end pipeline that screens on a CPU-only laptop with 4-bit inference, and the
   prompt/latency numbers to show what that costs (-43.4% prompt, -33.6% latency).
2. A **measurement of where the accuracy lives in such a system**: the score ceiling against
   provenance shows 0.268 of the rho gap is the evidence block and 0.011 is the quantized model,
   and the geometry sweep prices the fix (45/10: +0.184 ceiling for +13.1% prompt).
3. An honest account of two failure modes that the literature tends to skip: retrieval that
   fabricates evidence when a budget must fill (rho -0.083 → 0.753 with the floor), and a scorer
   that over-claims matched requirements against its own prompt (24 of 30 CVs).

Next work, in cost order:

1. Re-run the study at chunk 45/10, fixed budget - **the cheap one**, since the frontier is
   already measured: ~2 model runs, +13% prompt, ceiling +0.184. Re-calibrate `SIM_FLOOR` at the
   new chunking first (it was swept at 90/20).
2. Widen the corpus toward n > 250 only if you need to resolve the embedder comparison - the power
   slide says exactly what 33 can and cannot support.
3. A finer scorer scale (more output granularity, or rubric-decomposed sub-scores) before any new
   model size, because the 0.864 granularity ceiling is currently the binding constraint.

---

## Ten-minute cut

Keep slides 1, 4, 6, 8, 10, 14, 16, 19-20. That is still: system, headline, the choice that
mattered, the negative result, where the accuracy lives, the priced fix, the limits, and the
contribution. Drop the rest rather than rushing them.

## Demo policy

Do not run the live 4-bit model on the laptop during the talk - thermal throttling will make the
latency numbers you are quoting look wrong, and it adds a nondeterminism risk to nothing, since
every number is already in an artifact. Either show the Streamlit app in **Retrieval-only (no
LLM)** mode (no GGUF loaded, the ranking and per-requirement evidence still browse) or play one
pre-recorded screening. `streamlit_run.log` is app startup, not a measurement - never quote it.

## Expected questions, and the artifact that answers them

| question | answer | source |
|---|---|---|
| "Why only Q4_K_M?" | Scope decision: one quantization point measured end-to-end rather than several measured shallowly. It is a limitation, stated on slide 19. | README Limitations |
| "Isn't your rho meaningless because labels come from the rubric your pipeline encodes?" | Yes for validity-as-hiring-judgement, no for the claim being made (does the pipeline reproduce the rubric, and where does it lose information). The permutation test asks exactly that and the deck says so. | `significance.log`, slide 5 |
| "Your full-context baseline is broken, not worse." | Correct and deliberate: 33/33 identical scores is the finding - full context does not rank. MAE 31.2 vs 23.9 still stands. | `results_v2.csv`, slide 6 |
| "0.753 with 5 distinct scores?" | Reported as band agreement; `n_distinct_scores` and `modal_share` are printed by the tool itself, and the granularity ceiling is 0.864. | `evaluate.py`, slides 11-12 |
| "You never tested the geometry change you recommend." | The frontier is measured with no model (18 cells/embedder) and the cheapest cell is priced; what is *not* done is scoring a model at it, and slide 16 says so. | `geometry_sweep.log` |
| "Did your dataset bug change your results?" | 0 of 33 tfidf scores, 0 prompt lengths, 0 passages; 1 of 33 under dense. Old and new artifacts are both committed. | `results.csv` vs `results_v2.csv` |
| "Is it biased?" | Unknown: no effect separable from placebo, `n_effective = 1`, instrument noise 20-60 points. | `bias_q4.log` |

## Do not say

| tempting sentence | why it is wrong |
|---|---|
| "The quantized model scores 0.916 with smaller chunks." | No model ran at that geometry; 0.916 is a ceiling on a perfect reader of that block. |
| "RAG beats full context on accuracy." | Full context has no rank (rho undefined). The comparison is ordered-vs-constant plus MAE. |
| "TF-IDF retrieval is better than dense." | Measured and nothing: Δrho +0.083, CI [-0.175, +0.351], p = 0.514, and 15.1% power. |
| "The LLM improves ranking." | Only on TF-IDF (+0.447). On dense it is -0.062 and the heuristic alone reaches 0.732. |
| "The model hallucinates skills." | It over-claims matched requirements against its prompt markers; that is an unfollowed instruction, not evidence about the candidate. |
| "The system is fair / unbiased." | The audit could not separate identity from its placebo. Unresolved. |
| "13.3 s per candidate." | Add "on this 4-thread CPU laptop", and use `avg_latency_s`, not `avg_tok_per_s`, for comparisons. |
| "We should switch to 45/10 chunks." | Reasonable and priced, but the floor was swept at 90/20 and no model ran at 45/10. Present as the next experiment. |

## If you want slides, not an outline

This file is the source of truth for content. Say the word and it can be turned into a `.pptx`
(one slide per section, the tables as tables, the "do not say" list kept as speaker notes) - but
the numbers should not be re-derived from the deck; they come from the artifacts above.
