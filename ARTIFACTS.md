# Measurement artifacts

Every number quoted in `README.md` traces to one of the committed files below. This index
exists because the repo contains results from **two corpora x four retrieval configurations**
of the same 33 resumes, and three of those configurations produced different headline
correlations. A file named `results_*.csv` is not a synonym for `results_v2.csv`.

## The two corpora

On 2026-11-03 the dataset was found to contain 12 `Experience:` date ranges running backwards
(`2023-2022`) across 10 of the 30 generated CVs, and the generator was fixed. Every artifact
below is tagged with which corpus it was measured on:

| tag | corpus | how to tell |
|---|---|---|
| **v2** | corrected | filename contains `_v2`; `data/check_dataset.py` reports 0 FAIL / 2 WARN and no drift |
| pre | superseded corpus | everything without `_v2`; `check_dataset.log` at git `b043eb8` reported 12 FAIL / 3 WARN |

**The headline is the v2 pair (`results_v2.csv` + `results_v2_repeat.csv`), not `results.csv`.**
The dense side is a pair too (`results_v2_dense.csv` + `results_v2_dense_repeat.csv`), so all
four corrected-corpus LLM files are two reproductions of two configurations. Each pair carries
identical quality numbers - 0 of 66 per-resume rows differ in either pair - which is the point:
the corpus defect was cosmetic for retrieval, and its one real effect was to *hide* how
degenerate the full-context baseline is (2 distinct scores became 1). Read each pair as one
reproduced result, and read `results.csv` as its audit trail.

## How to read a run

| to learn | look at |
|---|---|
| which corpus | `_v2` in the filename, or the table above |
| which embedder ran | the `embedder` column in the CSV, or the first `embedder:` line of the `.log` |
| which floor was in force | every `results_v2*` / `eval_v2*` log echoes `sim_floor:`. The pre-correction logs **do not** - their floor is whatever `rag/config.py` held at the time, recoverable only from the filename and this index |
| whether the row is quotable | `modal_share` > 0.5 means most candidates got one score and the rho is carried by outliers; empty `spearman`/`kendall` means the mode emitted one single value and no rank exists |
| how a claim was computed | `README.md` names the file next to each table |

## Evaluation runs (33 resumes x 2 modes)

**Corrected corpus - these are the quotable ones.**

| file | embedder | floor | scorer | rag rho | rag MAE | status |
|---|---|---|---|---|---|---|
| `results_v2.csv` | tfidf | 0.024 | Q4_K_M | **0.753** | 23.91 | **headline.** Backed by `eval_v2.log`, which echoes `embedder: tfidf \| sim_floor: 0.024` |
| `results_v2_repeat.csv` | tfidf | 0.024 | Q4_K_M | **0.753** | 23.91 | **independent second process.** 0 of 66 mode x resume rows differ on score, `matched`, passages or prompt words; only latency moves (13.33 -> 12.85 s). This is what makes 0.753 a reproduced number |
| `per_resume_v2.csv` / `per_resume_v2_repeat.csv` | tfidf | 0.024 | Q4_K_M | - | - | per-candidate rows behind the floor, degeneracy (20 of 33 on score 20), +-10, coverage-divergence and prompt-length tables |
| `results_v2_fallback.csv` | tfidf | 0.024 | retrieval-only | 0.306 | 54.48 | "No-LLM baseline" section. rho/tau/MAE identical to the pre-correction `results_fallback_floor024.csv` - the baseline is floor-insensitive below its `SIM_ANCHORS` 0.05, so it cannot see the corpus edit either |
| `per_resume_v2_fallback.csv` | tfidf | 0.024 | retrieval-only | - | - | per-candidate fallback rows (12 distinct scores, modal share 0.303 - it is coarse but not degenerate) |
| `results_v2_fallback_dense.csv` | dense | 0.24 | retrieval-only | **0.732** | 26.03 | the same scorer on the dense embedder (`eval_v2_fallback_dense.log`), no GGUF. **This is the number the dense LLM row has to be compared against**: 24 distinct scores, modal share 0.091, within-10 0.333, coverage rho 0.679 |
| `per_resume_v2_fallback_dense.csv` | dense | 0.24 | retrieval-only | - | - | its per-candidate rows. `rag` and `full` are identical by construction - this scorer never sees the prompt |
| `results_v2_dense.csv` | dense | 0.24 | Q4_K_M | 0.670 | 16.55 | **dense headline on the corrected corpus**, backed by `eval_v2_dense.log` (`embedder: dense \| sim_floor: 0.24`). `modal_share` 0.576 (19 of 33 on 80), 4 distinct scores |
| `per_resume_v2_dense.csv` | dense | 0.24 | Q4_K_M | - | - | its per-candidate rows; also the file that shows `full` mode scoring 85 for all 33 candidates under *both* embedders |
| `results_v2_dense_repeat.csv` | dense | 0.24 | Q4_K_M | 0.670 | 16.55 | **independent second dense process**, backed by `eval_v2_dense_repeat.log`. Quality columns identical to the row above in both modes; **its `avg_latency_s` (21.2 rag / 12,812.7 full) and `model_load_s` (9.91) are void - do not quote them.** See "Latency is not comparable across files" |
| `per_resume_v2_dense_repeat.csv` | dense | 0.24 | Q4_K_M | - | - | 66 rows, 0 differing from `per_resume_v2_dense.csv` on score, coverage, `matched`, `missing`, passages, prompt words or output words. One row is stamped 421,575 s of latency (the laptop suspended mid-run); every other column is fine |

**Superseded corpus - retained so the correction is checkable, not because they are quotable.**

| file | embedder | floor | scorer | rag rho | rag MAE | note |
|---|---|---|---|---|---|---|
| `results.csv` | tfidf | 0.024 | Q4_K_M | 0.753 | 23.91 | **byte-identical to `results_floor024.csv`** (md5 `f2ae77cd`): one process, not a reproduction. Same quality figures as `results_v2.csv`, different corpus |
| `results_floor024.csv` | tfidf | 0.024 | Q4_K_M | 0.753 | 23.91 | duplicate of the above |
| `per_resume.csv` / `per_resume_floor024.csv` | tfidf | 0.024 | Q4_K_M | - | - | duplicate pair (md5 `ca20a7ee`) |
| `results_dense.csv` | dense | 0.24 | Q4_K_M | 0.668 | 16.58 | pre-correction dense run, single process. Superseded by `results_v2_dense.csv`, but the two agree (0.668 / 16.58 vs 0.670 / 16.55) after exactly one candidate's score moved |
| `per_resume_dense.csv` | dense | 0.24 | Q4_K_M | - | - | its per-candidate rows; `full` mode here is what proved full-context scores are embedder-independent |
| `results_floor030.csv` | tfidf | 0.030 | Q4_K_M | 0.560 | 27.76 | the floor A/B (0.030 vs the swept 0.024) |
| `results_q4_tfidf.csv` | tfidf | 0.030 | Q4_K_M | 0.560 | 27.76 | **not a copy of the row above** - a second process at the same floor, differing only in timing. Two independent runs agreed byte-for-byte on quality before the schema recorded `sim_floor` |
| `results_tfidf_q4.csv` | tfidf | unrecorded (0.0 = no floor) | Q4_K_M | **-0.083** | 32.30 | the pre-floor ablation. **No committed log backs it.** Quoted only as the left end of the floor curve, never as a headline |
| `results_fallback.csv` | not in schema | unrecorded | retrieval-only | 0.306 | 54.48 | written before the `embedder` column existed |
| `results_fallback_floor024.csv` | tfidf | 0.024 | retrieval-only | 0.306 | 54.48 | the same baseline on the old corpus |

The `full` rows of the pre-correction tfidf files all report rho 0.262 / MAE 30.48. That is
not a copy-paste error: `mode="full"` never looks at retrieved evidence, so with
`temperature=0` the same scores come out of any floor. They differ in `peak_rss_mb` only.
On the **corrected** corpus every `full` row of the four LLM files (`results_v2.csv`,
`results_v2_repeat.csv`, `results_v2_dense.csv`, `results_v2_dense_repeat.csv`) reports
**empty** rho and `modal_share` 1.0 - all 33 candidates score 85 - which is the finding, not a
regression.

### Latency is not comparable across files

Each LLM run is a separate process on one shared 4-thread CPU, so wall-clock columns drift
with host state while quality columns do not. Measured on this corpus:

| comparison | same config? | latency | output length | verdict |
|---|---|---|---|---|
| `results_v2.csv` vs `results_v2_repeat.csv` (rag) | yes | 13.33 -> 12.85 s | identical | ~4% host noise - the reproduction |
| `results_v2_dense.csv` vs `results_v2_dense_repeat.csv` (rag) | yes | 12.64 -> 21.22 s mean, 12.47 -> 13.68 s **median** | identical | void, not noise: the laptop suspended inside this run. The median moved ~1 s; the mean is dragged by rows that straddled the sleep. Quote the median or nothing |
| `results_dense.csv` vs `results_v2_dense.csv` (rag) | different corpus | 16.19 -> 12.64 s | 90.8 -> 88.1 words | all 33 candidates faster; host state, not content |
| rag vs full *inside* any one file | - | -34% to -37% | - | **the only latency claim worth quoting** |

The dense repeat is the cautionary case for the whole index: a run that is *perfect* on every
quality column and *unusable* on every timing column, because suspend/resume counts as elapsed
time. `per_resume_v2_dense_repeat.csv` has one row at 421,575 s (4.9 days, a `full` generation
that spans the sleep) and a `full`-mode mean of 12,812.7 s against a 27.4 s median. Greedy
decoding cannot be hurt by a sleep; `time.time()` can.

RSS is the exception: 2,079 / 2,119 / 2,463 / 2,487 MB are stable enough between processes to
compare, because the number is dominated by loaded weights rather than by CPU contention.
So `results_v2_dense.csv` being 12.6 s and `results_v2.csv` 13.3 s must **not** be read as
"dense retrieval is faster"; it means the dense process got a quieter machine.

## Floor sweeps

| file | embedder | corpus | what it is |
|---|---|---|---|
| `sweep_tfidf_v2.log` | tfidf | corrected | floors 0.000-0.080; peak str-weak gap **2.75 at 0.024** (marked `*`). Source of the current `SIM_FLOOR["tfidf"]`. Differs from the old grid in one row (0.008: gap 1.69 vs 1.78) |
| `sweep_dense_fine_v2.log` | dense | corrected | 0.030-step grid across the peak. Source of `SIM_FLOOR["dense"]` = 0.24 and of the "flat from 0.24 to 0.27" reading in README |
| `sweep_dense_v2.log` | dense | corrected | 0.050-step grid out to 0.500; shows the collapse past the peak (gap 0.02 at 0.500). First two lines are `sentence-transformers` progress-bar noise, not an error |
| `sweep_tfidf.log` / `sweep_dense.log` | tfidf / dense | superseded | the same grids on the old corpus. Kept to show that re-deriving the floors on the corrected data lands on the same two values |

The two grids are not comparable value-for-value: TF-IDF unit-vector cosine and MiniLM cosine
live on scales ~10x apart, which is the entire reason `SIM_FLOOR` is a dict. Neither sweep
involved the LLM.

## Fairness audit

| file | variants | what it is |
|---|---|---|
| `bias_q4.csv` + `bias_q4.log` | 108 rows = 6 resumes x 18 comparisons | the Q4_K_M audit **with the placebo factor**, tfidf at floor 0.024. Headline: mean abs delta 13.39 for identity vs 14.72 for content-free placebo lines. The only log in the repo that echoed `sim_floor` before the `_v2` runs |
| `bias_retrieval_only_control.csv` | 108 rows | the same 114 variants scored without an LLM, to separate scorer noise from model behaviour: max delta 2 points |
| `bias_results.csv` | 45 rows = 3 resumes x 4 factors | the pre-placebo audit. **Do not quote**: without a placebo it could not attribute a uniform shift, and the claim it supports ("names lower the score") is the one the placebo dissolved |
| `bias_control.log` | 96 variants, no `sim_floor` field | retrieval-only control run from before the placebo existed |

**Not re-run on the corrected corpus, and that is checkable rather than asserted.**
`bias_audit.py --limit 6` takes the first six resumes in sorted order
(`medium_candidate.txt`, `r01_weak.txt`-`r05_strong.txt`); none of those ten files is among
the ten the correction edited, so the audit's inputs are byte-identical across both corpora.
The audit is also the one measurement whose conclusion does not depend on the score scale:
the placebo moved scores *more* than the identity factors, so a shift in either direction
still fails to separate them.

## Dataset

| file | what it is |
|---|---|
| `data/resumes/*.txt`, `data/labels.csv`, `data/job_description.txt` | 33 labelled resumes (30 generated at seed 7 + 3 hand-written, labels 10/50/90) - **the corrected text** |
| `data/generate_dataset.py` | the generator. Deterministic at `--seed 7 --n 30`. `role_years()` is the date-range fix |
| `data/check_dataset.py` | integrity + drift checker. `python data/check_dataset.py` (no model needed) |
| `data/check_dataset.log` | its last run: **0 FAIL / 2 WARN, no generator drift**. The 2 WARNs are the 39- and 56-word hand-written CVs, under the 60-word floor the evidence budget was sized for - they are illustrations of the pipeline, not efficient cases |
| git `b043eb8` | the checker's previous output (**12 FAIL / 3 WARN + drift**), i.e. the defect and its fix, both in history |

## Do not quote

- `results_tfidf_q4.csv` (-0.083) as a headline or as "RAG does not work" - it is the
  no-floor ablation, has no backing log, and its `full` row (0.084) is also pre-floor.
- `results.csv` / `results_floor024.csv` as a *reproduction* of `results_v2.csv` - same
  numbers, different corpus, and they are one process between them, not two.
- `bias_results.csv` - pre-placebo.
- Any rho from a `full` row on the corrected corpus - there is none; the mode emits one value.
- Retrieval-only `rho 0.306` as "retrieval ranks the resumes on its own" - permutation
  p = 0.087 and the bootstrap interval covers zero (`significance.log`). Use it only as the
  paired floor the LLM beats, which is significant.
- "The quantized LLM out-ranks retrieval", full stop - it does on TF-IDF (Δrho +0.447,
  p = 0.0095) and does not on dense (Δrho −0.062, CI [−0.312, +0.191]). Name the embedder, or
  quote the defensible version: ranking value over the weak baseline, calibration value
  (MAE 16.55 vs 26.03, p = 0.044) over the strong one.
- `results_v2_fallback_dense.csv`'s rho 0.732 as "the no-LLM pipeline is as good as the LLM"
  without the paired number next to it. The two rows differ in *what they see*: the heuristic
  scores every requirement by its best-matching chunk, the prompt carries only the budgeted
  passages. The comparison is scorer-vs-scorer on identical retrieval, not "with RAG" vs
  "without".
- Any ordering of the two embedders from 0.753 vs 0.670 - the Fisher-z intervals at n = 33
  overlap almost completely ([0.553, 0.871] vs [0.424, 0.824]), and the second process on each
  side changes that: identical repeats under `temperature=0` demonstrate reproducibility, not
  precision, so no interval narrows.
- `avg_latency_s`, `avg_tok_per_s` or `model_load_s` from `results_v2_dense_repeat.csv` - that
  process spanned a laptop suspend, which inflates elapsed time without changing a token. Use
  the medians, or the first dense run.
- "The floor was re-tuned on the corrected corpus" as a change - re-deriving both grids lands
  on 0.024 and 0.24 again; the dense argmax at 0.25 is 2 requirement-matches out of 198 and
  was deliberately not chased (README, "Why the dense floor stays at 0.240").

## Prompt inflation, measured without an LLM

| file | what it is |
|---|---|
| `prompt_inflation.log` | `python prompt_inflation.py` - rebuilds the pre-dedup prompt exactly as `rag/screen.py` stood at commit `9d2e10a` (`TOP_K=4`, no floor, no dedup, no budget) and measures it against the current path on the corrected corpus. Evidence block **1,517 pasted words vs a 222-word document = 6.8x**; whole prompt **1,671 vs the 304-word full-context baseline**; current path **69 / 0.37x** and **172** prompt words. No model, no GGUF, runs in seconds |
| `prompt_inflation.py` | the script. It carries a note that the prompt *totals* conflate two fixes (the old path also pasted the whole job description), which is why the evidence-block figure is the one quoted for dedup |

**This replaced a number that could not be checked.** Earlier drafts of README claimed "304 ->
416 words" for the same bug; that was measured live during the fix and never saved, so no
committed file backed it. Re-deriving it from the git-recovered code gives 1,671 mean, not 416
- the old figure was almost certainly measured on the three short hand-written CVs before the
30 generated resumes existed. Quote `prompt_inflation.log`, not the retired 416.

## Significance tests, measured without an LLM

| file | what it is |
|---|---|
| `significance.log` | `python significance.py` (seed 7, 20,000 draws) - permutation p-value, bootstrap 95% interval, Fisher-z for contrast, and a leave-one-out rho span for every quotable correlation, then four **paired** comparisons that resample the shared 33 candidates together. Cheap: **6.1 s** warm, 14.4 s cold, measured. It reads the committed `per_resume_v2*.csv` by name, so it is a re-analysis of the runs above, not a new measurement |
| `significance.py` | the script. Its first output line is a self-check: the vectorised tie-aware Spearman must equal `scipy.stats.spearmanr` on 200 tie-heavy random vectors (max abs error **1.1e-16**), because every number in the log depends on it. Verified reproducible - three invocations produced identical logs (the only diff since is a UTF-16 BOM from a PowerShell redirect, not a number) |

**What it changed in the prose, and why the old wording must not come back:**

- `rho 0.753` and `rho 0.670` are significant (p < 1.0e-04 and p = 1.0e-04) and neither is
  carried by one CV (leave-one-out spans +0.727..+0.793 and +0.634..+0.712).
- The embedder comparison is **paired**, not two intervals: Δrho +0.083, CI
  [−0.175, +0.351], P(TF-IDF better) 0.743, two-sided 0.514. Quote that, not "the intervals
  overlap".
- **The headline negative result:** on the dense retriever the LLM does not beat its own
  no-LLM baseline. `results_v2_fallback_dense.csv` ranks at **0.732** with no generation,
  against the LLM's 0.670 - paired Δrho **−0.062**, CI [−0.312, +0.191], P(LLM better) 0.314.
  On TF-IDF the same comparison is Δrho +0.447 (p = 0.0095). Any sentence of the form "the
  quantized LLM out-ranks retrieval" must name the embedder, because it is only true of one.
- **Withdraw as a ranking result:** the TF-IDF retrieval-only rho 0.306 (permutation p = 0.087,
  CI [−0.067, +0.644] covers zero). It is still the correct measure of *retrieval ordering* and
  of what the LLM is being compared against - the LLM's advantage over it is significant
  (Δrho +0.447, CI [+0.098, +0.805], p = 0.0095) - but "retrieval alone ranks these resumes"
  is not supported at n = 33 on that embedder. It *is* supported on dense (0.732, p < 1.0e-04).
- **Withdraw as an advantage:** dense's MAE 16.55 over TF-IDF's 23.91 (Wilcoxon p = 0.176;
  dense closer on 11 candidates, TF-IDF on 7, 15 tied). The *shape* difference - TF-IDF's
  one-sided error needing recalibration, dense's not - is descriptive of these 33 and stays.
- The one LLM-vs-heuristic comparison that does survive on dense is accuracy, not rank: MAE
  16.55 vs 26.03, lower on 20 of 33 candidates (11 worse, 2 equal), Wilcoxon p = 0.044.
- Full-mode coverage rho −0.267 is not significant (p = 0.129); the full-mode score rho is
  undefined, so the mode contributes no rank claim at all.

## Scratch

`users.db` (login accounts), `models/*.gguf` (1,117 MB weights), `__pycache__/` and
`streamlit_run.log` (app startup, not a measurement) are git-ignored. Nothing else at the
top level is untracked-by-accident - if a new `results_*.csv` appears here, add it to the
table above in the same commit.
