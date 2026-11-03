# Measurement artifacts

Every number quoted in `README.md` traces to one of the committed files below. This index
exists because the repo contains results from **four different retrieval configurations**
of the same 33 resumes, and three of them produced different headline correlations. A
file named `results_*.csv` is not a synonym for `results.csv`.

## How to read a run

| to learn | look at |
|---|---|
| which embedder ran | the `embedder` column in the CSV, or the first `embedder:` line of the `.log` |
| which floor was in force | `bias_q4.log` echoes `sim_floor`. **No `evaluate.py`/`sweep_floor.py` log here echoes it** - they predate that line. For sweeps, the chosen floor is the row marked `*`; for evals, use the filename plus the table below |
| whether the row is quotable | `modal_share` > 0.5 means most candidates got one score and the rho is carried by outliers |
| how a claim was computed | `README.md` names the file next to each table |

The floor in force is *not* recorded inside the eval logs, so any of the older CSVs can
only be attributed by filename plus this index. `evaluate.py --floor <x>` now prints the
value into the run's first log line, so runs from here on are self-describing.

## Evaluation runs (33 resumes x 2 modes)

| file | embedder | floor | scorer | rag rho | rag MAE | backs | status |
|---|---|---|---|---|---|---|---|
| `results.csv` | tfidf | 0.024 | Q4_K_M | **0.753** | 23.91 | headline RAG-vs-full table | headline. **Byte-identical to `results_floor024.csv` (md5 `f2ae77cd`): one process, not a reproduction.** A repeat run is still outstanding |
| `results_floor024.csv` | tfidf | 0.024 | Q4_K_M | 0.753 | 23.91 | same rows as `results.csv` | duplicate of the above |
| `per_resume.csv` / `per_resume_floor024.csv` | tfidf | 0.024 | Q4_K_M | - | - | per-candidate rows behind the floor, degeneracy (20 of 33 on score 20) and ±10 tables | identical pair, same md5 `ca20a7ee` |
| `results_dense.csv` | dense | 0.24 | Q4_K_M | 0.668 | 16.58 | "TF-IDF vs dense retrieval" table | single dense process; `modal_share` 0.576 (19 of 33 on 80) |
| `per_resume_dense.csv` | dense | 0.24 | Q4_K_M | - | - | the dense row of the degeneracy table | - |
| `results_floor030.csv` | tfidf | 0.030 | Q4_K_M | 0.560 | 27.76 | floor-sensitivity discussion | **superseded** by the swept 0.024 run |
| `results_q4_tfidf.csv` | tfidf | 0.030 | Q4_K_M | 0.560 | 27.76 | same measurement as `results_floor030.csv`, different filename | superseded; `eval_q4_tfidf.log` and `eval_rerun_tfidf.log` both print 0.560 |
| `results_tfidf_q4.csv` | tfidf | unrecorded | Q4_K_M | **-0.083** | 32.30 | - | **pre-floor run. No committed log backs it. Do not quote.** Its `full` row (0.084) is also pre-floor |
| `results_fallback.csv` | not in schema | unrecorded | retrieval-only | 0.306 | 54.48 | no-LLM baseline | written before the `embedder` column existed |
| `results_fallback_floor024.csv` | tfidf | 0.024 | retrieval-only | 0.306 | 54.48 | "No-LLM baseline" section | identical rho/MAE to the row above: below `SIM_ANCHORS["tfidf"]`'s 0.05 the fallback score is floor-insensitive, which is why the baseline cannot be read as evidence for the floor |

The `full` rows of the tfidf files all report rho 0.262 / MAE 30.48. That is not a
copy-paste error: `mode="full"` never looks at retrieved evidence, so with
`temperature=0` the same process-independent scores come out of any floor. They differ in
`peak_rss_mb` only.

## Floor sweeps

| file | embedder | what it is |
|---|---|---|
| `sweep_tfidf.log` | tfidf | floors 0.000-0.064+; peak str-weak gap **2.75 at 0.024** (marked `*`). Source of `SIM_FLOOR["tfidf"]` |
| `sweep_dense.log` | dense | the same grid on MiniLM similarities; peak **2.31 at 0.24**. Source of `SIM_FLOOR["dense"]`. First two lines are `sentence-transformers` progress-bar noise, not an error |

The two grids are not comparable value-for-value: TF-IDF unit-vector cosine and MiniLM
cosine live on scales ~10x apart, which is the entire reason `SIM_FLOOR` is a dict.

## Fairness audit

| file | variants | what it is |
|---|---|---|
| `bias_q4.csv` + `bias_q4.log` | 108 rows = 6 resumes x 18 comparisons | the Q4_K_M audit **with the placebo factor**. Headline: mean abs delta 13.39 for identity vs 14.72 for content-free placebo lines |
| `bias_retrieval_only_control.csv` | 108 rows | same variants scored by the no-LLM baseline, to separate scorer noise from model behaviour |
| `bias_results.csv` | 45 rows = 3 resumes x 4 factors | the pre-placebo audit. **Superseded**: without a placebo it could not attribute a uniform shift. Its numbers still appear in old notes - do not quote them |
| `bias_control.log` | 96 variants, no `sim_floor` field | retrieval-only control run from before the placebo existed |

## Dataset

| file | what it is |
|---|---|
| `data/resumes/*.txt`, `data/labels.csv`, `data/job_description.txt` | 33 labelled resumes (30 generated at seed 7 + 3 hand-written, labels 10/50/90) |
| `data/generate_dataset.py` | the generator. Deterministic at `--seed 7 --n 30` |
| `data/check_dataset.py` | integrity + drift checker. `python data/check_dataset.py` |
| `data/check_dataset.log` | its last run: **12 FAIL / 3 WARN**. The FAILs are the reversed experience ranges (12 lines across 10 files, e.g. `2023-2022`) that the shipped dataset still contains; the WARNs are the two 39/56-word hand-written CVs and the generator drift. The drift is confined to those same 12 `Experience:` lines, so the fix in `role_years()` changes nothing else |

The shipped resumes were measured with those backwards date ranges in them. `role_years()`
fixes the generator, but the dataset has **not** been regenerated, because that would
change chunk text for 10 of 33 resumes and invalidate every results CSV above. Regenerate
and re-evaluate together, or leave both alone.

## Scratch

`users.db` (login accounts), `models/*.gguf` (1,117 MB weights), `__pycache__/` and
`streamlit_run.log` (app startup, not a measurement) are git-ignored. Nothing else at the
top level is untracked-by-accident - if a new `results_*.csv` appears here, add it to the
table above in the same commit.
