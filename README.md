# Can Language Models Do the Coder's Job? Assigning Regulatory Terms to Medical Device Adverse Event Narratives

Data, model answers, scores and code for the paper of that title (Tomi Thomo and Alex Thomo, submitted to Network Modeling Analysis in Health Informatics and Bioinformatics, 2026). The study asks nine language models to assign IMDRF device-problem (Annex A) and patient-effect (Annex E) codes to the narratives of voluntary MedWatch reports in FDA's MAUDE database, and measures agreement with the codes FDA's own intake staff assigned from the same text.

Everything in the paper's tables and figures is generated from the files here by `code/scripts/paper_tables.py`.

## Layout

```
data/      the study set, the retrieval context, the subsample lists, the verdicts, the code tables
answers/   every model answer
scores/    per-report scores for every model, task, arm and report
code/      the pipeline, from download to the paper's tables
paper/     numbers.txt, every number that appears in the paper
MANIFEST.md  size and SHA-256 of every file
```

## data/

**`device_release.parquet`, `patient_release.parquet`.** The cleaned voluntary study set, one row per report, for the device task (40,859 reports) and the patient task (26,065). Columns that matter:

| Column | Meaning |
|---|---|
| `mdr_report_key` | FDA's report key, joins to openFDA |
| `date_received` | YYYYMMDD |
| `event_type`, `report_source_code`, `reporter_occupation_code` | as in openFDA |
| `writer` | `lay` (patient or family), `professional`, `unknown` (manufacturer notifications and empty occupation); see the paper, Section 3.4 |
| `product_code` | FDA three-letter device category |
| `device_codes`, `patient_codes` | FDA's filed IMDRF codes after cleaning, the reference |
| `device_levels`, `patient_levels` | hierarchy level of each code |
| `text_event` | the narrative with the intake note stripped, what every model saw |
| `had_note` | whether the narrative carried FDA's intake note before stripping |
| `set` | `pool`, `comparison` or `unseen`; see the paper, Section 3.2 |
| `in_device_task`, `in_patient_task` | membership |

The narratives are FDA's published text with the intake note removed. In this release copy, and only here, `code/scripts/release_mask.py` has additionally replaced e-mail addresses, phone numbers, street addresses, SSN-shaped numbers and honorific-plus-surname mentions with FDA's own `(b)(6)` marker; `mask_counts.tsv` gives the counts (206 device and 146 patient narratives, 0.5%). The models were run on the published text before this substitution.

**`context_device.parquet`, `context_patient.parquet`.** The retrieval context for every test report (unseen and comparison sets): `shortlist_B` and `shortlist_C` (the candidate codes of Arms B and C), `example_keys` and `example_sims` (the eight retrieved pool reports and their cosine similarities), `prior_top1`, `nn_codes` and `knn_top1` (the three no-model baselines), and the coverage columns. `context_coverage.tsv` summarises coverage by set and writer.

**`frontier_subsample_*.tsv`.** The fixed 500-report subsample per task on which the frontier models ran, with set and writer group.

**`cases_verdicts.tsv`.** The hand classification of the 60 drawn disagreement cases (58 after two duplicates), one row per case: task, case number, report key, kind, note.

**`codebook_A_E.tsv`.** The IMDRF Annex A and E codes used, with FDA's legacy code, hierarchy level, status and path. The IMDRF definitions are not redistributed; they are in IMDRF/AE WG/N43 and are joined by `imdrf_code`. **`terms_A.tsv`, `terms_E.tsv`, `term_aliases.tsv`** map FDA's filed term strings onto the codes.

**`clean_counts.tsv`, `voluntary_counts.tsv`, `uncoded_by_writer.tsv`.** The counts behind Section 3 and the uncoded-report shares.

## answers/

**`results.jsonl.gz`.** Every answer from the OpenRouter runs, gzipped (read it with `gzip.open` or `pandas.read_json(..., lines=True)`), one JSON object per line: `id` is `model|task|mdr_report_key|arm`; `content` is the model's answer; `provider`, `usage` (prompt, completion and reasoning tokens, cost in dollars), `finish_reason`, `latency_s`, `attempts`. **`medgemma_results.jsonl`** is the same for the self-hosted MedGemma run. **`pilot_results.jsonl`, `pilot2_results.jsonl`** are the two pilots (Section 4.2).

The prompts as sent are not included; `code/scripts/make_jobs.py` regenerates every job file from the data files and `prompts.py`, and the paper's Appendix A shows one prompt in full for each arm.

## scores/

`scores_*.tsv`, one row per answer: model, task, arm, set, writer, report key, token counts, cost, parse flags, number of codes, and the six metrics (`recall`, `branch_recall`, `soft_recall`, `precision`, `soft_precision`, `exact`). `metrics.py` in `code/scripts` defines them. `analysis_frontier.txt` is the paired-contrast printout.

## code/

The pipeline in run order: `download_maude.py`, `extract_reports.py`, `build_codebook.py`, `build_clean.py`, `build_voluntary.py`, `build_context.py` and `build_shortlists.py` (retrieval context; the cluster versions are in `cluster/`), `make_jobs.py`, `run_openrouter.py` and `run_medgemma.py`, `score_runs.py`, `analyze_frontier.py`, `paper_tables.py`. Prompts are in `prompts.py`. The remaining scripts are the checks behind Section 3 of the paper (narrative coverage, duplicates, the intake note, writer groups, the release copy) and the count tables in `data/`. The cluster scripts under `cluster/` carry placeholders for the account and project path.

Run with `python -I`. API keys are read from environment variables and are not part of this release.

## Licence

The code under `code/` is released under the MIT licence (see `LICENSE`). The files we made, the scores, the retrieval context, the verdicts, the count tables and the model answers, are released under the Creative Commons Attribution 4.0 licence (CC BY 4.0). The narratives and codes in the study set are FDA's published MAUDE data, a work of the United States government, and carry no licence of ours.

## Citation

Thomo T, Thomo A. Can language models do the coder's job? Assigning regulatory terms to medical device adverse event narratives. Submitted, 2026. Repository: https://github.com/pyqmax/maude-llm
