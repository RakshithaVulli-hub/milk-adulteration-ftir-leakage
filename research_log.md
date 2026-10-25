# Research log: milk adulteration detection

This file records the experiment history and repository verification status. Personal recollections and declarations must be completed or confirmed by the author; this log does not substitute for checking the source papers or rerunning experiments.

## Dataset and study design

- Dataset: `13040_2019_200_MOESM2_ESM.csv`, supplementary data associated with Neto et al. (2019), *BioData Mining*, DOI 10.1186/s13040-019-0200-5.
- The file used for this study contains 969 samples, 518 spectral columns, six classes, and 259 distinct measurement dates.
- The study excludes row/sample identifiers, date/time, and laboratory composition fields from model features. Date is used as a grouping key for evaluation, not as a model input.
- The research question is whether reported performance changes when measurement dates are kept together in cross-validation, and whether preprocessing and confidence-based flagging are useful in this subset.

## Main protocol experiments

Three seeds were used: 42, 1, and 2. The repository contains `metrics_*.csv`, run logs, configuration files, and diagnostic figures in `results/protocols_seed42/`, `results/protocols_seed1/`, and `results/protocols_seed2/`.

- Protocol A is a stratified random 70:15:15 split.
- Protocol B uses five-fold cross-validation grouped by measurement date.
- The paper reports mean and standard deviation across seeds for Table X.
- Random Forest is labelled “Proposed” in some legacy CSV outputs. This label should not be interpreted as the final research claim; the paper correctly describes the contribution as an evaluation and triage workflow.

## Extra experiments

### Exp 1: random versus date-grouped CV

The available `exp1_per_seed.csv` and `exp1_summary.csv` contain results for the random/grouped comparison. The paper reports 86.3% date overlap for random five-fold CV and 0% for grouped CV. The reported macro-F1 reductions include 0.037 for multiclass Logistic Regression, 0.031 for Linear SVM, 0.055 for PCA + Logistic Regression, and 0.150 for Random Forest.

### Exp 2: preprocessing

The available `exp2_per_seed.csv` supports the preprocessing comparison. The paper reports Random Forest multiclass macro-F1 increasing from about 0.631 without preprocessing to 0.799 with the first derivative, and a best multiclass result of 0.933 for Linear SVM with SNV. These comparisons are exploratory because models and preprocessing variants were compared on the same grouped folds.

### Exp 3: confidence-based flagging

**Verification status: source files restored and values checked at summary level.** The uploaded `results_extra.zip` supplied `exp3_per_seed.csv`, `exp3_summary.csv`, `exp2_summary.csv`, and `run_log_extra.txt`; these are now included under `results/extra_experiments/`. The Exp 3 summary supports Table XII when rounded to the paper's precision: multiclass target-95% coverage 0.8039 (80.4 ± 3.5% as reported), accepted accuracy 0.953, flagged accuracy 0.6759 (0.676); binary target-95% coverage 0.8724 (87.2 ± 2.9%), accepted accuracy 0.9468 (0.947), flagged accuracy 0.8617 (0.862). The fixed-lowest-20% summary likewise rounds to the displayed Table XII values. These are file-to-paper checks, not a fresh rerun of the Python pipeline.

## Manuscript and repository checks

- The paper contains five related-work tables, a research-gap table, a system-architecture figure, methodology, results, limitations, an AI-assistance declaration, acknowledgment, and references.
- The paper currently has two `[GitHub repository URL]` placeholders: one in Table VII and one in Section V. Replace both only after the final public repository URL exists.
- The submitted DOCX must be reviewed after inserting the URL. No PDF was supplied with the current file upload, so matching DOCX/PDF layout has not been checked in this pass.
- The paper's Table IX and Table XI values were documented in the prior working notes as recomputed from the available Exp 1 and Exp 2 files. Table XII remains pending until the original Exp 3 outputs are restored.
- Verify all references and DOI details directly against the source records before making a final claim that they have all been checked.

## Author-confirmation items

- Add the actual date and reason the topic was selected; do not invent a personal recollection.
- Confirm that the acknowledgment and author line are correct and consistent with institutional guidance.
- Confirm the AI-assistance declaration is accurate.
- Rewrite any AI-assisted passages in the author's own words where required by the faculty.
- Inspect Fig. 1 and the scripts to confirm that the diagram reflects the implemented pipeline.
- The Exp 3 outputs and run log have now been restored from `results_extra.zip`; confirm the scripts regenerate matching files in the intended environment.
- Create the public repository, insert the final URL in both manuscript placeholders, and then perform a final repository-to-paper audit.

## Current status

The repository ZIP and the newly supplied results_extra.zip passed ZIP integrity checks. Exp 1–3 summary/per-seed files and the extra run log are now present, and Table XII matches the summary values at the paper's displayed precision. The package is better documented and file-level checked, but it should **not yet be described as fully verified for submission** until end-to-end reruns, manuscript URL replacement, and the final paper/PDF review are complete.
