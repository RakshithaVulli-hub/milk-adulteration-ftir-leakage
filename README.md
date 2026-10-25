# Leakage-Aware Detection of Milk Adulteration from FTIR Spectra

Reproducibility materials for **“Leakage-Aware Detection of Milk Adulteration from Infrared Spectra with Confidence-Based Flagging.”**

This project investigates milk adulteration classification using Fourier-transform infrared (FTIR) spectra. It compares random and measurement-date-grouped cross-validation, evaluates spectral preprocessing, and explores confidence-based flagging to identify predictions that should be referred for confirmatory laboratory testing.

![System architecture](figures/system_architecture.png)

## Repository Structure

```text
milk-adulteration-research/
├── README.md
├── .gitignore
├── requirements.txt
├── DATA_SOURCE.txt
├── 13040_2019_200_MOESM2_ESM.csv
├── code/
│   ├── milk_adulteration_ml.py
│   ├── milk_confidence_grouped.py
│   └── milk_extra_experiments.py
├── figures/
│   └── figures used in the paper
└── results/
    ├── protocols_seed42/
    ├── protocols_seed1/
    ├── protocols_seed2/
    ├── confidence_grouped/
    └── extra_experiments/
```

The `results/` directories contain experiment outputs, summaries, configuration files, logs, and diagnostic figures associated with the corresponding runs.

## Dataset

The study uses the supplementary dataset `13040_2019_200_MOESM2_ESM.csv`, associated with:

H. A. Neto et al., “On the utilization of deep and ensemble learning to detect milk adulteration,” *BioData Mining* (2019). [https://doi.org/10.1186/s13040-019-0200-5](https://doi.org/10.1186/s13040-019-0200-5)

The subset used in this study contains **969 spectra, 518 spectral features, six classes, and 259 distinct measurement dates**. Sample identifiers, date/time metadata, and laboratory composition fields are excluded from the model feature matrix. Measurement date is used to group samples during evaluation, not as a predictive feature.

See [`DATA_SOURCE.txt`](DATA_SOURCE.txt) for dataset provenance, attribution, and licensing information.

## Environment and Installation

The recorded experimental environment was Python 3.14.2, scikit-learn 1.9.1, NumPy 2.5.4, and pandas 3.1.0 on Windows 11, using an Intel Core i5-1340P and 16 GB RAM. No GPU was used.

Create and activate a virtual environment from the repository root.

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**macOS/Linux**

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproducing the Experiments

Run the commands from the repository root.

| Experiment | Command | Related manuscript content |
|---|---|---|
| Main protocols, seed 42 | `python code/milk_adulteration_ml.py --csv 13040_2019_200_MOESM2_ESM.csv --seed 42 --out results/protocols_seed42` | Main protocol results and seed-42 diagnostics |
| Main protocols, seed 1 | `python code/milk_adulteration_ml.py --csv 13040_2019_200_MOESM2_ESM.csv --seed 1 --out results/protocols_seed1` | Main protocol results |
| Main protocols, seed 2 | `python code/milk_adulteration_ml.py --csv 13040_2019_200_MOESM2_ESM.csv --seed 2 --out results/protocols_seed2` | Main protocol results |
| Earlier grouped-CV confidence experiment | `python code/milk_confidence_grouped.py --csv 13040_2019_200_MOESM2_ESM.csv --out results/confidence_grouped` | Earlier confidence experiment |
| Additional experiments | `python code/milk_extra_experiments.py --csv 13040_2019_200_MOESM2_ESM.csv --seeds 42 1 2 --out results/extra_experiments` | Leakage comparison, preprocessing, and confidence-based flagging |

The experiment outputs include per-seed and summary CSV files, including `exp1_per_seed.csv`, `exp1_summary.csv`, `exp2_per_seed.csv`, `exp2_summary.csv`, `exp3_per_seed.csv`, and `exp3_summary.csv`, as well as `run_log_extra.txt`.

## Main Findings

- Random five-fold cross-validation had **86.3%** test-sample overlap by measurement date with the training data, compared with **0%** for date-grouped cross-validation.
- Under date-grouped evaluation, Random Forest multiclass macro-F1 increased from approximately **0.631** without preprocessing to **0.799** with first-derivative preprocessing.
- Linear SVM with SNV preprocessing achieved a reported multiclass macro-F1 of **0.933**.
- In the confidence-based flagging experiment, the multiclass target-95% strategy accepted approximately **80.4%** of samples at **0.953** accuracy on accepted predictions. The binary target-95% strategy accepted approximately **87.2%** at **0.947** accuracy on accepted predictions.

These results describe the selected dataset and evaluation setup; they should not be interpreted as guaranteed performance on other datasets or in real-world settings.

## Interpretation and Limitations

Measurement date and batch are confounded in the available dataset, so their individual effects cannot be isolated. The study uses a 969-sample subset from one source and instrument; adulterated samples were prepared in a laboratory, and class sizes are imbalanced. Model and preprocessing comparisons are exploratory and use three random seeds.

Confidence-based flagging was evaluated using Logistic Regression on unprocessed spectra. The workflow is intended as a **screening aid**, not a replacement for confirmatory chemical analysis. Uncertain samples should be referred for laboratory testing, and model output should not be the sole basis for regulatory or legal decisions.

## AI Assistance

The project documentation and manuscript disclose the use of Claude (Anthropic) for assistance with code writing and debugging, experiment suggestions, and drafting. The authors are responsible for reviewing the final implementation, results, citations, and manuscript.

## Citation

When reusing the dataset, cite:

Neto, H. A., et al. (2019). “On the utilization of deep and ensemble learning to detect milk adulteration.” *BioData Mining*. [https://doi.org/10.1186/s13040-019-0200-5](https://doi.org/10.1186/s13040-019-0200-5).

Please also cite the research paper and any other sources listed in its references when reusing the methods or findings.
