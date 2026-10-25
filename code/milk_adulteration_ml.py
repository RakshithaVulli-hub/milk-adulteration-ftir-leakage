#!/usr/bin/env python3
"""
Milk adulteration detection from spectral data
==============================================
Dataset : 13040_2019_200_MOESM2_ESM.csv (supplementary data of
          "On the utilization of deep and ensemble learning to detect
          milk adulteration", BioData Mining, 2019)

What this script does
---------------------
1. Loads the CSV and uses ONLY the spectral columns ("SPC ...") as features.
   Index/time/date columns and the lab composition columns are NOT used.
2. Protocol A - random stratified 70:15:15 split (train/val/test).
      Baseline : Logistic Regression
      Proposed : Random Forest (tuned on the validation set)
      Extra    : Gradient Boosting (sklearn HistGradientBoosting)
3. Protocol B - leakage check: 5-fold StratifiedGroupKFold where all samples
   from the same measurement DATE stay in the same fold.
4. Own extension: confidence-based rejection. Predictions below a probability
   threshold are flagged "needs lab test"; the threshold is chosen on the
   validation set and evaluated on the test set (coverage vs accuracy).
5. Writes tables, figures and a run log to the results/ folder.

Run:  python milk_adulteration_ml.py --csv 13040_2019_200_MOESM2_ESM.csv
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LOG_LINES = []


def log(msg=""):
    print(msg)
    LOG_LINES.append(str(msg))


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    spc_cols = [c for c in df.columns if c.startswith("SPC")]
    X = df[spc_cols].to_numpy(dtype=float)
    groups = df["Date"].astype(str).to_numpy()
    targets = {
        "multiclass": df["Ingredient"].astype(str).to_numpy(),
        "binary": np.where(df["IsRawMilk"].astype(str).str.lower() == "yes",
                           "raw", "adulterated"),
    }
    wavenumbers = np.array([float(c.split()[1]) for c in spc_cols])
    return df, X, targets, groups, wavenumbers


def sanity_checks(df, X):
    log(f"Samples: {X.shape[0]}, spectral features: {X.shape[1]}")
    log(f"Missing values in spectra: {int(np.isnan(X).sum())}")
    dup = int(df[[c for c in df.columns if c.startswith('SPC')]].duplicated().sum())
    log(f"Exact duplicate spectra: {dup}")
    log("Class counts:\n" + df["Ingredient"].value_counts().to_string())
    top = df["Date"].value_counts().head(3)
    log("Largest measurement dates (possible batch effect):\n" + top.to_string())


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def make_baseline(seed):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced",
                                            random_state=seed))


def make_rf(seed, n_estimators=300, max_depth=None):
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                  class_weight="balanced", n_jobs=-1,
                                  random_state=seed)


def make_gb(seed):
    return HistGradientBoostingClassifier(class_weight="balanced",
                                          random_state=seed)


def tune_rf(Xtr, ytr, Xva, yva, seed):
    """Tiny grid search; selection is done on the VALIDATION set only."""
    best, best_f1, best_params = None, -1, None
    for n in (200, 500):
        for depth in (None, 20):
            m = make_rf(seed, n, depth).fit(Xtr, ytr)
            f1 = precision_recall_fscore_support(
                yva, m.predict(Xva), average="macro", zero_division=0)[2]
            log(f"  RF n_estimators={n} max_depth={depth} -> val macro-F1={f1:.4f}")
            if f1 > best_f1:
                best, best_f1, best_params = m, f1, (n, depth)
    log(f"  Selected RF params (n_estimators, max_depth) = {best_params}")
    return best, best_params


# --------------------------------------------------------------------------
# Metrics / plots
# --------------------------------------------------------------------------
def metrics_row(name, y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    return {"Model": name,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision (macro)": p, "Recall (macro)": r, "F1 (macro)": f}


def save_confusion(y_true, y_pred, labels, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_mean_spectra(X, y, wn, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for cls in sorted(set(y)):
        ax.plot(wn, X[y == cls].mean(axis=0), label=cls, lw=1.2)
    ax.set_xlabel("Wavenumber index (SPC column)")
    ax.set_ylabel("Mean absorbance")
    ax.set_title("Mean spectrum per class")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# Protocol A: random stratified 70:15:15
# --------------------------------------------------------------------------
def protocol_a(X, y, groups, task, seed, out, target_acc):
    log(f"\n=== Protocol A (random stratified 70:15:15) | task = {task} ===")
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=seed)
    log(f"Split sizes: train={len(tr)}, val={len(va)}, test={len(te)}")

    shared = np.isin(groups[te], np.unique(groups[tr])).mean()
    log(f"LEAKAGE INDICATOR: {shared:.1%} of test samples share a measurement "
        f"date with at least one training sample.")

    Xtr, Xva, Xte = X[tr], X[va], X[te]
    ytr, yva, yte = y[tr], y[va], y[te]
    labels = sorted(set(y))

    models, rows = {}, []
    t0 = time.time()
    models["Baseline: Logistic Regression"] = make_baseline(seed).fit(Xtr, ytr)
    log(f"Baseline trained in {time.time()-t0:.1f}s")

    t0 = time.time()
    log("Tuning Random Forest on validation set:")
    rf, _ = tune_rf(Xtr, ytr, Xva, yva, seed)
    models["Proposed: Random Forest"] = rf
    log(f"RF tuned + trained in {time.time()-t0:.1f}s")

    t0 = time.time()
    models["Gradient Boosting"] = make_gb(seed).fit(Xtr, ytr)
    log(f"Gradient Boosting trained in {time.time()-t0:.1f}s")

    for name, m in models.items():
        pred = m.predict(Xte)
        rows.append(metrics_row(name, yte, pred))
        log(f"\n--- {name} (TEST) ---\n" +
            classification_report(yte, pred, labels=labels, zero_division=0))
        safe = name.split(":")[-1].strip().lower().replace(" ", "_")
        save_confusion(yte, pred, labels, f"{name} ({task})",
                       out / f"cm_A_{task}_{safe}.png")

    pd.DataFrame(rows).round(4).to_csv(out / f"metrics_A_{task}.csv", index=False)

    # ---- Own extension: confidence-based rejection with the RF ------------
    rf_model = models["Proposed: Random Forest"]
    classes = rf_model.classes_
    p_va, p_te = rf_model.predict_proba(Xva), rf_model.predict_proba(Xte)
    conf_va, conf_te = p_va.max(1), p_te.max(1)
    pred_va, pred_te = classes[p_va.argmax(1)], classes[p_te.argmax(1)]

    thresholds = np.round(np.arange(0.30, 0.96, 0.05), 2)
    curve = []
    for t in thresholds:
        acc_m = conf_te >= t
        va_m = conf_va >= t
        curve.append({
            "threshold": t,
            "val_coverage": va_m.mean(),
            "val_acc_on_accepted": accuracy_score(yva[va_m], pred_va[va_m]) if va_m.any() else np.nan,
            "test_coverage": acc_m.mean(),
            "test_acc_on_accepted": accuracy_score(yte[acc_m], pred_te[acc_m]) if acc_m.any() else np.nan,
        })
    curve = pd.DataFrame(curve)
    curve.round(4).to_csv(out / f"confidence_curve_{task}.csv", index=False)

    ok = curve[curve["val_acc_on_accepted"] >= target_acc]
    chosen = (ok.iloc[0] if len(ok) else curve.iloc[-1])  # lowest threshold meeting target on VAL
    log(f"\nConfidence extension: threshold chosen on validation set "
        f"(val accuracy >= {target_acc}) = {chosen['threshold']}")
    log(f"  Test coverage (auto-decided) = {chosen['test_coverage']:.1%}; "
        f"test accuracy on those = {chosen['test_acc_on_accepted']:.4f}; "
        f"flagged for lab test = {1-chosen['test_coverage']:.1%}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curve["test_coverage"], curve["test_acc_on_accepted"], marker="o")
    ax.set_xlabel("Coverage (fraction auto-decided)")
    ax.set_ylabel("Accuracy on auto-decided samples")
    ax.set_title(f"Coverage vs accuracy ({task}, test set)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"coverage_vs_accuracy_{task}.png", dpi=200)
    plt.close(fig)
    return rows


# --------------------------------------------------------------------------
# Protocol B: grouped CV (same-date samples never split across folds)
# --------------------------------------------------------------------------
def protocol_b(X, y, groups, task, seed, out, n_splits=5):
    log(f"\n=== Protocol B (date-grouped {n_splits}-fold CV, pooled predictions) | task = {task} ===")
    labels = sorted(set(y))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(sgkf.split(X, y, groups))
    for i, (tr, te) in enumerate(folds):
        assert not set(groups[tr]) & set(groups[te]), "date leakage between folds!"
        log(f"  fold {i}: train={len(tr)}, test={len(te)}, "
            f"classes in test={sorted(set(y[te]))}")

    factories = {
        "Baseline: Logistic Regression": lambda: make_baseline(seed),
        "Proposed: Random Forest": lambda: make_rf(seed, 300, None),
        "Gradient Boosting": lambda: make_gb(seed),
    }
    rows = []
    for name, fac in factories.items():
        pooled = np.empty(len(y), dtype=object)
        for tr, te in folds:
            pooled[te] = fac().fit(X[tr], y[tr]).predict(X[te])
        pooled = pooled.astype(str)
        rows.append(metrics_row(name, y, pooled))
        safe = name.split(":")[-1].strip().lower().replace(" ", "_")
        save_confusion(y, pooled, labels, f"{name} - grouped CV ({task})",
                       out / f"cm_B_{task}_{safe}.png")
        log(f"  {name}: acc={rows[-1]['Accuracy']:.4f}, macro-F1={rows[-1]['F1 (macro)']:.4f}")
    pd.DataFrame(rows).round(4).to_csv(out / f"metrics_B_{task}.csv", index=False)
    return rows


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to 13040_2019_200_MOESM2_ESM.csv")
    ap.add_argument("--out", default="results", help="output folder")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-acc", type=float, default=0.95,
                    help="validation accuracy target for the confidence threshold")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    log(f"Run started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Python {sys.version.split()[0]} | sklearn {sklearn.__version__} | "
        f"numpy {np.__version__} | pandas {pd.__version__} | {platform.platform()}")
    log(f"Seed = {args.seed}")

    df, X, targets, groups, wn = load_data(args.csv)
    sanity_checks(df, X)
    plot_mean_spectra(X, targets["multiclass"], wn, out / "mean_spectra.png")

    for task, y in targets.items():
        protocol_a(X, y, groups, task, args.seed, out, args.target_acc)
        protocol_b(X, y, groups, task, args.seed, out)

    (out / "run_log.txt").write_text("\n".join(LOG_LINES), encoding="utf-8")
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    print(f"\nDone. All outputs are in: {out.resolve()}")


if __name__ == "__main__":
    main()
