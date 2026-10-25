#!/usr/bin/env python3
"""
Confidence-based rejection under date-grouped cross-validation
==============================================================
Improvement of the original extension:
  * Uses date-grouped CV (no measurement date appears in both train and test).
  * Compares Logistic Regression (best model) with Random Forest.
  * The rejection threshold is chosen on a CALIBRATION set that is
    date-disjoint from the training data and never touches the outer test fold.

For every outer fold:
    outer-train  -> split by date into  inner-train (~80%) / calibration (~20%)
    fit model on inner-train
    choose the lowest confidence threshold whose accuracy on accepted
        calibration samples >= target accuracy (and >= MIN_ACCEPT samples)
    apply that threshold to the outer-test fold

Run (needs milk_adulteration_ml.py in the same folder):
    python milk_confidence_grouped.py --csv 13040_2019_200_MOESM2_ESM.csv
"""

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from milk_adulteration_ml import load_data, make_baseline, make_rf

THRESHOLDS = np.round(np.arange(0.30, 0.96, 0.05), 2)
MIN_ACCEPT = 20  # minimum accepted calibration samples when picking a threshold


def choose_threshold(conf, correct, target):
    for t in THRESHOLDS:
        m = conf >= t
        if m.sum() >= MIN_ACCEPT and correct[m].mean() >= target:
            return float(t)
    return float(THRESHOLDS[-1])  # fallback: most conservative


def make_model(name, seed):
    return make_baseline(seed) if name == "lr" else make_rf(seed, 300, None)


def run_one(X, y, groups, model_name, seed, target, n_splits=5):
    outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n = len(y)
    pred = np.empty(n, dtype=object)
    conf = np.zeros(n)
    accepted = np.zeros(n, dtype=bool)
    fold_thr = []

    for tr, te in outer.split(X, y, groups):
        # inner date-disjoint split of the outer-train part
        inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        itr_rel, cal_rel = next(inner.split(X[tr], y[tr], groups[tr]))
        itr, cal = tr[itr_rel], tr[cal_rel]
        assert not set(groups[itr]) & set(groups[cal])
        assert not set(groups[tr]) & set(groups[te])

        model = make_model(model_name, seed).fit(X[itr], y[itr])
        classes = model.classes_

        p_cal = model.predict_proba(X[cal])
        c_cal = p_cal.max(1)
        ok_cal = classes[p_cal.argmax(1)] == y[cal]
        thr = choose_threshold(c_cal, ok_cal, target)
        fold_thr.append(thr)

        p_te = model.predict_proba(X[te])
        pred[te] = classes[p_te.argmax(1)]
        conf[te] = p_te.max(1)
        accepted[te] = conf[te] >= thr

    pred = pred.astype(str)
    correct = pred == y
    res = {
        "overall_acc_no_rejection": correct.mean(),
        "coverage": accepted.mean(),
        "acc_on_accepted": correct[accepted].mean() if accepted.any() else np.nan,
        "acc_on_flagged": correct[~accepted].mean() if (~accepted).any() else np.nan,
        "flagged_fraction": 1 - accepted.mean(),
        "mean_threshold": float(np.mean(fold_thr)),
        "fold_thresholds": ";".join(f"{t:.2f}" for t in fold_thr),
    }
    # fixed-threshold curve on pooled out-of-fold predictions
    curve = pd.DataFrame({
        "threshold": THRESHOLDS,
        "coverage": [(conf >= t).mean() for t in THRESHOLDS],
        "acc_on_accepted": [correct[conf >= t].mean() if (conf >= t).any() else np.nan
                            for t in THRESHOLDS],
    })
    return res, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="results_confidence")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    ap.add_argument("--target-acc", type=float, default=0.95)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _, X, targets, groups, _ = load_data(args.csv)

    rows, curves = [], []
    for task, y in targets.items():
        for model_name in ("lr", "rf"):
            for seed in args.seeds:
                t0 = time.time()
                res, curve = run_one(X, y, groups, model_name, seed, args.target_acc)
                res.update(task=task, model=model_name, seed=seed)
                rows.append(res)
                curve = curve.assign(task=task, model=model_name, seed=seed)
                curves.append(curve)
                print(f"{task:10s} {model_name} seed={seed:<3d} "
                      f"coverage={res['coverage']:.3f} "
                      f"acc_accepted={res['acc_on_accepted']:.3f} "
                      f"acc_overall={res['overall_acc_no_rejection']:.3f} "
                      f"({time.time()-t0:.0f}s)")

    per_seed = pd.DataFrame(rows)
    per_seed.round(4).to_csv(out / "confidence_grouped_per_seed.csv", index=False)

    metrics = ["overall_acc_no_rejection", "coverage", "acc_on_accepted",
               "acc_on_flagged", "flagged_fraction"]
    agg = per_seed.groupby(["task", "model"])[metrics].agg(["mean", "std"]).round(4)
    agg.to_csv(out / "confidence_grouped_summary.csv")
    print("\n=== Summary (mean/std across seeds; target accuracy "
          f"{args.target_acc}) ===")
    print(agg.to_string())

    curves = pd.concat(curves)
    mean_curve = (curves.groupby(["task", "model", "threshold"])[["coverage", "acc_on_accepted"]]
                  .mean().reset_index())
    mean_curve.round(4).to_csv(out / "confidence_grouped_curve.csv", index=False)

    for task in targets:
        fig, ax = plt.subplots(figsize=(6, 4))
        for model_name, label in (("lr", "Logistic Regression"), ("rf", "Random Forest")):
            d = mean_curve[(mean_curve.task == task) & (mean_curve.model == model_name)]
            ax.plot(d["coverage"], d["acc_on_accepted"], marker="o", label=label)
        ax.set_xlabel("Coverage (fraction auto-decided)")
        ax.set_ylabel("Accuracy on auto-decided samples")
        ax.set_title(f"Grouped-CV coverage vs accuracy ({task}, mean of seeds)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"coverage_vs_accuracy_grouped_{task}.png", dpi=200)
        plt.close(fig)
    print(f"\nDone. Outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
