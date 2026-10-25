#!/usr/bin/env python3
"""
Extra experiments to strengthen the milk adulteration study
===========================================================
Exp 1  Leakage test    : random stratified 5-fold CV vs date-grouped 5-fold CV
                         (same models, same data; only the splitting differs).
Exp 2  Preprocessing   : does spectral preprocessing (SNV, 1st derivative)
                         reduce the batch/date effect under grouped CV?
Exp 3  Threshold study : confidence threshold chosen from a LARGE pool of
                         date-grouped out-of-fold confidences (inner CV)
                         instead of one small calibration split.

Models: Logistic Regression (baseline), Linear SVM, PCA+Logistic Regression,
        Random Forest.   (Gradient Boosting is skipped: too slow for CV.)

Run (needs milk_adulteration_ml.py in the same folder):
    python milk_extra_experiments.py --csv 13040_2019_200_MOESM2_ESM.csv
Seeds default to 42 1 2 3 4. Expect roughly 5-15 minutes.
"""

import argparse
import platform
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from milk_adulteration_ml import load_data

warnings.filterwarnings("ignore")
LOG = []


def log(msg=""):
    print(msg)
    LOG.append(str(msg))


# ------------------------------------------------------------------ models
def make_model(name, seed):
    if name == "lr":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, class_weight="balanced",
                                                random_state=seed))
    if name == "svm":
        return make_pipeline(StandardScaler(),
                             LinearSVC(class_weight="balanced", dual="auto",
                                       max_iter=5000, random_state=seed))
    if name == "pca_lr":
        return make_pipeline(StandardScaler(), PCA(n_components=20, random_state=seed),
                             LogisticRegression(max_iter=2000, class_weight="balanced",
                                                random_state=seed))
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                      n_jobs=-1, random_state=seed)
    raise ValueError(name)


MODEL_LABELS = {"lr": "Logistic Regression", "svm": "Linear SVM",
                "pca_lr": "PCA + Logistic Reg.", "rf": "Random Forest"}


# ------------------------------------------------- spectral preprocessing
def snv(X):
    """Standard normal variate: each spectrum is centred and scaled by itself."""
    return (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)


def deriv1(X):
    """Savitzky-Golay first derivative (window 11, polynomial order 2)."""
    return savgol_filter(X, window_length=11, polyorder=2, deriv=1, axis=1)


PREPROCESS = {
    "none": lambda X: X,
    "snv": snv,
    "deriv1": deriv1,
    "snv+deriv1": lambda X: deriv1(snv(X)),
}
# All of these act on each spectrum independently (no statistics are learned
# from other samples), so applying them before splitting cannot leak labels.


# ----------------------------------------------------------------- helpers
def macro_f1(y, p):
    return precision_recall_fscore_support(y, p, average="macro", zero_division=0)[2]


def cv_pooled_predictions(X, y, groups, model_name, seed, splitter):
    pred = np.empty(len(y), dtype=object)
    if splitter == "random":
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        folds = list(cv.split(X, y))
    else:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        folds = list(cv.split(X, y, groups))
        for tr, te in folds:
            assert not set(groups[tr]) & set(groups[te]), "date leakage!"
    shared = []
    for tr, te in folds:
        shared.append(np.isin(groups[te], np.unique(groups[tr])).mean())
        pred[te] = make_model(model_name, seed).fit(X[tr], y[tr]).predict(X[te])
    return pred.astype(str), float(np.mean(shared))


def summarise(df, keys, cols):
    g = df.groupby(keys)[cols].agg(["mean", "std"])
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    return g.round(4).reset_index()


# ------------------------------------------------------------------- Exp 1
def exp1(X, targets, groups, seeds, out):
    log("\n=== Exp 1: random CV vs date-grouped CV (no preprocessing) ===")
    rows = []
    for task, y in targets.items():
        for m in MODEL_LABELS:
            for s in seeds:
                for sp in ("random", "grouped"):
                    p, shared = cv_pooled_predictions(X, y, groups, m, s, sp)
                    rows.append(dict(task=task, model=m, seed=s, split=sp,
                                     accuracy=accuracy_score(y, p), macro_f1=macro_f1(y, p),
                                     test_share_date_with_train=shared))
            log(f"  done {task} / {m}")
    df = pd.DataFrame(rows)
    df.round(4).to_csv(out / "exp1_per_seed.csv", index=False)
    summ = summarise(df, ["task", "model", "split"],
                     ["accuracy", "macro_f1", "test_share_date_with_train"])
    summ.to_csv(out / "exp1_summary.csv", index=False)
    log(summ.to_string(index=False))

    for task in targets:
        d = summ[summ.task == task]
        fig, ax = plt.subplots(figsize=(7, 4))
        w = 0.38
        xs = np.arange(len(MODEL_LABELS))
        for i, sp in enumerate(("random", "grouped")):
            v = d[d.split == sp].set_index("model").loc[list(MODEL_LABELS)]
            ax.bar(xs + (i - 0.5) * w, v["macro_f1_mean"], w, yerr=v["macro_f1_std"],
                   capsize=3, label=f"{sp} CV")
        ax.set_xticks(xs)
        ax.set_xticklabels(MODEL_LABELS.values(), rotation=15)
        ax.set_ylabel("Macro-F1")
        ax.set_title(f"Random vs date-grouped CV ({task})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"exp1_random_vs_grouped_{task}.png", dpi=200)
        plt.close(fig)


# ------------------------------------------------------------------- Exp 2
def exp2(X, targets, groups, seeds, out):
    log("\n=== Exp 2: spectral preprocessing under date-grouped CV ===")
    rows = []
    for pname, fn in PREPROCESS.items():
        Xp = fn(X)
        for task, y in targets.items():
            for m in MODEL_LABELS:
                for s in seeds:
                    p, _ = cv_pooled_predictions(Xp, y, groups, m, s, "grouped")
                    rows.append(dict(task=task, model=m, preprocessing=pname, seed=s,
                                     accuracy=accuracy_score(y, p), macro_f1=macro_f1(y, p)))
        log(f"  done preprocessing = {pname}")
    df = pd.DataFrame(rows)
    df.round(4).to_csv(out / "exp2_per_seed.csv", index=False)
    summ = summarise(df, ["task", "model", "preprocessing"], ["accuracy", "macro_f1"])
    summ.to_csv(out / "exp2_summary.csv", index=False)
    log(summ.to_string(index=False))

    for task in targets:
        d = summ[summ.task == task]
        fig, ax = plt.subplots(figsize=(8, 4))
        pn = list(PREPROCESS)
        w = 0.8 / len(pn)
        xs = np.arange(len(MODEL_LABELS))
        for i, p in enumerate(pn):
            v = d[d.preprocessing == p].set_index("model").loc[list(MODEL_LABELS)]
            ax.bar(xs + (i - (len(pn) - 1) / 2) * w, v["macro_f1_mean"], w,
                   yerr=v["macro_f1_std"], capsize=2, label=p)
        ax.set_xticks(xs)
        ax.set_xticklabels(MODEL_LABELS.values(), rotation=15)
        ax.set_ylabel("Macro-F1 (grouped CV)")
        ax.set_title(f"Effect of preprocessing ({task})")
        ax.legend(title="Preprocessing", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"exp2_preprocessing_{task}.png", dpi=200)
        plt.close(fig)


# ------------------------------------------------------------------- Exp 3
def exp3(X, targets, groups, seeds, out, target_acc=0.95, flag_fraction=0.20):
    log("\n=== Exp 3: threshold from pooled inner-CV confidences (Logistic Regression) ===")
    thr_grid = np.round(np.arange(0.30, 0.9951, 0.01), 2)
    rows = []
    for task, y in targets.items():
        for s in seeds:
            outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=s)
            conf_te = np.zeros(len(y)); corr_te = np.zeros(len(y), bool)
            acc_mask_t = np.zeros(len(y), bool); acc_mask_q = np.zeros(len(y), bool)
            thr_t_list, thr_q_list, reached = [], [], []
            for tr, te in outer.split(X, y, groups):
                # inner date-grouped CV on the outer-train part -> pooled OOF confidences
                inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=s)
                oof_conf = np.zeros(len(tr)); oof_ok = np.zeros(len(tr), bool)
                for itr, iva in inner.split(X[tr], y[tr], groups[tr]):
                    mdl = make_model("lr", s).fit(X[tr][itr], y[tr][itr])
                    pr = mdl.predict_proba(X[tr][iva])
                    oof_conf[iva] = pr.max(1)
                    oof_ok[iva] = mdl.classes_[pr.argmax(1)] == y[tr][iva]
                # (a) target-accuracy threshold on the pooled pool
                thr_t, hit = float(thr_grid[-1]), False
                for t in thr_grid:
                    m = oof_conf >= t
                    if m.sum() >= 50 and oof_ok[m].mean() >= target_acc:
                        thr_t, hit = float(t), True
                        break
                # (b) fixed flag fraction: flag the least-confident 20%
                thr_q = float(np.quantile(oof_conf, flag_fraction))
                final = make_model("lr", s).fit(X[tr], y[tr])
                pr = final.predict_proba(X[te])
                conf_te[te] = pr.max(1)
                corr_te[te] = final.classes_[pr.argmax(1)] == y[te]
                acc_mask_t[te] = pr.max(1) >= thr_t
                acc_mask_q[te] = pr.max(1) >= thr_q
                thr_t_list.append(thr_t); thr_q_list.append(thr_q); reached.append(hit)
            for name, mask, thrs in (("target_acc_0.95", acc_mask_t, thr_t_list),
                                     ("flag_lowest_20pct", acc_mask_q, thr_q_list)):
                rows.append(dict(task=task, seed=s, strategy=name,
                                 overall_acc=corr_te.mean(), coverage=mask.mean(),
                                 acc_on_accepted=corr_te[mask].mean() if mask.any() else np.nan,
                                 acc_on_flagged=corr_te[~mask].mean() if (~mask).any() else np.nan,
                                 folds_where_target_reached=int(sum(reached)) if name.startswith("target") else np.nan,
                                 fold_thresholds=";".join(f"{t:.2f}" for t in thrs)))
        log(f"  done {task}")
    df = pd.DataFrame(rows)
    df.round(4).to_csv(out / "exp3_per_seed.csv", index=False)
    summ = summarise(df, ["task", "strategy"],
                     ["overall_acc", "coverage", "acc_on_accepted", "acc_on_flagged",
                      "folds_where_target_reached"])
    summ.to_csv(out / "exp3_summary.csv", index=False)
    log(summ.to_string(index=False))


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="results_extra")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    ap.add_argument("--only", nargs="+", choices=["1", "2", "3"], default=["1", "2", "3"],
                    help="run only some experiments, e.g. --only 3")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log(f"Run started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Python {sys.version.split()[0]} | sklearn {sklearn.__version__} | "
        f"numpy {np.__version__} | pandas {pd.__version__} | {platform.platform()}")
    log(f"Seeds = {args.seeds}")

    _, X, targets, groups, _ = load_data(args.csv)
    t0 = time.time()
    if "1" in args.only:
        exp1(X, targets, groups, args.seeds, out)
    if "2" in args.only:
        exp2(X, targets, groups, args.seeds, out)
    if "3" in args.only:
        exp3(X, targets, groups, args.seeds, out)
    log(f"\nTotal time: {time.time() - t0:.0f}s")
    (out / "run_log_extra.txt").write_text("\n".join(LOG), encoding="utf-8")
    print(f"\nDone. Outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
