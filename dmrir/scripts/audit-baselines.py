#!/usr/bin/env python
"""
Audit of the DMR-IR feature-based baselines (patient-level, fixed 5 folds).

Run from the repository root:
    python dmrir/scripts/audit_baselines.py
or point at it explicitly:
    python audit_baselines.py --root "C:\\path\\to\\med-tech-project"

What it does (it never modifies your existing files):
  1. Column audit            - which numeric columns could leak labels/folds
  2. Label / fold audit      - label<->name mapping, features vs folds file, class-folder check
  3. Univariate AUC          - DESCRIPTIVE only (never use it to select features)
  4. OOF prediction audit    - one row per patient, fold consistency, metrics recomputed
  5. Feature-group ablation  - same folds, logistic regression, geometry vs intensity vs dynamics
  6. Label-permutation test  - null distribution of pooled out-of-fold AUC
  7. Patient-level bootstrap - uncertainty on pooled OOF AUC / sensitivity / specificity
  8. Repeated CV             - sensitivity of the result to the particular fold partition

Only numpy, pandas and scikit-learn are used (no scipy imports of our own).
Section 5-8 re-implement Baseline 1 (median imputer + StandardScaler + LogisticRegression,
class_weight="balanced"). Check that C / max_iter match train-baseline.py.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASES = [
    "roi_percentage", "mean", "median", "std", "min", "max", "p10", "p25", "p75",
    "p90", "iqr", "intensity_range", "coefficient_variation", "bbox_width", "bbox_height",
]
SUFFIXES = ["mean", "std", "min", "max", "slope"]
GEOMETRY_BASES = ["roi_percentage", "bbox_width", "bbox_height"]
INTENSITY_BASES = [b for b in BASES if b not in GEOMETRY_BASES]
EXPECTED = [f"{b}_{s}" for b in BASES for s in SUFFIXES]

REPORT = []


def log(msg=""):
    print(msg)
    REPORT.append(str(msg))


def header(title):
    log()
    log("=" * 78)
    log(title)
    log("=" * 78)


# ----------------------------------------------------------------------------- metrics
def safe_auc(y, score):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, score)


def binary_metrics(y, pred, score=None):
    y, pred = np.asarray(y), np.asarray(pred)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    div = lambda a, b: a / b if b else np.nan
    return {
        "accuracy": div(tp + tn, len(y)),
        "sensitivity": div(tp, tp + fn),
        "specificity": div(tn, tn + fp),
        "PPV": div(tp, tp + fp),
        "NPV": div(tn, tn + fn),
        "AUC": safe_auc(y, score) if score is not None else np.nan,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


def make_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=5000)),
    ])


def run_cv(X, y, folds):
    """Fixed-fold CV. Returns OOF score/pred arrays (aligned with rows of X)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    folds = np.asarray(folds)
    score = np.full(len(y), np.nan)
    pred = np.full(len(y), -1)
    for f in np.unique(folds):
        test = folds == f
        model = make_pipeline().fit(X[~test], y[~test])
        score[test] = model.predict_proba(X[test])[:, 1]
        pred[test] = model.predict(X[test])
    return score, pred


def summarize_cv(y, folds, score, pred):
    folds = np.asarray(folds)
    per_fold = [binary_metrics(y[folds == f], pred[folds == f], score[folds == f])
                for f in np.unique(folds)]
    pooled = binary_metrics(y, pred, score)
    fold_auc = np.array([m["AUC"] for m in per_fold])
    return pooled, np.nanmean(fold_auc)


def make_folds(y, k, seed):
    rng = np.random.RandomState(seed)
    folds = np.zeros(len(y), dtype=int)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            folds[i] = j % k
    return folds


# ----------------------------------------------------------------------------- sections
def section_columns(feat):
    header("1. COLUMN AUDIT (features file)")
    log(f"Rows: {len(feat)}   Columns: {feat.shape[1]}")
    present = [c for c in EXPECTED if c in feat.columns]
    missing = [c for c in EXPECTED if c not in feat.columns]
    log(f"Expected feature columns present: {len(present)} / {len(EXPECTED)}")
    if missing:
        log(f"  MISSING expected columns: {missing}")
    numeric = feat.select_dtypes(include=[np.number]).columns.tolist()
    extra = [c for c in numeric if c not in EXPECTED]
    log(f"Numeric columns that are NOT thermal features: {extra}")
    log("  -> If train-baseline.py picks features as 'all numeric columns minus a few',")
    log("     confirm every column in that list is excluded. Anything like label, fold,")
    log("     label_fold, fold_fold would be direct leakage.")
    nonnum = [c for c in feat.columns if c not in numeric]
    log(f"Non-numeric columns: {nonnum}")
    nan_cols = feat[present].columns[feat[present].isna().any()].tolist()
    log(f"Feature columns containing NaN: {nan_cols if nan_cols else 'none'}")
    const = [c for c in present if feat[c].nunique(dropna=True) <= 1]
    log(f"Constant feature columns: {const if const else 'none'}")
    return present


def section_labels(feat, folds_df, manifest):
    header("2. LABEL / FOLD AUDIT")
    log("Label vs label_name in folds file:")
    log(pd.crosstab(folds_df["label"], folds_df["label_name"]).to_string())
    merged = feat[["patient_id", "label", "fold"]].merge(
        folds_df[["patient_id", "label", "fold"]], on="patient_id", suffixes=("_feat", "_folds"))
    log(f"Patients in features: {feat['patient_id'].nunique()}   in folds file: "
        f"{folds_df['patient_id'].nunique()}   in both: {len(merged)}")
    log(f"Label disagreements features vs folds file: {(merged.label_feat != merged.label_folds).sum()}")
    log(f"Fold  disagreements features vs folds file: {(merged.fold_feat != merged.fold_folds).sum()}")
    log("Class counts per fold (rows=fold, cols=label):")
    log(pd.crosstab(folds_df["fold"], folds_df["label"]).to_string())
    if "sequence_suffix" in feat.columns:
        log("sequence_suffix (dir/esq) vs label (not used as feature, but check it is not a proxy):")
        log(pd.crosstab(feat["sequence_suffix"], feat["label"]).to_string())

    if manifest is not None:
        log()
        log("Manifest check:")
        if "patient_id" in manifest.columns:
            per_patient = manifest.groupby("patient_id").size()
            log(f"  images per patient: min={per_patient.min()} max={per_patient.max()} "
                f"(expected 20 for all)")
        path_cols = [c for c in manifest.columns if "path" in c.lower()]
        label_col = "label" if "label" in manifest.columns else None
        if path_cols and label_col:
            parts = manifest[path_cols[0]].astype(str).str.replace("\\", "/", regex=False).str.split("/")
            labels = manifest[label_col].values
            log("  Path components that appear in >=90% of one class's paths and in none of")
            log("  the other class's paths (these should be the original class folder names):")
            for lab in sorted(np.unique(labels)):
                in_lab = [set(p) for p, l in zip(parts, labels) if l == lab]
                out_lab = [set(p) for p, l in zip(parts, labels) if l != lab]
                counts = {}
                for s in in_lab:
                    for comp in s:
                        counts[comp] = counts.get(comp, 0) + 1
                other = set().union(*out_lab) if out_lab else set()
                cands = [c for c, n in counts.items() if n >= 0.9 * len(in_lab) and c not in other]
                log(f"    label={lab}: {cands}")
            log("  -> Verify by eye that label 1 really corresponds to the dataset's diseased")
            log("     class folder and label 0 to the healthy one (do not infer from counts).")
        else:
            log(f"  could not find path/label columns in manifest (columns: {list(manifest.columns)})")


def section_univariate(feat, cols, y):
    header("3. UNIVARIATE AUC PER FEATURE (descriptive only - never select features with this)")
    rows = []
    for c in cols:
        x = feat[c].astype(float).fillna(feat[c].median())
        a = roc_auc_score(y, x)
        rows.append((c, max(a, 1 - a), "geometry" if any(c.startswith(g + "_") for g in GEOMETRY_BASES) else "intensity"))
    df = pd.DataFrame(rows, columns=["feature", "abs_AUC", "group"]).sort_values("abs_AUC", ascending=False)
    log(df.head(15).to_string(index=False))
    log()
    log("Mean |AUC| by group: " + ", ".join(f"{g}={df[df.group == g].abs_AUC.mean():.3f}" for g in df.group.unique()))
    log("Mean |AUC| by temporal summary: " + ", ".join(
        f"{s}={df[df.feature.str.endswith('_' + s)].abs_AUC.mean():.3f}" for s in SUFFIXES))
    return df


def find_col(df, options):
    for o in options:
        if o in df.columns:
            return o
    return None


def load_oof(path, name, y_map, fold_map):
    header(f"4. OOF PREDICTION AUDIT - {name}")
    if not path.exists():
        log(f"{path} not found - skipped")
        return None
    oof = pd.read_csv(path)
    log(f"Columns: {list(oof.columns)}   Rows: {len(oof)}")
    pid = find_col(oof, ["patient_id"])
    true = find_col(oof, ["label", "y_true", "true_label", "true"])
    score = find_col(oof, ["probability", "y_score", "score", "prob_sick", "proba", "y_prob",
                           "probability_sick", "pred_proba", "predicted_probability"])
    pred = find_col(oof, ["prediction", "y_pred", "pred", "predicted_label", "predicted"])
    fold = find_col(oof, ["fold"])
    log(f"Detected -> patient:{pid} true:{true} score:{score} pred:{pred} fold:{fold}")
    if pid is None or true is None or score is None:
        log("Could not detect required columns; edit find_col options at the top of load_oof().")
        return None
    log(f"Rows per patient: min={oof.groupby(pid).size().min()} max={oof.groupby(pid).size().max()} (expect 1)")
    log(f"Patients missing from OOF: {sorted(set(y_map) - set(oof[pid]))}")
    log(f"Label mismatches vs folds file: {(oof[pid].map(y_map) != oof[true]).sum()}")
    if fold:
        log(f"Fold mismatches vs folds file: {(oof[pid].map(fold_map) != oof[fold]).sum()}")
    oof = oof.drop_duplicates(pid)
    y = oof[true].values
    s = oof[score].values
    p = oof[pred].values if pred else (s >= 0.5).astype(int)
    if pred and ((s >= 0.5).astype(int) != p).any():
        log(f"Note: stored prediction differs from score>=0.5 for {int(((s >= 0.5).astype(int) != p).sum())} patients "
            f"(fine for SVM predict() vs Platt-scaled probability; AUC uses the score column)")
    pooled = binary_metrics(y, p, s)
    log("Pooled OOF metrics (one confusion matrix over all patients - NOT the fold mean):")
    log("  " + ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in pooled.items()))
    if fold:
        rows = []
        for f in sorted(oof[fold].unique()):
            m = oof[fold] == f
            rows.append({"fold": f, **binary_metrics(y[m], p[m], s[m])})
        log("Per fold (recomputed from OOF):")
        log(pd.DataFrame(rows).round(3).to_string(index=False))
    wrong = oof[(p != y)]
    log(f"Misclassified patients at stored threshold: {wrong[pid].tolist()}")
    oof = oof.assign(_y=y, _s=s, _p=p)
    return oof.rename(columns={pid: "patient_id"})


def compare_oof(lr, svm):
    if lr is None or svm is None:
        return
    header("4b. LOGISTIC REGRESSION vs SVM OOF AGREEMENT")
    m = lr[["patient_id", "_y", "_s", "_p"]].merge(
        svm[["patient_id", "_s", "_p"]], on="patient_id", suffixes=("_lr", "_svm"))
    log(f"Patients compared: {len(m)}")
    log(f"Predicted-label disagreements: {(m._p_lr != m._p_svm).sum()}")
    log(f"AUC  LR={safe_auc(m._y, m._s_lr):.4f}   SVM={safe_auc(m._y, m._s_svm):.4f}")
    log("Identical mean accuracy/sensitivity/specificity for both models is plausible only if")
    log("their predicted labels are (nearly) identical - the disagreement count above shows that.")
    log("It also means the two baselines are not independent evidence.")


def section_ablation(X_all, y, folds):
    header("5. FEATURE-GROUP ABLATION (same fixed folds, logistic regression)")
    groups = {
        "all_75": EXPECTED,
        "intensity_only (no ROI/bbox)": [f"{b}_{s}" for b in INTENSITY_BASES for s in SUFFIXES],
        "geometry_only (ROI%/bbox)": [f"{b}_{s}" for b in GEOMETRY_BASES for s in SUFFIXES],
        "intensity_mean_aggregate_only (no dynamics)": [f"{b}_mean" for b in INTENSITY_BASES],
        "intensity_dynamics_only (_std + _slope)": [f"{b}_{s}" for b in INTENSITY_BASES for s in ["std", "slope"]],
    }
    rows = []
    for name, cols in groups.items():
        cols = [c for c in cols if c in X_all.columns]
        score, pred = run_cv(X_all[cols], y, folds)
        pooled, fold_mean_auc = summarize_cv(y, folds, score, pred)
        rows.append({"feature_set": name, "n_features": len(cols), "pooled_AUC": pooled["AUC"],
                     "fold_mean_AUC": fold_mean_auc, "sensitivity": pooled["sensitivity"],
                     "specificity": pooled["specificity"], "accuracy": pooled["accuracy"]})
    df = pd.DataFrame(rows).round(3)
    log(df.to_string(index=False))
    log()
    log("How to read this: if geometry_only is already high, ROI size/shape is a shortcut.")
    log("If intensity_only stays high and dynamics_only is near the mean-aggregate result, the")
    log("signal is mostly static intensity (which may depend on image scaling).")
    return df


def section_permutation(X, y, folds, n_perm, seed=0):
    header(f"6. LABEL-PERMUTATION TEST ({n_perm} permutations, labels shuffled within each fold)")
    score, pred = run_cv(X, y, folds)
    obs = safe_auc(y, score)
    rng = np.random.RandomState(seed)
    null = []
    for _ in range(n_perm):
        yp = y.copy()
        for f in np.unique(folds):
            idx = np.where(folds == f)[0]
            yp[idx] = rng.permutation(yp[idx])
        s, _ = run_cv(X, yp, folds)
        null.append(safe_auc(yp, s))
    null = np.array(null)
    p = (1 + (null >= obs).sum()) / (1 + n_perm)
    log(f"Observed pooled OOF AUC (all 75 features): {obs:.3f}")
    log(f"Null AUC: mean={null.mean():.3f}  95th pct={np.percentile(null, 95):.3f}  max={null.max():.3f}")
    log(f"Permutation p-value: {p:.4f}")
    log("Null mean should be close to 0.5. A null mean well above 0.5 indicates a pipeline leak")
    log("or a quirk of the fold structure (a small positive offset is normal with 56 patients).")


def section_bootstrap(y, score, pred, n_boot, seed=0):
    header(f"7. PATIENT-LEVEL BOOTSTRAP ({n_boot} resamples of pooled OOF predictions)")
    rng = np.random.RandomState(seed)
    n = len(y)
    aucs, sens, spec = [], [], []
    for _ in range(n_boot):
        i = rng.randint(0, n, n)
        yb, sb, pb = y[i], score[i], pred[i]
        if len(np.unique(yb)) < 2:
            continue
        m = binary_metrics(yb, pb, sb)
        aucs.append(m["AUC"]); sens.append(m["sensitivity"]); spec.append(m["specificity"])
    for name, vals in [("AUC", aucs), ("sensitivity", sens), ("specificity", spec)]:
        lo, hi = np.nanpercentile(vals, [2.5, 97.5])
        log(f"{name}: 95% bootstrap interval [{lo:.3f}, {hi:.3f}]")
    log("Percentile bootstrap with ~19 healthy / 37 sick patients is approximate and tends to be")
    log("optimistic near 1.0; treat as a rough indication of uncertainty, not exact coverage.")


def section_repeated(X_all, y, n_rep):
    header(f"8. REPEATED STRATIFIED 5-FOLD ({n_rep} different partitions; diagnostic only)")
    sets = {
        "all_75": EXPECTED,
        "intensity_only": [f"{b}_{s}" for b in INTENSITY_BASES for s in SUFFIXES],
    }
    for name, cols in sets.items():
        aucs, bals = [], []
        for seed in range(n_rep):
            folds = make_folds(y, 5, seed)
            score, pred = run_cv(X_all[cols], y, folds)
            m = binary_metrics(y, pred, score)
            aucs.append(m["AUC"]); bals.append((m["sensitivity"] + m["specificity"]) / 2)
        log(f"{name}: pooled AUC mean={np.mean(aucs):.3f} sd={np.std(aucs):.3f} "
            f"min={np.min(aucs):.3f};  balanced accuracy mean={np.mean(bals):.3f} sd={np.std(bals):.3f}")
    log("This does NOT replace the fixed folds used for model comparison; it only shows how much")
    log("the numbers depend on which patients happened to share a fold.")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="repository root (contains dmrir/)")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-rep", type=int, default=10)
    args = ap.parse_args()

    root = Path(args.root)
    d = root / "dmrir"
    feat = pd.read_csv(d / "dmrir_patient_features.csv")
    folds_df = pd.read_csv(d / "dmrir_patient_folds.csv")
    mpath = d / "dmrir_primary_manifest_folds.csv"
    manifest = pd.read_csv(mpath) if mpath.exists() else None

    cols = section_columns(feat)
    if len(cols) != len(EXPECTED):
        log("Stopping: expected 75 feature columns not all present.")
        return finish(d)
    # align feature rows with the folds file so every section uses identical order
    df = feat.drop(columns=[c for c in ["label", "fold"] if c in feat.columns]).merge(
        folds_df[["patient_id", "label", "fold"]], on="patient_id", how="inner")
    y = df["label"].values.astype(int)
    folds = df["fold"].values

    section_labels(feat, folds_df, manifest)
    section_univariate(df, cols, y)

    y_map = dict(zip(folds_df.patient_id, folds_df.label))
    fold_map = dict(zip(folds_df.patient_id, folds_df.fold))
    lr = load_oof(d / "results" / "oof_predictions.csv", "Logistic Regression", y_map, fold_map)
    svm = load_oof(d / "results" / "svm_oof_predictions.csv", "Linear SVM", y_map, fold_map)
    compare_oof(lr, svm)

    ablation = section_ablation(df[cols], y, folds)
    section_permutation(df[cols], y, folds, args.n_perm)

    if lr is not None:
        section_bootstrap(lr["_y"].values, lr["_s"].values, lr["_p"].values, args.n_boot)
    else:
        score, pred = run_cv(df[cols], y, folds)
        section_bootstrap(y, score, pred, args.n_boot)

    section_repeated(df[cols], y, args.n_rep)
    out = d / "results" / "audit"
    out.mkdir(parents=True, exist_ok=True)
    ablation.to_csv(out / "ablation_results.csv", index=False)
    finish(d)


def finish(d):
    out = d / "results" / "audit"
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit_report.txt").write_text("\n".join(REPORT), encoding="utf-8")
    log()
    log(f"Report saved to {out / 'audit_report.txt'}")


if __name__ == "__main__":
    main()