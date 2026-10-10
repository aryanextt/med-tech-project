#!/usr/bin/env python
"""
Confound checks for the DMR-IR feature baselines. Follow-up to audit_baselines.py.

Run from the repository root:
    python dmrir/scripts/audit_confounds.py --root .
    python dmrir/scripts/audit_confounds.py --root . --images     # also inspects raw PNGs

Questions it answers:
  A. Does patient_id (a stand-in for registration order / acquisition period) predict the label?
  B. The dir/esq suffix: every Healthy patient is "dir", but Sick patients are dir and esq.
     Are the model's errors concentrated in one suffix? Do esq-Sick patients score differently?
  C. If we keep only suffix == dir (19 Healthy vs 19 Sick, suffix constant), does the result hold?
  D. Do features separate dir-Sick from esq-Sick (i.e. does image side/orientation matter)?
  E. (--images) Do raw pixel values look like calibrated temperatures or display-scaled images,
     and do those properties differ between classes?

Only numpy, pandas, scikit-learn and Pillow (for --images) are used.
It never modifies existing files; it writes dmrir/results/audit/confound_report.txt.
"""
import argparse
import re
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
EXPECTED = [f"{b}_{s}" for b in BASES for s in SUFFIXES]
REPORT = []


def log(msg=""):
    print(msg)
    REPORT.append(str(msg))


def header(t):
    log()
    log("=" * 78)
    log(t)
    log("=" * 78)


def auc(y, s):
    y = np.asarray(y)
    return roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan


def pipe():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=5000)),
    ])


def stratified_folds(y, k, seed):
    rng = np.random.RandomState(seed)
    f = np.zeros(len(y), dtype=int)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            f[i] = j % k
    return f


def cv_pooled(X, y, folds):
    X = np.asarray(X, float)
    s = np.full(len(y), np.nan)
    p = np.full(len(y), -1)
    for f in np.unique(folds):
        te = folds == f
        m = pipe().fit(X[~te], y[~te])
        s[te] = m.predict_proba(X[te])[:, 1]
        p[te] = m.predict(X[te])
    tp = ((y == 1) & (p == 1)).sum(); fn = ((y == 1) & (p == 0)).sum()
    tn = ((y == 0) & (p == 0)).sum(); fp = ((y == 0) & (p == 1)).sum()
    return auc(y, s), tp / max(tp + fn, 1), tn / max(tn + fp, 1)


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]


def section_patient_id(df, y):
    header("A. PATIENT ID AS THE ONLY PREDICTOR (acquisition-order / batch check)")
    pid = df["patient_id"].astype(float).values
    a = auc(y, pid)
    log(f"AUC of patient_id alone (rank-based, 0.5 = no relation): {a:.3f}")
    for lab, name in [(0, "Healthy"), (1, "Sick")]:
        v = np.sort(pid[y == lab])
        log(f"  {name}: n={len(v)}  min={v.min():.0f}  median={np.median(v):.0f}  max={v.max():.0f}")
    log("Sorted patient_id with class (H=healthy, S=sick), to look for runs/blocks:")
    order = np.argsort(pid)
    log("  " + " ".join(f"{int(pid[i])}{'S' if y[i] else 'H'}" for i in order))
    log("A strong departure from 0.5, or long single-class runs, suggests healthy and sick")
    log("patients were recruited/imaged in different periods (a possible batch effect).")


def section_suffix(df, y, oof):
    header("B. dir / esq SUFFIX")
    suf = df["sequence_suffix"].astype(str).values
    log(pd.crosstab(pd.Series(suf, name="suffix"), pd.Series(y, name="label")).to_string())
    log("(Names are Portuguese for right/left, but whether they match image orientation or")
    log(" anatomical side is UNVERIFIED. What matters here is that the suffix is not independent")
    log(" of the class: no Healthy patient has esq.)")
    if oof is None:
        log("OOF file not found/recognised; skipping error-by-suffix table.")
        return
    m = df[["patient_id", "sequence_suffix"]].merge(oof, on="patient_id")
    m["error"] = (m["pred"] != m["y"]).astype(int)
    m["group"] = np.where(m["y"] == 1, "Sick", "Healthy") + "-" + m["sequence_suffix"].astype(str)
    t = m.groupby("group").agg(n=("error", "size"), errors=("error", "sum"),
                               mean_score=("score", "mean"), min_score=("score", "min"),
                               max_score=("score", "max")).round(3)
    log("Out-of-fold results by class and suffix (score = P(Sick)):")
    log(t.to_string())
    log("Misclassified patients: " + str(m.loc[m.error == 1, ["patient_id", "group", "score"]].round(3).values.tolist()))


def section_dir_only(df, cols, y, n_rep):
    header("C. dir-ONLY SUBSET (suffix constant: Healthy vs Sick cannot be separated by suffix)")
    keep = (df["sequence_suffix"].astype(str) == "dir").values
    d, yy = df.loc[keep].reset_index(drop=True), y[keep]
    log(f"Patients kept: {len(yy)}  (Healthy={int((yy == 0).sum())}, Sick={int((yy == 1).sum())})")
    for name, c in [("all_75", cols),
                    ("no ROI/bbox", [x for x in cols if not any(x.startswith(g + "_") for g in ["roi_percentage", "bbox_width", "bbox_height"])])]:
        res = []
        for seed in range(n_rep):
            res.append(cv_pooled(d[c], yy, stratified_folds(yy, 5, seed)))
        r = np.array(res)
        log(f"{name:12s} repeated 5-fold x{n_rep}: pooled AUC {r[:,0].mean():.3f} (sd {r[:,0].std():.3f}, "
            f"min {r[:,0].min():.3f});  sens {r[:,1].mean():.3f};  spec {r[:,2].mean():.3f}")
    allres = np.array([cv_pooled(df[cols], y, stratified_folds(y, 5, s)) for s in range(n_rep)])
    log(f"{'all patients':12s} same procedure on all 56: pooled AUC {allres[:,0].mean():.3f}")
    log("Compare dir-only with all-56. If dir-only stays high, the suffix is not what drives the")
    log("result. If it drops a lot, the esq patients (all Sick) were inflating it. 38 patients is small:")
    log("expect noisier estimates; do not over-read differences of a few hundredths.")


def section_side_features(df, cols, y):
    header("D. DO FEATURES SEPARATE dir-SICK FROM esq-SICK?")
    sick = df[y == 1]
    suf = (sick["sequence_suffix"].astype(str) == "esq").astype(int).values
    if len(np.unique(suf)) < 2:
        log("Only one suffix among Sick patients; skipped.")
        return
    rows = []
    for c in cols:
        x = sick[c].astype(float).fillna(sick[c].median()).values
        a = auc(suf, x)
        rows.append((c, max(a, 1 - a)))
    r = pd.DataFrame(rows, columns=["feature", "abs_AUC"]).sort_values("abs_AUC", ascending=False)
    log(f"Sick only: dir (n={int((suf == 0).sum())}) vs esq (n={int((suf == 1).sum())}); top features:")
    log(r.head(8).round(3).to_string(index=False))
    log(f"Mean |AUC| over all 75 features: {r.abs_AUC.mean():.3f} (about 0.60 is typical noise at this n)")
    log("High values mean images from the two suffixes differ systematically (framing/orientation/")
    log("mirroring), so mixing suffixes can create a side-related shortcut.")


def section_images(manifest, labels, n_frames=3):
    header("E. RAW IMAGE CHECK (display scaling vs calibrated temperature)")
    from PIL import Image
    pc = [c for c in manifest.columns if "path" in c.lower()]
    if not pc:
        log("No path column in manifest; skipped.")
        return
    pc = pc[0]
    rows = []
    for pid, g in manifest.groupby("patient_id"):
        paths = sorted(g[pc].astype(str).tolist(), key=natural_key)
        pick = [paths[0], paths[len(paths) // 2], paths[-1]][:n_frames]
        stats = []
        for p in pick:
            try:
                a = np.asarray(Image.open(p).convert("L"))
            except Exception as e:  # noqa
                continue
            roi = a[a > 0]
            if roi.size == 0:
                continue
            stats.append([roi.min(), roi.max(), np.percentile(roi, 1), np.percentile(roi, 99),
                          len(np.unique(roi)), (roi == roi.max()).mean(), (roi >= 250).mean()])
        if stats:
            s = np.mean(stats, axis=0)
            rows.append({"patient_id": pid, "roi_min": s[0], "roi_max": s[1], "p01": s[2], "p99": s[3],
                         "n_unique_levels": s[4], "frac_at_roi_max": s[5], "frac_ge_250": s[6]})
    if not rows:
        log("Could not read any images (check the paths in the manifest on this machine).")
        return
    df = pd.DataFrame(rows).merge(labels, on="patient_id")
    log(f"Patients with readable images: {len(df)}  (3 frames each: first, middle, last by frame number)")
    cols = ["roi_min", "roi_max", "p01", "p99", "n_unique_levels", "frac_at_roi_max", "frac_ge_250"]
    summ = df.groupby("label")[cols].mean().round(3)
    log("Class means (label 0 = Healthy, 1 = Sick):")
    log(summ.to_string())
    log("Per-property AUC for separating the classes (0.5 = none):")
    for c in cols:
        a = auc(df["label"].values, df[c].values)
        log(f"  {c:18s} {max(a, 1 - a):.3f}")
    log("Interpretation guide: if every frame's ROI is stretched to the same min/max (roi_max ~255,")
    log("many pixels at max, similar across patients), the images are display-scaled and absolute")
    log("intensity is NOT temperature. If roi_max or n_unique_levels differ strongly by class, the")
    log("classes may have been exported with different scaling/windowing, which a classifier can")
    log("exploit without any physiological signal. This check cannot prove which case applies;")
    log("the dataset documentation / camera export settings are the authority.")


def load_oof(path):
    if not path.exists():
        return None
    o = pd.read_csv(path)
    def pick(opts):
        for c in opts:
            if c in o.columns:
                return c
    pid, y, s, p = (pick(["patient_id"]), pick(["true_label", "label", "y_true"]),
                    pick(["probability_sick", "probability", "y_score", "score"]),
                    pick(["predicted_label", "prediction", "y_pred", "pred"]))
    if None in (pid, y, s, p):
        return None
    return o[[pid, y, s, p]].rename(columns={pid: "patient_id", y: "y", s: "score", p: "pred"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--n-rep", type=int, default=20)
    ap.add_argument("--images", action="store_true", help="inspect raw PNGs listed in the manifest")
    a = ap.parse_args()
    d = Path(a.root) / "dmrir"

    feat = pd.read_csv(d / "dmrir_patient_features.csv")
    folds = pd.read_csv(d / "dmrir_patient_folds.csv")[["patient_id", "label", "fold"]]
    df = feat.drop(columns=[c for c in ["label", "fold"] if c in feat.columns]).merge(folds, on="patient_id")
    y = df["label"].values.astype(int)
    cols = [c for c in EXPECTED if c in df.columns]

    section_patient_id(df, y)
    section_suffix(df, y, load_oof(d / "results" / "oof_predictions.csv"))
    section_dir_only(df, cols, y, a.n_rep)
    section_side_features(df, cols, y)
    if a.images:
        mp = d / "dmrir_primary_manifest_folds.csv"
        if mp.exists():
            section_images(pd.read_csv(mp), folds[["patient_id", "label"]])
        else:
            log("Manifest not found; skipping E.")
    else:
        log()
        log("(Section E skipped; rerun with --images to inspect raw pixel scaling.)")

    out = d / "results" / "audit"
    out.mkdir(parents=True, exist_ok=True)
    (out / "confound_report.txt").write_text("\n".join(REPORT), encoding="utf-8")
    log(f"\nReport saved to {out / 'confound_report.txt'}")


if __name__ == "__main__":
    main()