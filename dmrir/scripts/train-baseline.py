
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "dmrir_patient_features.csv"
OUTPUT_DIR = PROJECT_ROOT / "dmrir" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE)

required = {"patient_id", "label", "fold"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

# Metadata and target must never enter the feature matrix.
excluded = {"patient_id", "label", "label_name", "fold", "sequence_suffix"}
feature_cols = [
    c for c in df.columns
    if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
]

if not feature_cols:
    raise ValueError("No numeric feature columns found.")

if df["patient_id"].duplicated().any():
    raise ValueError("Expected one row per patient, but duplicate patient IDs exist.")

if df["label"].isna().any() or df["fold"].isna().any():
    raise ValueError("Missing labels or fold assignments found.")

X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
y = df["label"].astype(int)
folds = sorted(df["fold"].unique())

all_predictions = []
fold_metrics = []

for fold in folds:
    train_mask = df["fold"] != fold
    test_mask = df["fold"] == fold

    X_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_train, y_test = y.loc[train_mask], y.loc[test_mask]

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(f"Fold {fold} does not have both classes in train and test.")

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=5000,
            random_state=42,
        ),
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "fold": int(fold),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "tn_fp_fn_tp": confusion_matrix(
            y_test, pred, labels=[0, 1]
        ).ravel().tolist(),
    }
    fold_metrics.append(metrics)

    for idx, prediction, probability in zip(
        df.index[test_mask], pred, probabilities
    ):
        all_predictions.append({
            "patient_id": df.loc[idx, "patient_id"],
            "fold": int(fold),
            "actual_label": int(y.loc[idx]),
            "predicted_label": int(prediction),
            "probability_class_1": float(probability),
        })

metrics_df = pd.DataFrame(fold_metrics)
predictions_df = pd.DataFrame(all_predictions)

metrics_df.to_csv(OUTPUT_DIR / "baseline_fold_metrics.csv", index=False)
predictions_df.to_csv(OUTPUT_DIR / "baseline_predictions.csv", index=False)

summary = {
    "n_patients": int(len(df)),
    "n_features": int(len(feature_cols)),
    "folds": [int(f) for f in folds],
    "mean_accuracy": float(metrics_df["accuracy"].mean()),
    "mean_balanced_accuracy": float(metrics_df["balanced_accuracy"].mean()),
    "mean_precision": float(metrics_df["precision"].mean()),
    "mean_recall": float(metrics_df["recall"].mean()),
    "mean_f1": float(metrics_df["f1"].mean()),
    "mean_roc_auc": float(metrics_df["roc_auc"].mean()),
}

with open(OUTPUT_DIR / "baseline_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n=== Fold metrics ===")
print(metrics_df.to_string(index=False))
print("\n=== Mean across folds ===")
for key, value in summary.items():
    print(f"{key}: {value}")
print(f"\nSaved results to: {OUTPUT_DIR}")

