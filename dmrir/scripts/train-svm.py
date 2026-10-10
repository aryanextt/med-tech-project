
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

FEATURES_FILE = PROJECT_DIR / "dmrir" / "dmrir_patient_features.csv"
FOLDS_FILE = PROJECT_DIR / "dmrir" / "dmrir_patient_folds.csv"
RESULTS_DIR = PROJECT_DIR / "dmrir" / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

features = pd.read_csv(FEATURES_FILE)
folds = pd.read_csv(FOLDS_FILE)

features.columns = features.columns.str.strip()
folds.columns = folds.columns.str.strip()


def find_column(df, candidates, description):
    for name in candidates:
        if name in df.columns:
            return name

    raise ValueError(
        f"Cannot identify {description}. "
        f"Available columns: {df.columns.tolist()}"
    )


feature_id = find_column(
    features,
    ["patient_id", "Patient_ID", "patient", "Patient"],
    "patient ID in feature file",
)

fold_id = find_column(
    folds,
    ["patient_id", "Patient_ID", "patient", "Patient"],
    "patient ID in fold file",
)

feature_label = find_column(
    features,
    ["label", "Label", "target", "Target"],
    "feature label",
)

fold_label = find_column(
    folds,
    ["label", "Label", "target", "Target"],
    "fold label",
)

fold_number = find_column(
    folds,
    ["fold", "Fold", "fold_id", "Fold_ID"],
    "fold number",
)


# ============================================================
# 3. VALIDATE AND MERGE
# ============================================================

if features[feature_id].duplicated().any():
    raise ValueError("Duplicate patient IDs in feature file.")

if folds[fold_id].duplicated().any():
    raise ValueError("Duplicate patient IDs in fold file.")

if set(features[feature_id]) != set(folds[fold_id]):
    raise ValueError("Patient IDs differ between the two files.")

# Use the fold file as the source of fold assignments.
# Do not bring duplicate labels/folds into the model features.
fold_info = folds[[fold_id, fold_label, fold_number]].copy()

data = features.merge(
    fold_info,
    left_on=feature_id,
    right_on=fold_id,
    how="left",
    validate="one_to_one",
    suffixes=("", "_fold"),
)

if data[fold_number].isna().any():
    raise ValueError("Some patients have no fold assignment.")

if not np.array_equal(
    data[feature_label].astype(str).to_numpy(),
    data[fold_label].astype(str).to_numpy(),
):
    raise ValueError("Labels differ between the feature and fold files.")

if data[fold_number].nunique() != 5:
    raise ValueError("Expected exactly five folds.")


# ============================================================
# 4. ENCODE LABELS
# ============================================================

# Confirmed project convention:
# 0 = Healthy
# 1 = Sick

raw_labels = data[feature_label]
normalized_labels = raw_labels.astype(str).str.strip().str.lower()

if normalized_labels.isin(["healthy", "sick"]).all():
    y = normalized_labels.map({"healthy": 0, "sick": 1}).astype(int)
else:
    numeric_labels = pd.to_numeric(raw_labels, errors="coerce")

    if numeric_labels.isna().any():
        raise ValueError("Unexpected or missing labels.")

    if set(numeric_labels.unique()) != {0, 1}:
        raise ValueError("Expected numeric labels 0 and 1.")

    # Verify the mapping against label_name if present.
    if "label_name" in data.columns:
        label_check = data[["label", "label_name"]].drop_duplicates()
        observed = {}

        for _, row in label_check.iterrows():
            name = str(row["label_name"]).strip().lower()
            value = int(row["label"])

            if name not in {"healthy", "sick"}:
                raise ValueError(
                    f"Unexpected label_name value: {row['label_name']}"
                )

            observed[value] = name

        if observed != {0: "healthy", 1: "sick"}:
            raise ValueError(
                f"Label mapping does not match expected convention: {observed}"
            )

    y = numeric_labels.astype(int)

if set(y.unique()) != {0, 1}:
    raise ValueError("Both Healthy and Sick classes must be present.")

print("Patients:", len(data))
print("Label counts:", y.map({0: "Healthy", 1: "Sick"}).value_counts().to_dict())
print("Fold counts:", data[fold_number].value_counts().sort_index().to_dict())


# ============================================================
# 5. SELECT FEATURES; EXCLUDE METADATA
# ============================================================

metadata_columns = {
    feature_id,
    fold_id,
    feature_label,
    fold_label,
    fold_number,
    "patient_id",
    "patient",
    "label",
    "label_fold",
    "label_x",
    "label_y",
    "fold",
    "fold_fold",
    "fold_x",
    "fold_y",
    "label_name",
    "Label_Name",
    "class_name",
    "sequence_suffix",
}

candidate_data = data.drop(
    columns=[
        col for col in metadata_columns
        if col in data.columns
    ],
    errors="ignore",
)

X = candidate_data.select_dtypes(include=[np.number]).copy()

X = X.dropna(axis=1, how="all")

# Exclude constant features
constant_columns = [
    col for col in X.columns
    if X[col].nunique(dropna=True) <= 1
]

X = X.drop(columns=constant_columns)

if X.shape[1] == 0:
    raise ValueError("No usable numeric features found.")

forbidden_features = {
    "label",
    "label_fold",
    "label_x",
    "label_y",
    "fold",
    "fold_fold",
    "fold_x",
    "fold_y",
    "patient_id",
    "patient",
}

leaked = forbidden_features.intersection(X.columns)

if leaked:
    raise ValueError(
        f"Metadata leakage detected: {sorted(leaked)}"
    )

print("Leakage check passed.")
print("Numeric features:", X.shape[1])
print("Feature names:", X.columns.tolist())


# ============================================================
# 6. FIVE-FOLD SVM CROSS-VALIDATION
# ============================================================

fold_results = []
all_predictions = []

for fold in sorted(data[fold_number].unique()):

    train_mask = data[fold_number] != fold
    val_mask = data[fold_number] == fold

    X_train = X.loc[train_mask]
    X_val = X.loc[val_mask]

    y_train = y.loc[train_mask]
    y_val = y.loc[val_mask]

    if y_train.nunique() != 2 or y_val.nunique() != 2:
        raise ValueError(
            f"Fold {fold} must contain both classes in train and validation."
        )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="linear",
            C=1.0,
            class_weight="balanced",
            probability=True,
            random_state=42,
        )),
    ])

    # Fit preprocessing and model on training patients only.
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)
    probabilities = model.predict_proba(X_val)[:, 1]

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1],
    ).ravel()

    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan

    result = {
        "fold": fold,
        "n_train": int(train_mask.sum()),
        "n_validation": int(val_mask.sum()),
        "accuracy": accuracy_score(y_val, predictions),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision_PPV": ppv,
        "NPV": npv,
        "F1": f1_score(y_val, predictions, zero_division=0),
        "ROC_AUC": roc_auc_score(y_val, probabilities),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    fold_results.append(result)

    validation_patients = data.loc[val_mask, feature_id].tolist()

    for patient, actual, predicted, probability in zip(
        validation_patients,
        y_val.tolist(),
        predictions.tolist(),
        probabilities.tolist(),
    ):
        all_predictions.append({
            "patient_id": patient,
            "fold": fold,
            "true_label": int(actual),
            "true_label_name": "Sick" if actual == 1 else "Healthy",
            "predicted_label": int(predicted),
            "predicted_label_name": (
                "Sick" if predicted == 1 else "Healthy"
            ),
            "probability_sick": float(probability),
        })

    print(
        f"Fold {fold}: "
        f"Accuracy={result['accuracy']:.3f}, "
        f"Sensitivity={sensitivity:.3f}, "
        f"Specificity={specificity:.3f}, "
        f"F1={result['F1']:.3f}, "
        f"ROC-AUC={result['ROC_AUC']:.3f}"
    )


# ============================================================
# 7. SAVE RESULTS
# ============================================================

metrics_df = pd.DataFrame(fold_results)
predictions_df = pd.DataFrame(all_predictions)

metrics_df.to_csv(
    RESULTS_DIR / "svm_fold_metrics.csv",
    index=False,
)

predictions_df.to_csv(
    RESULTS_DIR / "svm_oof_predictions.csv",
    index=False,
)

metric_columns = [
    "accuracy",
    "sensitivity",
    "specificity",
    "precision_PPV",
    "NPV",
    "F1",
    "ROC_AUC",
]

summary_df = pd.DataFrame({
    "mean": metrics_df[metric_columns].mean(),
    "std": metrics_df[metric_columns].std(ddof=1),
})

summary_df.to_csv(
    RESULTS_DIR / "svm_metrics_summary.csv"
)


# ============================================================
# 8. REPORT
# ============================================================

print("\n" + "=" * 60)
print("SVM FOLD-LEVEL RESULTS")
print("=" * 60)
print(metrics_df.to_string(index=False))

print("\n" + "=" * 60)
print("SVM MEAN AND STANDARD DEVIATION")
print("=" * 60)
print(summary_df.to_string())

print("\nResults saved to:", RESULTS_DIR)
print("SVM baseline completed.")
