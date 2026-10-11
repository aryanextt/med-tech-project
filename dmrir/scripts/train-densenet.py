"""
DenseNet121 training-only baseline for the DMR-IR dataset.

- Uses existing patient-level five-fold splits.
- Stage 1: Train classifier head with frozen backbone.
- Stage 2: Fine-tune denseblock4 and classifier.
- Uses ImageNet-pretrained weights.
- Requires CUDA; does not silently fall back to CPU.
- Saves checkpoints and loss histories.
- Resumes Stage 2 from an existing Stage 1 checkpoint.
- Does not calculate evaluation metrics.
"""

from pathlib import Path
import random
import json

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from torchvision.models import DenseNet121_Weights
from torch.utils.data import Dataset, DataLoader


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = PROJECT_ROOT / "dmrir" / "dmrir_primary_manifest_v2.csv"
FOLDS_PATH = PROJECT_ROOT / "dmrir" / "dmrir_patient_folds.csv"
RESULTS_ROOT = PROJECT_ROOT / "dmrir" / "results" / "densenet121"

NUM_EPOCHS_STAGE1 = 15
NUM_EPOCHS_STAGE2 = 10

BATCH_SIZE = 16
NUM_WORKERS = 2

LR_STAGE1 = 1e-3
LR_STAGE2 = 1e-4

SEED = 42

IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ============================================================
# REPRODUCIBILITY AND DEVICE
# ============================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Activate densenet-env and verify "
            "the CUDA-enabled PyTorch installation before training."
        )

    device = torch.device("cuda")
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    return device


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def resize_and_pad(image):
    """Preserve the 4:3 aspect ratio and pad to 224x224."""
    image = image.convert("RGB")

    width, height = image.size
    scale = IMAGE_SIZE / height

    new_width = round(width * scale)
    new_height = IMAGE_SIZE

    image = image.resize(
        (new_width, new_height),
        Image.Resampling.BILINEAR
    )

    canvas = Image.new(
        "RGB",
        (IMAGE_SIZE, IMAGE_SIZE),
        (0, 0, 0)
    )

    left = (IMAGE_SIZE - new_width) // 2
    canvas.paste(image, (left, 0))

    return canvas


train_transform = T.Compose([
    T.Lambda(resize_and_pad),
    T.RandomAffine(
        degrees=10,
        translate=(0.05, 0.05)
    ),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

validation_transform = T.Compose([
    T.Lambda(resize_and_pad),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])


# ============================================================
# DATASET
# ============================================================

class DMRIRDataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_path = Path(row["image_path"])

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(
            float(row["label"]),
            dtype=torch.float32
        )

        patient_id = str(row["patient_id"])

        return image, label, patient_id


# ============================================================
# MANIFEST AND FOLD LOADING
# ============================================================

def find_column(dataframe, candidates, description):
    """Find a column using case-insensitive exact matching."""
    lookup = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate]

    raise ValueError(
        f"Could not find {description}. "
        f"Available columns: {list(dataframe.columns)}"
    )


def load_data():
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    if not FOLDS_PATH.is_file():
        raise FileNotFoundError(
            f"Fold file not found: {FOLDS_PATH}"
        )

    manifest = pd.read_csv(MANIFEST_PATH)
    folds = pd.read_csv(FOLDS_PATH)

    patient_col = find_column(
        manifest,
        ["patient_id", "patient", "patientid"],
        "patient ID column in manifest"
    )

    label_col = find_column(
        manifest,
        ["label", "target", "class_label"],
        "label column in manifest"
    )

    path_col = find_column(
        manifest,
        ["image_path", "filepath", "file_path", "path"],
        "image path column in manifest"
    )

    fold_patient_col = find_column(
        folds,
        ["patient_id", "patient", "patientid"],
        "patient ID column in folds file"
    )

    fold_col = find_column(
        folds,
        ["fold", "fold_id", "fold_number"],
        "fold assignment column"
    )

    manifest = manifest.rename(columns={
        patient_col: "patient_id",
        label_col: "label",
        path_col: "image_path",
    })

    folds = folds.rename(columns={
        fold_patient_col: "patient_id",
        fold_col: "fold",
    })

    manifest["patient_id"] = manifest["patient_id"].astype(str)
    folds["patient_id"] = folds["patient_id"].astype(str)

    if folds["patient_id"].duplicated().any():
        raise ValueError(
            "The folds file has duplicate patient IDs. "
            "Resolve these before training."
        )

    manifest["label"] = pd.to_numeric(
        manifest["label"],
        errors="raise"
    )

    if not set(manifest["label"].unique()).issubset({0, 1}):
        raise ValueError(
            "Expected binary labels: 0 = Healthy, 1 = Sick. "
            "Inspect the manifest before training."
        )

    manifest["label"] = manifest["label"].astype(int)

    missing_patients = (
        set(manifest["patient_id"])
        - set(folds["patient_id"])
    )

    if missing_patients:
        raise ValueError(
            f"Patients without fold assignments: {missing_patients}"
        )

    # Remove a label column from the fold table if present.
    # The manifest remains the source of labels.
    duplicate_label_columns = [
        column for column in folds.columns
        if column.lower() in {"label", "target", "class_label"}
    ]

    if duplicate_label_columns:
        folds = folds.drop(columns=duplicate_label_columns)

    data = manifest.merge(
        folds[["patient_id", "fold"]],
        on="patient_id",
        how="left",
        validate="many_to_one"
    )

    if data["fold"].isna().any():
        raise ValueError("Some image rows lack fold assignments.")

    data["fold"] = data["fold"].astype(int)

    # Each patient must have one consistent label.
    label_counts = data.groupby("patient_id")["label"].nunique()

    if (label_counts > 1).any():
        raise ValueError(
            "At least one patient has inconsistent labels."
        )

    # Confirm patient-level folds are non-overlapping.
    patient_fold_counts = data.groupby("patient_id")["fold"].nunique()

    if (patient_fold_counts != 1).any():
        raise ValueError(
            "A patient appears in more than one fold."
        )

    return data


# ============================================================
# MODEL
# ============================================================

def get_model(stage):
    # Use explicitly specified ImageNet pretrained weights.
    model = models.densenet121(
        weights=DenseNet121_Weights.IMAGENET1K_V1
    )

    number_of_features = model.classifier.in_features

    model.classifier = nn.Linear(
        number_of_features,
        1
    )

    if stage == 1:
        for parameter in model.parameters():
            parameter.requires_grad = False

        for parameter in model.classifier.parameters():
            parameter.requires_grad = True

    elif stage == 2:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = (
                "denseblock4" in name
                or name.startswith("classifier.")
            )

    else:
        raise ValueError(f"Unknown training stage: {stage}")

    return model


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):
    model.train()

    total_loss = 0.0
    total_images = 0

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_images += batch_size

    return total_loss / max(total_images, 1)


def save_checkpoint(
    checkpoint_path,
    fold,
    stage,
    model,
    optimizer,
    loss_history,
    train_df,
    validation_df,
    stage1_history=None,
):
    checkpoint = {
        "fold": int(fold),
        "stage": stage,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss_history": loss_history,
        "loss_history_stage1": stage1_history or [],
        "class_mapping": {
            0: "Healthy",
            1: "Sick",
        },
        "preprocess": {
            "resize": "height 224 preserving 4:3 aspect ratio",
            "padding": "224x224 black background",
            "normalize_mean": MEAN,
            "normalize_std": STD,
        },
        "training_config": {
            "epochs_stage1": NUM_EPOCHS_STAGE1,
            "epochs_stage2": NUM_EPOCHS_STAGE2,
            "batch_size": BATCH_SIZE,
            "learning_rate_stage1": LR_STAGE1,
            "learning_rate_stage2": LR_STAGE2,
            "optimizer": "Adam",
            "loss": "BCEWithLogitsLoss",
        },
        "patient_ids": sorted(
            train_df["patient_id"].unique().tolist()
        ),
        "validation_patient_ids": sorted(
            validation_df["patient_id"].unique().tolist()
        ),
    }

    torch.save(checkpoint, checkpoint_path)


def main():
    set_seed()
    device = get_device()

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    data = load_data()

    print(f"Total images: {len(data)}")
    print(f"Total patients: {data['patient_id'].nunique()}")
    print(f"Fold IDs: {sorted(data['fold'].unique().tolist())}")

    # Save experiment configuration separately.
    config = {
        "model": "DenseNet121",
        "pretrained_weights": "IMAGENET1K_V1",
        "epochs_stage1": NUM_EPOCHS_STAGE1,
        "epochs_stage2": NUM_EPOCHS_STAGE2,
        "batch_size": BATCH_SIZE,
        "learning_rate_stage1": LR_STAGE1,
        "learning_rate_stage2": LR_STAGE2,
        "num_workers": NUM_WORKERS,
        "device": str(device),
        "class_mapping": {
            "0": "Healthy",
            "1": "Sick",
        },
    }

    with open(
        RESULTS_ROOT / "training_config.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(config, file, indent=2)

    for fold in sorted(data["fold"].unique()):
        print(f"\n========== Fold {fold} ==========")

        train_df = data[data["fold"] != fold].copy()
        validation_df = data[data["fold"] == fold].copy()

        train_patients = set(train_df["patient_id"])
        validation_patients = set(validation_df["patient_id"])

        overlap = train_patients.intersection(validation_patients)

        if overlap:
            raise RuntimeError(
                f"Patient leakage in fold {fold}: {overlap}"
            )

        print(
            f"Training patients: {len(train_patients)} | "
            f"Validation patients: {len(validation_patients)}"
        )

        train_dataset = DMRIRDataset(
            train_df,
            transform=train_transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        # Use the manifest's 0/1 labels.
        positive_ratio = float(train_df["label"].mean())

        if positive_ratio <= 0 or positive_ratio >= 1:
            raise ValueError(
                f"Fold {fold} training data contains only one class."
            )

        positive_weight = (1.0 - positive_ratio) / positive_ratio

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                [positive_weight],
                dtype=torch.float32,
                device=device
            )
        )

        stage1_path = (
            RESULTS_ROOT / f"model_fold_{fold}_stage1.pth"
        )

        final_path = (
            RESULTS_ROOT / f"model_fold_{fold}_final.pth"
        )

        # ----------------------------------------------------
        # Stage 1: frozen backbone
        # ----------------------------------------------------

        if stage1_path.exists():
            print(
                "Stage 1 checkpoint exists. "
                "Loading it and skipping Stage 1."
            )

            # Only load checkpoints generated by this trusted script.
            stage1_checkpoint = torch.load(
                stage1_path,
                map_location=device,
                weights_only=False
            )

            if stage1_checkpoint.get("fold") != int(fold):
                raise ValueError(
                    f"Checkpoint fold does not match fold {fold}."
                )

            if stage1_checkpoint.get("stage") != "stage1":
                raise ValueError(
                    f"Unexpected checkpoint stage in {stage1_path}"
                )

            loss_history_stage1 = stage1_checkpoint.get(
                "loss_history",
                []
            )

        else:
            print("Stage 1: training classifier head")

            model = get_model(stage=1).to(device)

            optimizer = torch.optim.Adam(
                filter(
                    lambda parameter: parameter.requires_grad,
                    model.parameters()
                ),
                lr=LR_STAGE1
            )

            loss_history_stage1 = []

            for epoch in range(1, NUM_EPOCHS_STAGE1 + 1):
                loss = train_one_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    device
                )

                loss_history_stage1.append(loss)

                print(
                    f"Fold {fold} | Stage 1 | "
                    f"Epoch {epoch}/{NUM_EPOCHS_STAGE1} | "
                    f"Loss: {loss:.4f}"
                )

            save_checkpoint(
                stage1_path,
                fold,
                "stage1",
                model,
                optimizer,
                loss_history_stage1,
                train_df,
                validation_df,
            )

            print(f"Stage 1 checkpoint saved: {stage1_path}")

            del model, optimizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ----------------------------------------------------
        # Stage 2: fine-tune denseblock4
        # ----------------------------------------------------

        if final_path.exists():
            print(
                "Final checkpoint already exists. "
                "Skipping this fold."
            )
            continue

        print("Stage 2: fine-tuning denseblock4")

        model = get_model(stage=2).to(device)

        # Checkpoint is created locally by this training script.
        stage1_checkpoint = torch.load(
            stage1_path,
            map_location=device,
            weights_only=False
        )

        model.load_state_dict(
            stage1_checkpoint["model_state_dict"],
            strict=True
        )

        optimizer = torch.optim.Adam(
            filter(
                lambda parameter: parameter.requires_grad,
                model.parameters()
            ),
            lr=LR_STAGE2
        )

        loss_history_stage2 = []

        for epoch in range(1, NUM_EPOCHS_STAGE2 + 1):
            loss = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device
            )

            loss_history_stage2.append(loss)

            print(
                f"Fold {fold} | Stage 2 | "
                f"Epoch {epoch}/{NUM_EPOCHS_STAGE2} | "
                f"Loss: {loss:.4f}"
            )

        save_checkpoint(
            final_path,
            fold,
            "final",
            model,
            optimizer,
            loss_history_stage2,
            train_df,
            validation_df,
            stage1_history=loss_history_stage1,
        )

        print(f"Final checkpoint saved: {final_path}")

        del model, optimizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nTraining loop finished.")
    print(f"Checkpoint directory: {RESULTS_ROOT}")


if __name__ == "__main__":
    main()