from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np


# ============================================================
# 1. PATHS
# ============================================================

DATASET_DIR = Path(
    r"C:\Users\LENOVO\OneDrive\medtech project\Imagens e Matrizes da Tese de Thiago Alves Elias da Silva"
)

MANIFEST_FILE = DATASET_DIR / "dmrir_primary_manifest_folds.csv"

OUTPUT_FILE = DATASET_DIR / "dmrir_roi_audit.csv"


# ============================================================
# 2. LOAD MANIFEST
# ============================================================

df = pd.read_csv(MANIFEST_FILE)

print("========================================")
print("DMR-IR ROI AUDIT")
print("========================================")

print(f"Total images: {len(df)}")
print(f"Total patients: {df['patient_id'].nunique()}")


# ============================================================
# 3. AUDIT EACH IMAGE
# ============================================================

records = []

for index, row in df.iterrows():

    image_path = Path(row["image_path"])

    try:

        with Image.open(image_path) as img:

            # Convert everything to grayscale
            img = img.convert("L")

            arr = np.array(img)

    except Exception as e:

        print(f"ERROR reading: {image_path}")
        print(e)
        continue


    # --------------------------------------------------------
    # Define ROI
    #
    # Existing segmented images have black background.
    # Pixels > 0 are considered part of the ROI.
    # --------------------------------------------------------

    roi_mask = arr > 0

    total_pixels = arr.size
    roi_pixels = roi_mask.sum()

    black_pixels = total_pixels - roi_pixels

    roi_percentage = (
        roi_pixels / total_pixels
    ) * 100

    black_percentage = (
        black_pixels / total_pixels
    ) * 100


    # --------------------------------------------------------
    # ROI intensity statistics
    # --------------------------------------------------------

    if roi_pixels > 0:

        roi_values = arr[roi_mask]

        roi_min = int(roi_values.min())
        roi_max = int(roi_values.max())

        roi_mean = float(
            roi_values.mean()
        )

        roi_std = float(
            roi_values.std()
        )

        # ----------------------------------------------------
        # ROI bounding box
        # ----------------------------------------------------

        ys, xs = np.where(roi_mask)

        x_min = int(xs.min())
        x_max = int(xs.max())

        y_min = int(ys.min())
        y_max = int(ys.max())

        bbox_width = x_max - x_min + 1
        bbox_height = y_max - y_min + 1

    else:

        roi_min = np.nan
        roi_max = np.nan
        roi_mean = np.nan
        roi_std = np.nan

        x_min = np.nan
        x_max = np.nan
        y_min = np.nan
        y_max = np.nan

        bbox_width = np.nan
        bbox_height = np.nan


    # --------------------------------------------------------
    # Basic flags
    # --------------------------------------------------------

    flags = []

    if roi_pixels == 0:
        flags.append("NO_ROI")

    if roi_percentage < 5:
        flags.append("VERY_SMALL_ROI")

    if roi_percentage > 80:
        flags.append("VERY_LARGE_ROI")

    if roi_std == 0:
        flags.append("ZERO_VARIATION")

    if roi_max <= 1:
        flags.append("VERY_LOW_INTENSITY")


    # --------------------------------------------------------
    # Save record
    # --------------------------------------------------------

    records.append({

        "patient_id": row["patient_id"],
        "label": row["label"],
        "label_name": row["label_name"],

        "fold": row["fold"],

        "frame": row["frame"],
        "filename": row["filename"],
        "image_path": row["image_path"],

        "image_width": arr.shape[1],
        "image_height": arr.shape[0],

        "total_pixels": total_pixels,

        "roi_pixels": int(roi_pixels),
        "roi_percentage": roi_percentage,

        "black_pixels": int(black_pixels),
        "black_percentage": black_percentage,

        "roi_min": roi_min,
        "roi_max": roi_max,
        "roi_mean": roi_mean,
        "roi_std": roi_std,

        "bbox_width": bbox_width,
        "bbox_height": bbox_height,

        "flags": "|".join(flags)

    })


# ============================================================
# 4. CREATE DATAFRAME
# ============================================================

audit_df = pd.DataFrame(records)


# ============================================================
# 5. SAVE AUDIT
# ============================================================

audit_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 6. SUMMARY
# ============================================================

print("\n========================================")
print("ROI SUMMARY")
print("========================================")

print(
    f"Images successfully analyzed: "
    f"{len(audit_df)}"
)

print(
    f"Mean ROI percentage: "
    f"{audit_df['roi_percentage'].mean():.2f}%"
)

print(
    f"Minimum ROI percentage: "
    f"{audit_df['roi_percentage'].min():.2f}%"
)

print(
    f"Maximum ROI percentage: "
    f"{audit_df['roi_percentage'].max():.2f}%"
)


print("\nROI percentage statistics:")

print(
    audit_df["roi_percentage"].describe()
)


# ============================================================
# 7. BOUNDING BOX SUMMARY
# ============================================================

print("\n========================================")
print("BOUNDING BOX")
print("========================================")

print(
    f"Mean width: "
    f"{audit_df['bbox_width'].mean():.2f}"
)

print(
    f"Mean height: "
    f"{audit_df['bbox_height'].mean():.2f}"
)

print(
    f"Minimum width: "
    f"{audit_df['bbox_width'].min():.0f}"
)

print(
    f"Maximum width: "
    f"{audit_df['bbox_width'].max():.0f}"
)

print(
    f"Minimum height: "
    f"{audit_df['bbox_height'].min():.0f}"
)

print(
    f"Maximum height: "
    f"{audit_df['bbox_height'].max():.0f}"
)


# ============================================================
# 8. INTENSITY SUMMARY
# ============================================================

print("\n========================================")
print("ROI INTENSITY")
print("========================================")

print(
    f"Mean ROI intensity: "
    f"{audit_df['roi_mean'].mean():.2f}"
)

print(
    f"Mean ROI standard deviation: "
    f"{audit_df['roi_std'].mean():.2f}"
)

print(
    f"Global minimum ROI intensity: "
    f"{audit_df['roi_min'].min():.0f}"
)

print(
    f"Global maximum ROI intensity: "
    f"{audit_df['roi_max'].max():.0f}"
)


# ============================================================
# 9. FLAGGED IMAGES
# ============================================================

flagged = audit_df[
    audit_df["flags"] != ""
]

print("\n========================================")
print("FLAGGED IMAGES")
print("========================================")

print(
    f"Flagged images: "
    f"{len(flagged)}"
)

if len(flagged) > 0:

    print("\nFlag distribution:")

    print(
        flagged["flags"].value_counts()
    )

    print("\nFirst 20 flagged images:")

    print(
        flagged[
            [
                "patient_id",
                "label_name",
                "frame",
                "roi_percentage",
                "roi_min",
                "roi_max",
                "flags"
            ]
        ].head(20).to_string(index=False)
    )

else:

    print("No abnormal images detected.")


# ============================================================
# 10. CHECK FOR MISSING VALUES
# ============================================================

print("\n========================================")
print("MISSING VALUE CHECK")
print("========================================")

print(
    f"Rows with missing ROI measurements: "
    f"{audit_df[['roi_percentage', 'roi_mean', 'roi_std']].isna().any(axis=1).sum()}"
)


# ============================================================
# 11. FINAL
# ============================================================

print("\n========================================")
print("AUDIT COMPLETE")
print("========================================")

print("Saved to:")

print(OUTPUT_FILE)