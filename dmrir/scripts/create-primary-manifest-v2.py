from pathlib import Path
import re
import pandas as pd

# ============================================================
# 1. DATASET PATH
# ============================================================

DATASET_DIR = Path(
    r"C:\Users\LENOVO\OneDrive\medtech project\Imagens e Matrizes da Tese de Thiago Alves Elias da Silva"
)

OUTPUT_FILE = DATASET_DIR / "dmrir_primary_manifest_v2.csv"


# ============================================================
# 2. DATASET STRUCTURE
# ============================================================

SECTIONS = [
    "12 Novos Casos de Testes",
    "Desenvolvimento da Metodologia"
]

# DOENTE = Sick
# SAUDÁVEIS = Healthy
LABEL_MAP = {
    "DOENTE": (1, "Sick"),
    "SAUD": (0, "Healthy")
}


# ============================================================
# 3. IMAGE FILENAME PATTERN
# ============================================================

# Example:
# PAC_02_DN0-dir.png
# PAC_02_DN19-esq.png

pattern = re.compile(
    r"_DN(\d+)-(dir|esq)\.png$",
    re.IGNORECASE
)


# ============================================================
# 4. FIND PATIENT FOLDERS
# ============================================================

patient_records = []

for section_name in SECTIONS:

    section_path = DATASET_DIR / section_name

    if not section_path.exists():
        print(f"WARNING: Section not found: {section_path}")
        continue

    for status_folder in section_path.iterdir():

        if not status_folder.is_dir():
            continue

        status_name = status_folder.name.upper()

        label_info = None

        for key, value in LABEL_MAP.items():
            if key in status_name:
                label_info = value
                break

        if label_info is None:
            continue

        label, label_name = label_info

        # ----------------------------------------------------
        # Each folder inside DOENTE / SAUDÁVEIS = one patient
        # ----------------------------------------------------

        for patient_folder in status_folder.iterdir():

            if not patient_folder.is_dir():
                continue

            patient_id = patient_folder.name

            png_files = list(patient_folder.rglob("*.png"))

            # ------------------------------------------------
            # Group dynamic images by suffix
            # ------------------------------------------------

            sequences = {
                "dir": {},
                "esq": {}
            }

            for img_path in png_files:

                match = pattern.search(img_path.name)

                if match is None:
                    continue

                frame = int(match.group(1))
                suffix = match.group(2).lower()

                if 0 <= frame <= 19:
                    sequences[suffix][frame] = img_path

            # ------------------------------------------------
            # Check which sequence is complete
            # ------------------------------------------------

            required_frames = set(range(20))

            complete_sequences = []

            for suffix in ["dir", "esq"]:

                available_frames = set(sequences[suffix].keys())

                if available_frames == required_frames:
                    complete_sequences.append(suffix)

            # ------------------------------------------------
            # Select ONE complete sequence
            #
            # If both dir and esq exist, choose dir
            # consistently.
            #
            # If only esq exists, choose esq.
            # ------------------------------------------------

            if "dir" in complete_sequences:
                selected_suffix = "dir"

            elif "esq" in complete_sequences:
                selected_suffix = "esq"

            else:
                print(
                    f"WARNING: No complete 20-frame sequence "
                    f"for patient {patient_id}"
                )
                continue

            # ------------------------------------------------
            # Add exactly 20 frames
            # ------------------------------------------------

            for frame in range(20):

                img_path = sequences[selected_suffix][frame]

                patient_records.append({
                    "patient_id": patient_id,
                    "label": label,
                    "label_name": label_name,
                    "section": section_name,
                    "status_folder": status_folder.name,
                    "sequence_suffix": selected_suffix,
                    "frame": frame,
                    "filename": img_path.name,
                    "image_path": str(img_path)
                })


# ============================================================
# 5. CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(patient_records)


# ============================================================
# 6. VALIDATION
# ============================================================

print("\n========================================")
print("PRIMARY MANIFEST V2 VALIDATION")
print("========================================")

print(f"Total images: {len(df)}")
print(f"Total patients: {df['patient_id'].nunique()}")

print("\nPatients by class:")
print(
    df.groupby(["label", "label_name"])["patient_id"]
      .nunique()
)

print("\nImages by class:")
print(
    df.groupby(["label", "label_name"]).size()
)

print("\nImages per patient:")
print(
    df.groupby("patient_id").size().value_counts().sort_index()
)

print("\nSequence suffix selected:")
print(
    df.groupby(["label_name", "sequence_suffix"])["patient_id"]
      .nunique()
)

print("\nFrames per patient:")
print(
    df.groupby("patient_id")["frame"]
      .agg(["min", "max", "count"])
      .head(10)
)


# ============================================================
# 7. SAFETY CHECKS
# ============================================================

assert df["patient_id"].nunique() == 56, \
    "ERROR: Expected 56 patients."

assert len(df) == 1120, \
    "ERROR: Expected 1120 images."

assert df.groupby("patient_id").size().eq(20).all(), \
    "ERROR: Every patient must have exactly 20 images."

assert df.groupby("patient_id")["frame"].apply(
    lambda x: set(x) == set(range(20))
).all(), \
    "ERROR: Every patient must contain frames DN0-DN19."

assert df["image_path"].nunique() == len(df), \
    "ERROR: Duplicate image paths detected."

assert not df["filename"].str.contains(
    "ESTATICO",
    case=False,
    regex=False
).any(), \
    "ERROR: Static images found in primary manifest."

assert set(df["label_name"].unique()) == {"Healthy", "Sick"}, \
    "ERROR: Unexpected labels found."


# ============================================================
# 8. SAVE
# ============================================================

df = df.sort_values(
    by=["label", "patient_id", "frame"]
).reset_index(drop=True)

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\n========================================")
print("SUCCESS")
print("========================================")

print(f"Manifest saved to:")
print(OUTPUT_FILE)

print("\nFinal expected result:")
print("56 patients")
print("37 Sick")
print("19 Healthy")
print("20 dynamic frames per patient")
print("1120 total images")
print("Static images excluded")
print("Patient-level unit preserved")