from pathlib import Path
import argparse
import os
import subprocess

import nibabel as nib
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# ARGOMENTI
# ---------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Preprocess MRI volumes for FlowLet training."
)

parser.add_argument(
    "--input_csv",
    default="/mnt/data/Palmisano/datasets/main_dataset_catalog.csv",
    help="Catalog containing FilePath, SubjectID, Age, Condition, etc.",
)

parser.add_argument(
    "--output_root",
    default="/mnt/data/Palmisano/datasets/ADNI_flowlet_preprocessed",
)

parser.add_argument(
    "--output_csv",
    default="/mnt/data/Palmisano/datasets/main_dataset_catalog_preprocessed.csv",
)

parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Reprocess images even if the final output already exists.",
)

parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Optional limit for testing.",
)

args = parser.parse_args()


INPUT_CSV = Path(args.input_csv)
OUTPUT_ROOT = Path(args.output_root)
OUTPUT_CSV = Path(args.output_csv)

FSLDIR = Path("/mnt/data/Palmisano/tools/fsl")

MNI = FSLDIR / "data/standard/MNI152_T1_2mm.nii.gz"
FLIRT = FSLDIR / "share/fsl/bin/flirt"
BET = FSLDIR / "share/fsl/bin/bet"


# ---------------------------------------------------------
# AMBIENTE FSL
# ---------------------------------------------------------

env = os.environ.copy()

env["FSLDIR"] = str(FSLDIR)
env["FSLOUTPUTTYPE"] = "NIFTI_GZ"

env["PATH"] = (
    f"{FSLDIR / 'bin'}:"
    f"{FSLDIR / 'share/fsl/bin'}:"
    f"{env.get('PATH', '')}"
)


# ---------------------------------------------------------
# CONTROLLI
# ---------------------------------------------------------

for path, name in [
    (INPUT_CSV, "Input CSV"),
    (MNI, "MNI template"),
    (FLIRT, "FLIRT"),
    (BET, "BET"),
]:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")


df = pd.read_csv(INPUT_CSV)

required_columns = [
    "FilePath",
    "SubjectID",
    "Age",
    "Condition",
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


if args.limit is not None:
    df = df.head(args.limit).copy()


OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

records = []
failed = []


# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------

for index, row in df.iterrows():

    subject = str(row["SubjectID"]).strip()

    # ImageID evita collisioni se in futuro avremo
    # più visite dello stesso soggetto.
    if "ImageID" in row and pd.notna(row["ImageID"]):
        image_id = str(row["ImageID"]).strip()
    else:
        image_id = f"scan_{index:05d}"

    scan_id = f"{subject}_{image_id}"

    input_path = Path(str(row["FilePath"]))

    out_dir = OUTPUT_ROOT / scan_id

    mni_path = out_dir / "mni.nii.gz"
    affine_path = out_dir / "affine.mat"
    brain_path = out_dir / "brain.nii.gz"
    final_path = out_dir / "brain_zscore.nii.gz"

    print("\n" + "=" * 75)
    print(f"[{index + 1}/{len(df)}] {scan_id}")
    print(f"Input: {input_path}")
    print("=" * 75)

    if not input_path.exists():
        print("ERROR: input file does not exist.")
        failed.append((scan_id, "input missing"))
        continue

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Se già preprocessato, non rifarlo
    # -----------------------------------------------------

    if final_path.exists() and not args.overwrite:

        try:
            img = nib.load(final_path)

            if img.shape != (91, 109, 91):
                raise ValueError(
                    f"Existing output has wrong shape: {img.shape}"
                )

            print("Already processed -> skipping.")

            new_row = row.copy()
            new_row["OriginalFilePath"] = str(input_path.resolve())
            new_row["FilePath"] = str(final_path.resolve())

            records.append(new_row)
            continue

        except Exception as exc:
            print(
                f"Existing output invalid ({exc}), reprocessing..."
            )

    try:

        # -------------------------------------------------
        # 1. FLIRT: registrazione affine MNI152 2 mm
        # -------------------------------------------------

        print("[1/3] FLIRT -> MNI152 2 mm")

        subprocess.run(
            [
                str(FLIRT),
                "-in",
                str(input_path),
                "-ref",
                str(MNI),
                "-out",
                str(mni_path),
                "-omat",
                str(affine_path),
                "-dof",
                "12",
                "-interp",
                "trilinear",
            ],
            check=True,
            env=env,
        )

        mni_img = nib.load(mni_path)

        print(
            "      shape:",
            mni_img.shape,
        )

        print(
            "      spacing:",
            mni_img.header.get_zooms()[:3],
        )

        if mni_img.shape != (91, 109, 91):
            raise ValueError(
                f"Unexpected MNI shape: {mni_img.shape}"
            )

        # -------------------------------------------------
        # 2. BET: skull stripping
        # -------------------------------------------------

        print("[2/3] BET -> skull stripping")

        subprocess.run(
            [
                str(BET),
                str(mni_path),
                str(brain_path),
                "-R",
                "-f",
                "0.5",
                "-g",
                "0",
            ],
            check=True,
            env=env,
        )

        if not brain_path.exists():
            raise FileNotFoundError(
                f"BET output missing: {brain_path}"
            )

        # -------------------------------------------------
        # 3. Z-SCORE
        # -------------------------------------------------

        print("[3/3] Z-score normalization")

        img = nib.load(brain_path)

        data = img.get_fdata().astype(
            np.float32
        )

        mask = data != 0

        if not np.any(mask):
            raise ValueError(
                "Empty brain mask after BET."
            )

        mean = float(
            data[mask].mean()
        )

        std = float(
            data[mask].std()
        )

        if std <= 0:
            raise ValueError(
                "Invalid standard deviation."
            )

        normalized = np.zeros_like(
            data,
            dtype=np.float32,
        )

        normalized[mask] = (
            data[mask] - mean
        ) / (std + 1e-8)

        header = img.header.copy()
        header.set_data_dtype(
            np.float32
        )

        final_img = nib.Nifti1Image(
            normalized,
            img.affine,
            header,
        )

        nib.save(
            final_img,
            final_path,
        )

        # -------------------------------------------------
        # VALIDAZIONE OUTPUT
        # -------------------------------------------------

        check_img = nib.load(final_path)
        check_data = check_img.get_fdata()

        check_mask = check_data != 0

        if check_img.shape != (91, 109, 91):
            raise ValueError(
                f"Final shape invalid: {check_img.shape}"
            )

        print(
            "      final shape:",
            check_img.shape,
        )

        print(
            "      final spacing:",
            check_img.header.get_zooms()[:3],
        )

        print(
            "      mean:",
            float(check_data[check_mask].mean()),
        )

        print(
            "      std:",
            float(check_data[check_mask].std()),
        )

        print(
            "      saved:",
            final_path,
        )

        # -------------------------------------------------
        # AGGIORNA CATALOGO
        # -------------------------------------------------

        new_row = row.copy()

        new_row["OriginalFilePath"] = str(
            input_path.resolve()
        )

        new_row["FilePath"] = str(
            final_path.resolve()
        )

        records.append(new_row)

    except Exception as exc:

        print(
            f"FAILED: {scan_id}"
        )

        print(
            f"Reason: {exc}"
        )

        failed.append(
            (scan_id, str(exc))
        )


# ---------------------------------------------------------
# SALVATAGGIO CATALOGO FINALE
# ---------------------------------------------------------

processed = pd.DataFrame(records)

processed.to_csv(
    OUTPUT_CSV,
    index=False,
)

print("\n" + "=" * 75)
print("PREPROCESSING FINISHED")
print("=" * 75)

print(
    f"Successful: {len(processed)}"
)

print(
    f"Failed: {len(failed)}"
)

print(
    f"Output CSV: {OUTPUT_CSV}"
)

if failed:

    print("\nFailed scans:")

    for scan_id, reason in failed:
        print(
            f" - {scan_id}: {reason}"
        )
