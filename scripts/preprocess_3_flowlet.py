from pathlib import Path
import os
import subprocess

import pandas as pd
import nibabel as nib
import numpy as np


# =========================
# PATH PRINCIPALI
# =========================

CATALOG = Path(
    "/mnt/data/Palmisano/datasets/main_dataset_catalog.csv"
)

OUT_ROOT = Path(
    "/mnt/data/Palmisano/datasets/ADNI_flowlet_preprocessed/pilot3"
)

OUT_CSV = Path(
    "/mnt/data/Palmisano/datasets/main_dataset_catalog_pilot3.csv"
)

FSLDIR = Path(
    "/mnt/data/Palmisano/tools/fsl"
)

MNI = FSLDIR / "data/standard/MNI152_T1_2mm.nii.gz"

# Percorsi assoluti ai programmi FSL
FLIRT = FSLDIR / "share/fsl/bin/flirt"
BET = FSLDIR / "share/fsl/bin/bet"


# =========================
# CONFIGURAZIONE AMBIENTE FSL
# =========================

env = os.environ.copy()

env["FSLDIR"] = str(FSLDIR)
env["FSLOUTPUTTYPE"] = "NIFTI_GZ"

fsl_bin = str(FSLDIR / "share/fsl/bin")
fsl_bin2 = str(FSLDIR / "bin")

env["PATH"] = f"{fsl_bin}:{fsl_bin2}:{env.get('PATH', '')}"


# =========================
# CONTROLLI PRELIMINARI
# =========================

if not CATALOG.exists():
    raise FileNotFoundError(f"Catalogo non trovato: {CATALOG}")

if not MNI.exists():
    raise FileNotFoundError(f"Template MNI non trovato: {MNI}")

if not FLIRT.exists():
    raise FileNotFoundError(f"FLIRT non trovato: {FLIRT}")

if not BET.exists():
    raise FileNotFoundError(f"BET non trovato: {BET}")


# =========================
# SOGGETTI DEL PILOT
# =========================

subjects = [
    "136_S_0196",
    "136_S_0184",
    "136_S_0086",
]


# =========================
# CARICAMENTO CATALOGO
# =========================

df = pd.read_csv(CATALOG)

selected = df[df["SubjectID"].isin(subjects)].copy()

if len(selected) != len(subjects):
    found = set(selected["SubjectID"].astype(str))
    missing = [s for s in subjects if s not in found]

    raise ValueError(
        f"Alcuni soggetti non sono presenti nel catalogo: {missing}"
    )

records = []


# =========================
# PREPROCESSING
# =========================

for _, row in selected.iterrows():

    subject = str(row["SubjectID"])
    input_path = Path(row["FilePath"])

    if not input_path.exists():
        print(f"\nATTENZIONE: file non trovato per {subject}")
        print(input_path)
        continue

    out_dir = OUT_ROOT / subject
    out_dir.mkdir(parents=True, exist_ok=True)

    mni_path = out_dir / "mni.nii.gz"
    brain_path = out_dir / "brain.nii.gz"
    zscore_path = out_dir / "brain_zscore.nii.gz"
    mat_path = out_dir / "affine.mat"

    print("\n" + "=" * 70)
    print(f"Processing {subject}")
    print(f"Input: {input_path}")
    print("=" * 70)

    # -----------------------------------------
    # 1. Registrazione affine nello spazio MNI
    # -----------------------------------------

    print("\n[1/3] FLIRT: registrazione affine a MNI152 2mm...")

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
            str(mat_path),
            "-dof",
            "12",
            "-interp",
            "trilinear",
        ],
        check=True,
        env=env,
    )

    mni_img = nib.load(mni_path)

    print("FLIRT completato")
    print("Shape MNI:", mni_img.shape)
    print("Spacing MNI:", mni_img.header.get_zooms()[:3])

    if mni_img.shape != (91, 109, 91):
        raise ValueError(
            f"Shape MNI inattesa per {subject}: "
            f"{mni_img.shape}, atteso (91, 109, 91)"
        )

    # -----------------------------------------
    # 2. Skull stripping con BET
    # -----------------------------------------

    print("\n[2/3] BET: skull stripping...")

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
            f"BET non ha creato il file atteso: {brain_path}"
        )

    brain_img = nib.load(brain_path)

    print("BET completato")
    print("Shape brain:", brain_img.shape)

    # -----------------------------------------
    # 3. Normalizzazione z-score
    # -----------------------------------------

    print("\n[3/3] Normalizzazione z-score...")

    data = brain_img.get_fdata().astype(np.float32)

    # I voxel non-zero vengono considerati cervello
    mask = data != 0

    if mask.sum() == 0:
        raise ValueError(
            f"Maschera cerebrale vuota per {subject}"
        )

    mean = float(data[mask].mean())
    std = float(data[mask].std())

    if std == 0:
        raise ValueError(
            f"Deviazione standard nulla per {subject}"
        )

    normalized = np.zeros_like(data, dtype=np.float32)

    normalized[mask] = (
        data[mask] - mean
    ) / (std + 1e-8)

    header = brain_img.header.copy()
    header.set_data_dtype(np.float32)

    normalized_img = nib.Nifti1Image(
        normalized,
        brain_img.affine,
        header,
    )

    nib.save(
        normalized_img,
        zscore_path,
    )

    final_img = nib.load(zscore_path)
    final_data = final_img.get_fdata()

    final_mask = final_data != 0

    print("Z-score completato")
    print("Shape finale:", final_img.shape)
    print(
        "Spacing finale:",
        final_img.header.get_zooms()[:3],
    )

    if final_mask.any():
        print(
            "Brain mean:",
            final_data[final_mask].mean(),
        )

        print(
            "Brain std:",
            final_data[final_mask].std(),
        )

    print("Salvato:", zscore_path)

    # -----------------------------------------
    # Aggiornamento riga CSV
    # -----------------------------------------

    new_row = row.copy()

    new_row["FilePath"] = str(
        zscore_path.resolve()
    )

    records.append(new_row)


# =========================
# CREAZIONE CSV PILOT
# =========================

pilot = pd.DataFrame(records)

if len(pilot) == 0:
    raise RuntimeError(
        "Nessuna MRI è stata preprocessata correttamente."
    )

pilot.to_csv(
    OUT_CSV,
    index=False,
)

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETATO")
print("=" * 70)

print(
    f"MRI preprocessate: {len(pilot)}"
)

print(
    f"CSV creato: {OUT_CSV}"
)

print()

print(
    pilot[
        [
            "SubjectID",
            "Age",
            "Condition",
            "FilePath",
        ]
    ].to_string(index=False)
)
