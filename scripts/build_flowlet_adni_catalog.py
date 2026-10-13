from pathlib import Path
import pandas as pd

CSV_PATH = Path("/mnt/data/Palmisano/datasets/ADNI1_Complete_2Yr_3T_metadata.csv")
ADNI_ROOT = Path("/mnt/data/Palmisano/datasets/ADNI")
OUTPUT = Path("/mnt/data/Palmisano/datasets/main_dataset_catalog.csv")

df = pd.read_csv(CSV_PATH)

# Pulizia nomi colonne
df.columns = [c.strip() for c in df.columns]

print("Colonne trovate:")
print(df.columns.tolist())

# Filtro iniziale per FlowLet
filtered = df[
    (df["Group"].astype(str).str.strip() == "CN")
    & (df["Visit"].astype(str).str.strip().str.lower() == "bl")
    & (df["Modality"].astype(str).str.strip() == "MRI")
    & (df["Type"].astype(str).str.contains("Pre-processed", case=False, na=False))
    & (df["Description"].astype(str).str.contains("N3", case=False, na=False))
    & (df["Description"].astype(str).str.contains("Scaled", case=False, na=False))
].copy()

print("Righe dopo filtro:", len(filtered))

# Tutti i file NIfTI scaricati
nii_files = list(ADNI_ROOT.rglob("*.nii")) + list(ADNI_ROOT.rglob("*.nii.gz"))

records = []

for _, row in filtered.iterrows():
    subject = str(row["Subject"]).strip()
    image_id = str(row["Image Data ID"]).strip()

    # Match preferenziale tramite Image Data ID, es. I120416
    matches = [
        p for p in nii_files
        if image_id in p.name
    ]

    # fallback tramite subject
    if not matches:
        matches = [
            p for p in nii_files
            if subject in p.name
        ]

    if not matches:
        continue

    # Se ci sono più match, prendiamo il primo per ora
    path = matches[0]

    records.append({
        "FilePath": str(path.resolve()),
        "Subject": subject,
        "Age": row["Age"],
        "Condition": row["Group"],
        "Visit": row["Visit"],
        "ImageID": image_id,
        "Description": row["Description"],
    })

catalog = pd.DataFrame(records)

# Una sola immagine per soggetto
catalog = catalog.drop_duplicates(subset=["Subject"], keep="first")

catalog.to_csv(OUTPUT, index=False)

print("\nCatalogo creato:", OUTPUT)
print("Soggetti finali:", len(catalog))
print(catalog.head(10).to_string(index=False))
