import json
import csv
import time
import subprocess
from pathlib import Path

ROOT = Path("/mnt/data/Palmisano/Medical_AI")
MAISI = ROOT / "external" / "NV-Generate-CTMR"

BASE_CONFIG = MAISI / "configs" / "config_maisi_diff_model_rflow-mr-brain.json"
BASE_ENV = MAISI / "configs" / "environment_maisi_diff_model_rflow-mr-brain.json"
MODEL_DEF = MAISI / "configs" / "config_network_rflow.json"

OUTPUT_ROOT = ROOT / "results" / "maisi_rflow_mr_brain" / "steps"
METRICS_DIR = ROOT / "results" / "maisi_rflow_mr_brain" / "metrics"

TEMP_CONFIG = ROOT / "tmp_maisi_steps_config.json"
TEMP_ENV = ROOT / "tmp_maisi_steps_env.json"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

with open(BASE_CONFIG, "r") as f:
    base_config = json.load(f)

with open(BASE_ENV, "r") as f:
    base_env = json.load(f)

# Manteniamo fisso il seed
SEED = 0

# Varia solo il numero di inference steps
STEPS_LIST = [10, 20, 30, 40, 50]

results = []

for steps in STEPS_LIST:
    print("\n" + "=" * 60)
    print(f"Benchmark inference steps = {steps}")
    print("=" * 60)

    output_dir = OUTPUT_ROOT / f"steps_{steps:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copie indipendenti dei JSON
    config = json.loads(json.dumps(base_config))
    env = json.loads(json.dumps(base_env))

    # Parametri mantenuti fissi
    config["diffusion_unet_inference"]["random_seed"] = SEED
    config["diffusion_unet_inference"]["modality"] = 9

    # Parametro variabile
    config["diffusion_unet_inference"]["num_inference_steps"] = steps

    # Output separato
    env["output_dir"] = str(output_dir)

    with open(TEMP_CONFIG, "w") as f:
        json.dump(config, f, indent=4)

    with open(TEMP_ENV, "w") as f:
        json.dump(env, f, indent=4)

    start = time.perf_counter()

    subprocess.run(
        [
            "python",
            "-m",
            "scripts.diff_model_infer",
            "-t",
            str(MODEL_DEF),
            "-e",
            str(TEMP_ENV),
            "-c",
            str(TEMP_CONFIG),
        ],
        cwd=MAISI,
        check=True,
    )

    elapsed = time.perf_counter() - start

    print(f"Steps {steps}: {elapsed:.2f} secondi")

    results.append(
        {
            "seed": SEED,
            "modality": 9,
            "inference_steps": steps,
            "time_seconds": round(elapsed, 3),
        }
    )

# Salvataggio CSV
csv_path = METRICS_DIR / "maisi_inference_steps.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "seed",
            "modality",
            "inference_steps",
            "time_seconds",
        ],
    )
    writer.writeheader()
    writer.writerows(results)

print("\n" + "=" * 60)
print("TEST INFERENCE STEPS COMPLETATO")
print("=" * 60)

for r in results:
    print(
        f"{r['inference_steps']:>3} steps -> "
        f"{r['time_seconds']:.2f} s"
    )

print(f"\nCSV salvato in: {csv_path}")
