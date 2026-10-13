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

OUTPUT_ROOT = ROOT / "results" / "maisi_rflow_mr_brain" / "timing"
METRICS_DIR = ROOT / "results" / "maisi_rflow_mr_brain" / "metrics"

TEMP_CONFIG = ROOT / "tmp_maisi_timing_config.json"
TEMP_ENV = ROOT / "tmp_maisi_timing_env.json"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

with open(BASE_CONFIG, "r") as f:
    base_config = json.load(f)

with open(BASE_ENV, "r") as f:
    base_env = json.load(f)

results = []

# Benchmark su 5 seed
seeds = range(5)

for seed in seeds:
    print(f"\n{'=' * 60}")
    print(f"Benchmark seed {seed}")
    print(f"{'=' * 60}")

    seed_dir = OUTPUT_ROOT / f"seed_{seed:03d}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Copia indipendente delle configurazioni
    config = json.loads(json.dumps(base_config))
    env = json.loads(json.dumps(base_env))

    # Parametri baseline
    config["diffusion_unet_inference"]["random_seed"] = seed
    config["diffusion_unet_inference"]["num_inference_steps"] = 30
    config["diffusion_unet_inference"]["modality"] = 9

    # Output dedicato al benchmark
    env["output_dir"] = str(seed_dir)

    with open(TEMP_CONFIG, "w") as f:
        json.dump(config, f, indent=4)

    with open(TEMP_ENV, "w") as f:
        json.dump(env, f, indent=4)

    # Sincronizzazione GPU prima della misura
    subprocess.run(
        [
            "python",
            "-c",
            "import torch; "
            "torch.cuda.synchronize() if torch.cuda.is_available() else None"
        ],
        check=True,
    )

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

    end = time.perf_counter()
    elapsed = end - start

    print(f"\nSeed {seed} completato in {elapsed:.2f} secondi")

    results.append(
        {
            "seed": seed,
            "modality": 9,
            "inference_steps": 30,
            "time_seconds": round(elapsed, 3),
        }
    )

# Salvataggio CSV
csv_path = METRICS_DIR / "maisi_generation_times.csv"

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

# Statistiche finali
times = [r["time_seconds"] for r in results]

mean_time = sum(times) / len(times)
min_time = min(times)
max_time = max(times)

print("\n" + "=" * 60)
print("RISULTATI BENCHMARK")
print("=" * 60)
print(f"Run eseguite: {len(times)}")
print(f"Tempo medio: {mean_time:.2f} s")
print(f"Tempo minimo: {min_time:.2f} s")
print(f"Tempo massimo: {max_time:.2f} s")
print(f"CSV salvato in: {csv_path}")
