import json
import csv
import time
import subprocess
import threading
from pathlib import Path

ROOT = Path("/mnt/data/Palmisano/Medical_AI")
MAISI = ROOT / "external" / "NV-Generate-CTMR"

BASE_CONFIG = MAISI / "configs" / "config_maisi_diff_model_rflow-mr-brain.json"
BASE_ENV = MAISI / "configs" / "environment_maisi_diff_model_rflow-mr-brain.json"
MODEL_DEF = MAISI / "configs" / "config_network_rflow.json"

OUTPUT_ROOT = ROOT / "results" / "maisi_rflow_mr_brain" / "steps_vram"
METRICS_DIR = ROOT / "results" / "maisi_rflow_mr_brain" / "metrics"

TEMP_CONFIG = ROOT / "tmp_maisi_steps_vram_config.json"
TEMP_ENV = ROOT / "tmp_maisi_steps_vram_env.json"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

with open(BASE_CONFIG, "r") as f:
    base_config = json.load(f)

with open(BASE_ENV, "r") as f:
    base_env = json.load(f)

SEED = 0
STEPS_LIST = [10, 20, 30, 40, 50]

results = []


def monitor_vram(stop_event, result_holder, gpu_index=0):
    max_used_mib = 0

    while not stop_event.is_set():
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={gpu_index}",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()

            used_mib = int(output.splitlines()[0])

            if used_mib > max_used_mib:
                max_used_mib = used_mib

        except Exception as e:
            print(f"Errore monitor VRAM: {e}")

        time.sleep(0.2)

    result_holder["peak_mib"] = max_used_mib


for steps in STEPS_LIST:
    print("\n" + "=" * 60)
    print(f"Benchmark steps = {steps}")
    print("=" * 60)

    output_dir = OUTPUT_ROOT / f"steps_{steps:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(json.dumps(base_config))
    env = json.loads(json.dumps(base_env))

    config["diffusion_unet_inference"]["random_seed"] = SEED
    config["diffusion_unet_inference"]["modality"] = 9
    config["diffusion_unet_inference"]["num_inference_steps"] = steps

    env["output_dir"] = str(output_dir)

    with open(TEMP_CONFIG, "w") as f:
        json.dump(config, f, indent=4)

    with open(TEMP_ENV, "w") as f:
        json.dump(env, f, indent=4)

    stop_event = threading.Event()
    vram_result = {}

    monitor_thread = threading.Thread(
        target=monitor_vram,
        args=(stop_event, vram_result, 0),
        daemon=True,
    )

    monitor_thread.start()

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

    stop_event.set()
    monitor_thread.join()

    peak_mib = vram_result.get("peak_mib", 0)
    peak_gib = peak_mib / 1024

    print(f"Steps {steps}: {elapsed:.2f} s")
    print(f"Peak VRAM: {peak_gib:.2f} GB")

    results.append(
        {
            "seed": SEED,
            "modality": 9,
            "inference_steps": steps,
            "time_seconds": round(elapsed, 3),
            "peak_vram_mib": peak_mib,
            "peak_vram_gib": round(peak_gib, 3),
        }
    )

csv_path = METRICS_DIR / "maisi_inference_steps_vram.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "seed",
            "modality",
            "inference_steps",
            "time_seconds",
            "peak_vram_mib",
            "peak_vram_gib",
        ],
    )
    writer.writeheader()
    writer.writerows(results)

print("\n" + "=" * 60)
print("RISULTATI FINALI")
print("=" * 60)

for r in results:
    print(
        f"{r['inference_steps']:>3} steps -> "
        f"{r['time_seconds']:.2f} s | "
        f"{r['peak_vram_gib']:.2f} GB"
    )

print(f"\nCSV salvato in: {csv_path}")
