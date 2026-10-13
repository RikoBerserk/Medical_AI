import json
import subprocess
from pathlib import Path

ROOT = Path("/mnt/data/Palmisano/Medical_AI")
MAISI = ROOT / "external" / "NV-Generate-CTMR"

base_config = MAISI / "configs" / "config_maisi_diff_model_rflow-mr-brain.json"
base_env = MAISI / "configs" / "environment_maisi_diff_model_rflow-mr-brain.json"
model_def = MAISI / "configs" / "config_network_rflow.json"

output_root = ROOT / "results" / "maisi_rflow_mr_brain" / "seeds"
temp_config = ROOT / "configs_maisi_seed_tmp.json"
temp_env = ROOT / "environment_maisi_seed_tmp.json"

output_root.mkdir(parents=True, exist_ok=True)

with open(base_config) as f:
    config = json.load(f)

with open(base_env) as f:
    env = json.load(f)

for seed in range(20):
    print(f"\n=== Generazione seed {seed} ===")

    seed_dir = output_root / f"seed_{seed:03d}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    config["diffusion_unet_inference"]["random_seed"] = seed

    # Manteniamo la baseline fissa
    config["diffusion_unet_inference"]["num_inference_steps"] = 30
    config["diffusion_unet_inference"]["modality"] = 9

    env["output_dir"] = str(seed_dir)

    with open(temp_config, "w") as f:
        json.dump(config, f, indent=4)

    with open(temp_env, "w") as f:
        json.dump(env, f, indent=4)

    subprocess.run(
        [
            "python",
            "-m",
            "scripts.diff_model_infer",
            "-t",
            str(model_def),
            "-e",
            str(temp_env),
            "-c",
            str(temp_config),
        ],
        cwd=MAISI,
        check=True,
    )

print("\nGenerazione completata.")
