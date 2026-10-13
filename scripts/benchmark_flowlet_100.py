import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from flowlet.models import WaveletFlowMatching
from flowlet.utils import set_seed


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FlowLet with model loaded once."
    )

    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--condition_ranges_path", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--num_flow_steps", type=int, default=10)
    parser.add_argument("--age", type=float, default=78.0)
    parser.add_argument("--base_seed", type=int, default=1234)

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # LOAD CONFIG
    # ============================================================

    with open(args.config_path, "r") as f:
        model_config = json.load(f)

    with open(args.condition_ranges_path, "r") as f:
        condition_ranges = json.load(f)

    model_input_size = tuple(
        model_config["model_input_size"]
    )

    condition_vars = model_config.get(
        "condition_vars", []
    )

    attention_res = tuple(
        map(
            int,
            model_config["unet_attention_res"].split(",")
        )
    )

    channel_mult = tuple(
        map(
            int,
            model_config["unet_channel_mult"].split(",")
        )
    )

    condition_dims_dict = {
        var: 1 for var in condition_vars
    }

    # Stessa ricostruzione usata da scripts/generate.py
    unet_args = {
        "in_channels": 8,
        "model_channels": model_config.get(
            "unet_model_channels", 128
        ),
        "out_channels": 8,

        "num_res_blocks": model_config.get(
            "unet_num_res_blocks", 2
        ),

        "attention_resolutions": attention_res,

        "dropout": model_config.get(
            "unet_dropout", 0.1
        ),

        "channel_mult": channel_mult,

        "conv_resample": True,
        "dims": 3,

        "use_checkpoint": model_config.get(
            "use_checkpointing", False
        ),

        "num_heads": model_config.get(
            "unet_num_heads", 8
        ),

        "num_head_channels": model_config.get(
            "unet_num_head_channels", -1
        ),

        "use_scale_shift_norm": True,
        "resblock_updown": True,

        "condition_dims": condition_dims_dict,

        "condition_embedding_dim": model_config.get(
            "condition_embedding_dim", 512
        ),

        "use_xformers": model_config.get(
            "use_xformers", True
        ),

        "use_cross_attention": (
            not model_config.get(
                "unet_disable_cross_attn", False
            )
            and bool(condition_dims_dict)
        ),

        "norm_num_groups": model_config.get(
            "unet_norm_num_groups", 32
        ),

        "norm_eps": 1e-6,
    }

    # ============================================================
    # LOAD MODEL ONCE
    # ============================================================

    print("=" * 70)
    print("FLOWLET MULTI-SAMPLE BENCHMARK")
    print("=" * 70)

    print(f"Device:                 {device}")
    print(f"Samples:                {args.num_samples}")
    print(f"ODE steps:              {args.num_flow_steps}")
    print(f"Age:                    {args.age}")
    print(f"Model input size:       {model_input_size}")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    load_start = time.perf_counter()

    checkpoint = torch.load(
        args.checkpoint_path,
        map_location=device
    )

    model = WaveletFlowMatching(
        u_net_args=unet_args,
        num_flow_steps=args.num_flow_steps,
    ).to(device)

    state_dict = checkpoint.get(
        "flow_net_state_dict",
        checkpoint.get(
            "model_state_dict",
            checkpoint
        )
    )

    if not state_dict:
        raise KeyError(
            "No model state dictionary found in checkpoint."
        )

    # Gestisce eventuali checkpoint torch.compile
    saved_compiled = any(
        k.startswith("_orig_mod.")
        for k in state_dict.keys()
    )

    current_compiled = hasattr(
        model.flow_net,
        "_orig_mod"
    )

    if saved_compiled and not current_compiled:
        state_dict = {
            k.replace("_orig_mod.", ""): v
            for k, v in state_dict.items()
        }

    model.flow_net.load_state_dict(
        state_dict
    )

    model.eval()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    model_loading_time = (
        time.perf_counter() - load_start
    )

    print(
        f"Model loading time:     "
        f"{model_loading_time:.3f} s"
    )

    # ============================================================
    # CONDITION
    # ============================================================

    if "Age" not in condition_dims_dict:
        raise ValueError(
            "The trained model is not conditioned on Age."
        )

    age_range = condition_ranges["Age"]

    age_min = float(age_range["min"])
    age_max = float(age_range["max"])

    if age_max > age_min:
        age_norm = (
            args.age - age_min
        ) / (
            age_max - age_min
        )
    else:
        age_norm = 0.5

    age_norm = float(
        np.clip(age_norm, 0.0, 1.0)
    )

    conditions = {
        "Age": torch.tensor(
            [[age_norm]],
            dtype=torch.float32,
            device=device,
        )
    }

    print(
        f"Normalized Age:         {age_norm:.6f}"
    )

    # ============================================================
    # CROP 112^3 -> 91 x 109 x 91
    # ============================================================

    save_size = (
        91,
        109,
        91,
    )

    crop_indices = []

    for target, source in zip(
        save_size,
        model_input_size
    ):
        diff = source - target

        if diff < 0:
            raise ValueError(
                "Save size larger than model output size."
            )

        start = diff // 2
        end = start + target

        crop_indices.append(
            (start, end)
        )

    (
        (d_start, d_end),
        (h_start, h_end),
        (w_start, w_end),
    ) = crop_indices

    # ============================================================
    # GENERATION LOOP
    # ============================================================

    sampling_times = []
    complete_times = []

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    generation_start = time.perf_counter()

    for i in range(args.num_samples):

        seed = args.base_seed + i

        # Nuovo rumore iniziale per ogni MRI
        set_seed(seed)

        print(
            f"\nGenerating "
            f"{i + 1}/{args.num_samples} "
            f"| seed={seed}"
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        complete_start = time.perf_counter()
        sampling_start = time.perf_counter()

        with (
            torch.no_grad(),
            torch.amp.autocast(
                "cuda",
                enabled=torch.cuda.is_available()
            )
        ):

            synthetic = model.sample(
                1,
                model_input_size,
                conditions,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        sampling_time = (
            time.perf_counter()
            - sampling_start
        )

        sampling_times.append(
            sampling_time
        )

        # ========================================================
        # CENTER CROP
        # ========================================================

        synthetic = synthetic[
            :,
            :,
            d_start:d_end,
            h_start:h_end,
            w_start:w_end,
        ]

        # stessa conversione di generator.py
        synthetic = (
            synthetic + 1.0
        ) / 2.0

        synthetic = torch.clamp(
            synthetic,
            0.0,
            1.0
        )

        # ========================================================
        # SAVE NIFTI
        # ========================================================

        img_data = (
            synthetic[0, 0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        img_nifti = nib.Nifti1Image(
            img_data,
            affine=np.eye(4)
        )

        output_path = (
            output_dir
            / (
                f"FlowLet_"
                f"seed{seed}_"
                f"Age{args.age:.2f}_"
                f"steps{args.num_flow_steps}.nii.gz"
            )
        )

        nib.save(
            img_nifti,
            output_path
        )

        complete_time = (
            time.perf_counter()
            - complete_start
        )

        complete_times.append(
            complete_time
        )

        print(
            f"Sampling:               "
            f"{sampling_time:.3f} s"
        )

        print(
            f"Sampling + save:        "
            f"{complete_time:.3f} s"
        )

    # ============================================================
    # FINAL STATS
    # ============================================================

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    generation_total = (
        time.perf_counter()
        - generation_start
    )

    total_with_loading = (
        model_loading_time
        + generation_total
    )

    mean_sampling = (
        sum(sampling_times)
        / len(sampling_times)
    )

    mean_complete = (
        sum(complete_times)
        / len(complete_times)
    )

    throughput = (
        args.num_samples
        / generation_total
        * 60
    )

    print()
    print("=" * 70)
    print("FLOWLET BENCHMARK RESULTS")
    print("=" * 70)

    print(
        f"Images generated:              "
        f"{args.num_samples}"
    )

    print(
        f"ODE steps / MRI:               "
        f"{args.num_flow_steps}"
    )

    print(
        f"Model loading time:            "
        f"{model_loading_time:.3f} s"
    )

    print(
        f"Generation benchmark time:     "
        f"{generation_total:.3f} s"
    )

    print(
        f"Total incl. model loading:      "
        f"{total_with_loading:.3f} s"
    )

    print(
        f"Mean sampling time / MRI:       "
        f"{mean_sampling:.3f} s"
    )

    print(
        f"Mean sampling + save / MRI:     "
        f"{mean_complete:.3f} s"
    )

    print(
        f"Fastest sampling:               "
        f"{min(sampling_times):.3f} s"
    )

    print(
        f"Slowest sampling:               "
        f"{max(sampling_times):.3f} s"
    )

    print(
        f"Throughput:                     "
        f"{throughput:.2f} MRI/min"
    )

    print(
        f"Output directory:               "
        f"{output_dir}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
