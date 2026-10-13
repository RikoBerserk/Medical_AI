import argparse
import inspect
import time
from datetime import datetime
from pathlib import Path

import torch

from scripts.diff_model_infer import (
    load_models,
    prepare_tensors,
    run_inference,
    save_image,
    set_random_seed,
)

from scripts.diff_model_setting import (
    initialize_distributed,
    load_config,
    setup_logging,
)


@torch.inference_mode()
def main():

    # ============================================================
    # ARGOMENTI
    # ============================================================

    parser = argparse.ArgumentParser(
        description="Benchmark MAISI generating multiple MRI volumes "
                    "while loading the models only once."
    )

    parser.add_argument(
        "--env_config",
        required=True,
    )

    parser.add_argument(
        "--model_config",
        required=True,
    )

    parser.add_argument(
        "--model_def",
        required=True,
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--base_seed",
        type=int,
        default=1234,
    )

    parser.add_argument(
        "--output_dir",
        default=(
            "/mnt/data/Palmisano/Medical_AI/results/"
            "maisi_rflow_mr_brain/benchmark_100_reuse_model"
        ),
    )

    args_cli = parser.parse_args()

    # ============================================================
    # CONFIGURAZIONE MAISI
    # ============================================================

    args = load_config(
        args_cli.env_config,
        args_cli.model_config,
        args_cli.model_def,
    )

    local_rank, world_size, device = initialize_distributed(1)

    logger = setup_logging("benchmark_maisi_100")

    logger.info(
        f"Using {device} of {world_size}"
    )

    # ============================================================
    # PARAMETRI INFERENCE
    # ============================================================

    output_size = tuple(
        args.diffusion_unet_inference["dim"]
    )

    out_spacing = tuple(
        args.diffusion_unet_inference["spacing"]
    )

    modality = args.diffusion_unet_inference["modality"]

    args.cfg_guidance_scale = (
        args.diffusion_unet_inference["cfg_guidance_scale"]
    )

    output_prefix = args.output_prefix

    output_dir = Path(args_cli.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    num_steps = args.diffusion_unet_inference[
        "num_inference_steps"
    ]

    logger.info(
        f"Output size: {output_size}"
    )

    logger.info(
        f"Spacing: {out_spacing}"
    )

    logger.info(
        f"Modality: {modality}"
    )

    logger.info(
        f"Inference steps: {num_steps}"
    )

    logger.info(
        f"Number of samples: {args_cli.num_samples}"
    )

    logger.info(
        f"Output directory: {output_dir}"
    )

    # ============================================================
    # CARICAMENTO MODELLI UNA SOLA VOLTA
    # ============================================================

    logger.info(
        "Loading autoencoder and diffusion U-Net ONCE..."
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    load_start = time.perf_counter()

    # La repo ha avuto versioni differenti della funzione.
    # Controlliamo automaticamente se accetta low_vram.

    load_signature = inspect.signature(load_models)

    if "low_vram" in load_signature.parameters:

        autoencoder, unet, scale_factor = load_models(
            args,
            device,
            logger,
            low_vram=False,
        )

    else:

        autoencoder, unet, scale_factor = load_models(
            args,
            device,
            logger,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    model_loading_time = (
        time.perf_counter() - load_start
    )

    logger.info(
        f"Models loaded in {model_loading_time:.3f} s"
    )

    # ============================================================
    # PARAMETRI LATENT SPACE
    # ============================================================

    num_downsample_level = max(
        1,
        (
            len(
                args.diffusion_unet_def["num_channels"]
            )
            if isinstance(
                args.diffusion_unet_def["num_channels"],
                list,
            )
            else len(
                args.diffusion_unet_def[
                    "attention_levels"
                ]
            )
        ),
    )

    divisor = 2 ** (
        num_downsample_level - 2
    )

    logger.info(
        f"num_downsample_level={num_downsample_level}, "
        f"divisor={divisor}"
    )

    (
        top_region_index_tensor,
        bottom_region_index_tensor,
        spacing_tensor,
        modality_tensor,
    ) = prepare_tensors(
        args,
        device,
    )

    # ============================================================
    # BENCHMARK
    # ============================================================

    inference_times = []
    complete_sample_times = []

    logger.info("")
    logger.info(
        "Starting generation benchmark..."
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    benchmark_start = time.perf_counter()

    run_signature = inspect.signature(
        run_inference
    )

    for i in range(args_cli.num_samples):

        seed = args_cli.base_seed + i

        random_seed = set_random_seed(seed)

        logger.info("")
        logger.info(
            "=" * 60
        )

        logger.info(
            f"Generating MRI "
            f"{i + 1}/{args_cli.num_samples} "
            f"| seed={random_seed}"
        )

        logger.info(
            "=" * 60
        )

        # --------------------------------------------------------
        # TEMPO INFERENCE
        # --------------------------------------------------------

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        sample_total_start = time.perf_counter()
        inference_start = time.perf_counter()

        common_args = [
            args,
            device,
            autoencoder,
            unet,
            scale_factor,
            top_region_index_tensor,
            bottom_region_index_tensor,
            spacing_tensor,
            modality_tensor,
            output_size,
            divisor,
            logger,
        ]

        # Anche run_inference ha versioni con/senza low_vram

        if "low_vram" in run_signature.parameters:

            data = run_inference(
                *common_args,
                low_vram=False,
            )

        else:

            data = run_inference(
                *common_args
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        inference_time = (
            time.perf_counter()
            - inference_start
        )

        inference_times.append(
            inference_time
        )

        # --------------------------------------------------------
        # SALVATAGGIO NIFTI
        # --------------------------------------------------------

        timestamp = datetime.now().strftime(
            "%Y%m%d%H%M%S"
        )

        output_path = (
            output_dir
            / (
                f"{output_prefix}"
                f"_seed{random_seed}"
                f"_size{output_size[0]}x"
                f"{output_size[1]}x"
                f"{output_size[2]}"
                f"_spacing{out_spacing[0]:.2f}x"
                f"{out_spacing[1]:.2f}x"
                f"{out_spacing[2]:.2f}"
                f"_modality{modality}"
                f"_{timestamp}.nii.gz"
            )
        )

        save_image(
            data,
            output_size,
            out_spacing,
            str(output_path),
            logger,
        )

        complete_sample_time = (
            time.perf_counter()
            - sample_total_start
        )

        complete_sample_times.append(
            complete_sample_time
        )

        logger.info(
            f"MRI {i + 1}/{args_cli.num_samples} completed"
        )

        logger.info(
            f"Inference time: "
            f"{inference_time:.3f} s"
        )

        logger.info(
            f"Inference + save: "
            f"{complete_sample_time:.3f} s"
        )

    # ============================================================
    # RISULTATI
    # ============================================================

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    benchmark_time = (
        time.perf_counter()
        - benchmark_start
    )

    total_with_loading = (
        model_loading_time
        + benchmark_time
    )

    mean_inference = (
        sum(inference_times)
        / len(inference_times)
    )

    mean_complete_sample = (
        sum(complete_sample_times)
        / len(complete_sample_times)
    )

    throughput_sec = (
        args_cli.num_samples
        / benchmark_time
    )

    throughput_min = (
        throughput_sec * 60
    )

    # ============================================================
    # STAMPA FINALE
    # ============================================================

    print()
    print("=" * 70)
    print("MAISI MULTI-SAMPLE BENCHMARK RESULTS")
    print("=" * 70)

    print(
        f"Images generated:              "
        f"{args_cli.num_samples}"
    )

    print(
        f"Inference steps / MRI:         "
        f"{num_steps}"
    )

    print(
        f"Model loading time:            "
        f"{model_loading_time:.3f} s"
    )

    print(
        f"Generation benchmark time:     "
        f"{benchmark_time:.3f} s"
    )

    print(
        f"Total incl. model loading:      "
        f"{total_with_loading:.3f} s"
    )

    print(
        f"Mean inference time / MRI:      "
        f"{mean_inference:.3f} s"
    )

    print(
        f"Mean inference + save / MRI:    "
        f"{mean_complete_sample:.3f} s"
    )

    print(
        f"Fastest inference:              "
        f"{min(inference_times):.3f} s"
    )

    print(
        f"Slowest inference:              "
        f"{max(inference_times):.3f} s"
    )

    print(
        f"Throughput:                     "
        f"{throughput_min:.2f} MRI/min"
    )

    print(
        f"Output directory:               "
        f"{output_dir}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()

