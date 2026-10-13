#!/bin/bash

set -e

PROJECT="/mnt/data/Palmisano/Medical_AI"

FLOWLET="$PROJECT/external/FlowLet"

CSV="/mnt/data/Palmisano/datasets/main_dataset_catalog_preprocessed.csv"

RESULTS="$PROJECT/results/flowlet"

RUN_NAME="${1:-FlowLet_ADNI_33}"

EPOCHS="${2:-10}"

BATCH_SIZE="${3:-1}"


source "$PROJECT/.venv-flowlet/bin/activate"

cd "$FLOWLET"


echo "=============================================="
echo "FlowLet training"
echo "=============================================="
echo "CSV:        $CSV"
echo "Run:        $RUN_NAME"
echo "Epochs:     $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "=============================================="


CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=. \
python -u scripts/train.py \
  --metadata_csv "$CSV" \
  --condition_vars Age \
  --csv_filter_col Condition \
  --csv_filter_value CN \
  --epochs "$EPOCHS" \
  --lr 3e-6 \
  --batch_size "$BATCH_SIZE" \
  --val_split 0.2 \
  --flow_type rectified \
  --model_input_size 112 112 112 \
  --save_size 91 109 91 \
  --unet_model_channels 128 \
  --unet_channel_mult "1,2,4,8" \
  --unet_attention_res "4,8" \
  --use_checkpointing \
  --use_xformers \
  --num_workers 2 \
  --checkpoint_dir "$RESULTS" \
  --run_name "$RUN_NAME" \
  --no-wandb \
  --no-generate_after_train \
  --seed 42 \
  --device cuda
