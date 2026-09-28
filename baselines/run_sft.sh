#!/usr/bin/env bash
# Train the three off-policy SFT baselines with LoRA (r=8, alpha=16) on 2 GPUs.
# Effective batch size = 2 GPUs x 4 per device x 4 accumulation steps = 32.
# Usage: bash baselines/run_sft.sh <model_family>   (e.g. R1-0528 or R1-Qwen-14B)
set -euo pipefail

MODEL_FAMILY=${1:-R1-Qwen-14B}
cd "$(dirname "$0")"

launch() {
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} accelerate launch \
        --config_file ../configs/deepspeed_zero2.yaml finetune.py \
        --model_family "${MODEL_FAMILY}" \
        --gradient_accumulation_steps 4 --per_device_train_batch_size 4 --lora_r 8 "$@"
}

launch --loss_type SFT      --data_path ../data/train/safechain_sft_2k   # SafeChain
launch --loss_type SFT      --data_path ../data/train/star1_1k           # STAR-1
launch --loss_type safepath --data_path ../data/train/sara_train_2k      # SafePath
