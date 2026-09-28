#!/usr/bin/env bash
# Full evaluation pipeline for one model.
# Usage: bash evaluation/run_eval.sh <model_path> <model_family> [output_dir]
#   model_family is a key in configs/model_config.yaml (R1-0528, R1-Qwen-14B, R1-Llama-8B, Gemma-4, gpt-oss).
set -euo pipefail

MODEL_PATH=$1
MODEL_FAMILY=$2
OUTPUT_DIR=${3:-results/${MODEL_FAMILY}/$(basename "${MODEL_PATH}")}
TP=${TP:-1}

# Resolve local paths before changing directory (HF model ids are left as-is).
if [ -e "${MODEL_PATH}" ]; then MODEL_PATH=$(realpath "${MODEL_PATH}"); fi
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR=$(realpath "${OUTPUT_DIR}")

cd "$(dirname "$0")"

# 1. Generate reasoning traces + final answers on all benchmarks.
python generate.py --model_path "${MODEL_PATH}" --model_family "${MODEL_FAMILY}" \
    --output_dir "${OUTPUT_DIR}" --tp "${TP}" --gpu_memory_utilization 0.9 --trust_remote_code

# 2. Qwen3Guard: full-reasoning safety (r_safe) and final-answer safety (SS).
python judge_guard.py --output_dir "${OUTPUT_DIR}" --tp "${TP}"

# 3. gpt-oss-safeguard sentence-level judge: SAR and DSAR.
python judge_safety_aware.py --output_dir "${OUTPUT_DIR}" --tp "${TP}"

# 4. gpt-oss-safeguard refusal judge: Helpfulness Score on OR-Bench / Fortress / XSTest.
python judge_helpfulness.py --output_dir "${OUTPUT_DIR}" --tp "${TP}"

echo "Metrics written to ${OUTPUT_DIR}/eval_summary.json"
