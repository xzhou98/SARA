# bash recipe/recap/run_recap.sh
#!/usr/bin/env bash
set -xeuo pipefail

# ===========================================================================
# RECAP Safety Alignment Experiment
#
# GPU layout:
#   CUDA 0 — verl training (actor FSDP + internal rollout vLLM + ref model)
#   CUDA 1 — safeguard vLLM reward server (port 8001)
#
# The policy rollout vLLM is managed internally by verl's Ray workers on
# CUDA 0 — we do NOT start a separate server for it.
#
# The safeguard model (openai/gpt-oss-safeguard-20b) requires vLLM >= 0.9
# for mxfp4 quantization support, so we use the "temp" conda env.
# ===========================================================================

project_name='RECAP'
exp_name='RECAP-DeepSeek-R1-0528-Qwen3-8B'

# ---------- Models ----------
MODEL_PATH="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
SAFEGUARD_MODEL="openai/gpt-oss-safeguard-20b"
SAFEGUARD_CHAT_TEMPLATE="/data/XZhou/ICL_Jailbreak_LRM/New-Project/config/chat_template/gpt_oss_chat_template.jinja"

# ---------- Data ----------
TRAIN_FILE="/scr/xzhou/safety_alignment_CL/data/merged/train_2k_R1-0528"
TEST_FILE="${TRAIN_FILE}"

# ---------- Checkpoints ----------
CKPTS_DIR="/scr/xzhou/verl/ckpts/${project_name}/${exp_name}"

# ---------- Algorithm ----------
adv_estimator=grpo
clip_ratio_low=0.2
clip_ratio_high=0.28
loss_agg_mode="token-mean"
temperature=1.0
top_p=1.0
top_k=-1

# ---------- Training hyperparameters ----------
max_prompt_length=$((1024 * 1))
max_response_length=$((1024 * 2))
n_resp_per_prompt=4
total_epochs=2
train_prompt_bsz=16
gen_prompt_bsz=${train_prompt_bsz}
train_prompt_mini_bsz=2

# ---------- Ports ----------
SAFEGUARD_PORT=8001

# ---------- Conda envs ----------
SAFEGUARD_CONDA_ENV="temp"    # has vLLM >= 0.9 with mxfp4 support
TRAIN_CONDA_ENV="temp"        # needs vLLM >= 0.9 for verl's run_headless API

# ---------- Helper: wait for a vLLM server to become healthy ----------
wait_for_server() {
    local port=$1
    local name=$2
    local max_wait=600
    local elapsed=0
    echo "[RECAP] Waiting for ${name} on port ${port}..."
    while ! curl -s "http://127.0.0.1:${port}/health" > /dev/null 2>&1; do
        sleep 15
        elapsed=$((elapsed + 15))
        if [ ${elapsed} -ge ${max_wait} ]; then
            echo "[RECAP] ERROR: ${name} did not start within ${max_wait}s"
            exit 1
        fi
    done
    echo "[RECAP] ${name} is ready (took ~${elapsed}s)"
}

# ========================== 1. Safeguard vLLM server ==========================
echo "[RECAP] Starting safeguard vLLM server on CUDA 1, port ${SAFEGUARD_PORT}..."
echo "[RECAP] Using conda env '${SAFEGUARD_CONDA_ENV}' (vLLM with mxfp4 support)"

CUDA_VISIBLE_DEVICES=0 conda run -n "${SAFEGUARD_CONDA_ENV}" \
    vllm serve "${SAFEGUARD_MODEL}" \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.3 \
    --max-model-len 4096 \
    --chat-template "${SAFEGUARD_CHAT_TEMPLATE}" \
    --host 127.0.0.1 \
    --port ${SAFEGUARD_PORT} &
SAFEGUARD_PID=$!
echo "[RECAP] Safeguard server PID: ${SAFEGUARD_PID}"

# ========================== 2. Wait for safeguard server ======================
wait_for_server ${SAFEGUARD_PORT} "Safeguard RM"

# ========================== 3. Run RECAP training =============================
echo "[RECAP] Starting verl training on CUDA 0..."
echo "[RECAP] Using conda env '${TRAIN_CONDA_ENV}'"

set +e
CUDA_VISIBLE_DEVICES=1 conda run -n "${TRAIN_CONDA_ENV}" \
    python3 -m recipe.recap_plus.main_recap \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.instruction_key=instruction \
    data.prefill_key=prefill_prompts \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=False \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=3072 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    reward.num_workers=1 \
    reward.reward_model.n_gpus_per_node=1 \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=5 \
    trainer.save_freq=5 \
    trainer.total_epochs=${total_epochs} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto

TRAIN_EXIT=$?
set -e

# ========================== 4. Cleanup ========================================
echo "[RECAP] Training finished (exit code: ${TRAIN_EXIT}). Shutting down safeguard server..."
kill ${SAFEGUARD_PID} 2>/dev/null || true
wait ${SAFEGUARD_PID} 2>/dev/null || true
echo "[RECAP] Done."
exit ${TRAIN_EXIT}
