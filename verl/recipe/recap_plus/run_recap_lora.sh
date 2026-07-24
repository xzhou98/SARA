# bash recipe/recap_plus/run_recap_lora.sh
#!/usr/bin/env bash
set -xeuo pipefail


project_name='Deceptive_Alignment'
exp_name='Our-DeepSeek-R1-0528-Qwen3-8B-LoRA'

# ---------- Models ----------
MODEL_PATH="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"

# ---------- Data ----------
TRAIN_FILE="/home/xzhou19/expanse/Deceptive Alignment LRM/train_2k_R1-0528_0%"
TEST_FILE="${TRAIN_FILE}"



SAFEGUARD_HOST="127.0.0.1"
SAFEGUARD_PORT=8002
SAFEGUARD_BASE_URL="http://${SAFEGUARD_HOST}:${SAFEGUARD_PORT}"

REWARDMODEL_PORT=8001
REWARDMODEL_BASE_URL="http://${SAFEGUARD_HOST}:${REWARDMODEL_PORT}"


# ---------- LoRA ----------
lora_rank=8
lora_alpha=16
target_modules="all-linear"

# ---------- Algorithm ----------
adv_estimator=grpo
clip_ratio_low=0.2
clip_ratio_high=0.28
loss_agg_mode="token-mean"
temperature=1.0
top_p=1.0
top_k=-1
learning_rate=3e-5

# ---------- Training hyperparameters ----------
max_prompt_length=$((1024 * 1))
max_response_length=$((2048 * 1))
n_resp_per_prompt=4
total_epochs=1
train_prompt_bsz=32
gen_prompt_bsz=${train_prompt_bsz}
train_prompt_mini_bsz=4

# ---------- Checkpoints ----------
CKPTS_DIR="/home/xzhou19/expanse/Deceptive Alignment LRM/SARA/verl/ckpts/${project_name}/${exp_name}_lr-${learning_rate}_epoch-${total_epochs}"

# ---------- Conda envs ----------
TRAIN_CONDA_ENV="verl"        # needs vLLM >= 0.9 for verl's run_headless API

wait_for_server() {
    local host=$1
    local port=$2
    local name=$3
    local max_wait=600
    local elapsed=0
    echo "[RECAP] Waiting for ${name} at ${host}:${port}..."
    while ! curl -s "http://${host}:${port}/health" > /dev/null 2>&1; do
        sleep 10
        elapsed=$((elapsed + 10))
        if [ ${elapsed} -ge ${max_wait} ]; then
            echo "[RECAP] ERROR: ${name} did not start within ${max_wait}s"
            exit 1
        fi
    done
    echo "[RECAP] ${name} is ready (took ~${elapsed}s)"
}

echo "[RECAP] Using remote safeguard server at ${SAFEGUARD_BASE_URL}"
wait_for_server "${SAFEGUARD_HOST}" "${SAFEGUARD_PORT}" "Safeguard RM"

export SAFEGUARD_BASE_URL="http://127.0.0.1:8002"
export SAFEGUARD_HOST="${SAFEGUARD_HOST}"
export SAFEGUARD_PORT="${SAFEGUARD_PORT}"

export REWARDMODEL_HOST="${SAFEGUARD_HOST}"
export REWARDMODEL_PORT="${REWARDMODEL_PORT}"
export REWARDMODEL_BASE_URL="http://127.0.0.1:8001"


set +e
CUDA_VISIBLE_DEVICES=0,1 conda run -n "${TRAIN_CONDA_ENV}" \
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
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$((train_prompt_mini_bsz/1)) \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$((train_prompt_mini_bsz/1)) \
    actor_rollout_ref.nccl_timeout=1200 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.lora_rank=${lora_rank} \
    actor_rollout_ref.model.lora_alpha=${lora_alpha} \
    actor_rollout_ref.model.target_modules=${target_modules} \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${learning_rate} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$((train_prompt_mini_bsz/1)) \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=2048 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
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
    reward.num_workers=32 \
    reward.reward_model.n_gpus_per_node=1 \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.test_freq=0 \
    trainer.save_freq=10 \
    trainer.total_epochs=${total_epochs} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.use_legacy_worker_impl=enable

TRAIN_EXIT=$?
set -e

# ========================== 4. Cleanup ========================================
echo "[RECAP] Training finished (exit code: ${TRAIN_EXIT})."
echo "[RECAP] Remote safeguard server at ${SAFEGUARD_BASE_URL} was not managed by this script."
echo "[RECAP] Done."
exit ${TRAIN_EXIT}
