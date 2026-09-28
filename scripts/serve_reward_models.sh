#!/usr/bin/env bash
# Serve the reward models for RL training as OpenAI-compatible vLLM servers.
#
# Default layout on a 4-GPU node (training itself uses GPUs 0,1):
#   GPU 2  deepseek-ai/DeepSeek-R1-Distill-Qwen-32B  port 8003  over-refusal judge (benign prompts)
#   GPU 3  ibm-granite/granite-guardian-3.3-8b       port 8002  safety guard -> R_cot, R_ans
#   GPU 3  openai/gpt-oss-safeguard-20b              port 8001  sentence-level judge -> R_SA   (SARA only)
#
# The two models on GPU 3 share it: each gets 45% of the GPU memory, and they are started one after
# the other so the second one sees the memory left by the first.
#
# Usage:
#   bash scripts/serve_reward_models.sh sara    # all three models (for SARA)
#   bash scripts/serve_reward_models.sh recap   # guard + refusal judge only (for RECAP)
# Logs are written to logs/reward_servers/. Stop the servers with Ctrl-C.
set -euo pipefail

MODE=${1:-sara}
REFUSAL_JUDGE_GPU=${REFUSAL_JUDGE_GPU:-2}
GUARD_GPU=${GUARD_GPU:-3}
SENTENCE_JUDGE_GPU=${SENTENCE_JUDGE_GPU:-3}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/logs/reward_servers"
mkdir -p "${LOG_DIR}"

trap 'kill $(jobs -p) 2>/dev/null' EXIT

wait_for_server() {
    local port=$1
    local name=$2
    echo "Waiting for ${name} on port ${port}..."
    until curl -s "http://127.0.0.1:${port}/health" > /dev/null 2>&1; do
        if ! kill -0 "$3" 2>/dev/null; then
            echo "ERROR: ${name} exited; see ${LOG_DIR}/${name}.log"
            exit 1
        fi
        sleep 10
    done
    echo "${name} is ready on port ${port}"
}

# Over-refusal judge: a 32B model in bf16 needs a full 80GB GPU. --reasoning-parser keeps the judge's
# <think> block out of the returned `content`, so only the final score is parsed.
CUDA_VISIBLE_DEVICES=${REFUSAL_JUDGE_GPU} vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --host 0.0.0.0 --port 8003 --gpu-memory-utilization 0.95 --max-model-len 20480 \
    --reasoning-parser deepseek_r1 > "${LOG_DIR}/refusal_judge.log" 2>&1 &
REFUSAL_PID=$!

# Safety guard.
CUDA_VISIBLE_DEVICES=${GUARD_GPU} vllm serve ibm-granite/granite-guardian-3.3-8b \
    --host 0.0.0.0 --port 8002 --gpu-memory-utilization 0.45 --max-model-len 8192 \
    > "${LOG_DIR}/guard.log" 2>&1 &
GUARD_PID=$!
wait_for_server 8002 guard ${GUARD_PID}

# Sentence-level safety-awareness judge (SARA only), started after the guard on the same GPU.
if [ "${MODE}" = "sara" ]; then
    CUDA_VISIBLE_DEVICES=${SENTENCE_JUDGE_GPU} vllm serve openai/gpt-oss-safeguard-20b \
        --host 0.0.0.0 --port 8001 --gpu-memory-utilization 0.45 --max-model-len 20480 \
        > "${LOG_DIR}/sentence_judge.log" 2>&1 &
    SENTENCE_PID=$!
    wait_for_server 8001 sentence_judge ${SENTENCE_PID}
fi

wait_for_server 8003 refusal_judge ${REFUSAL_PID}
echo "All reward servers are up (mode: ${MODE})."
wait
