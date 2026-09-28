"""Shared helpers for the evaluation scripts."""

import json
import os

import yaml
from vllm import LLM, SamplingParams

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data", "eval")
MODEL_CONFIG_PATH = os.path.join(REPO_ROOT, "configs", "model_config.yaml")

# Safety benchmarks (scored with Qwen3Guard + the sentence-level judge).
JAILBREAK_TASKS = ["strongreject", "safechain", "icl", "h-cot"]
# Over-refusal benchmarks (scored with the refusal judge).
OVER_REFUSAL_TASKS = ["or_bench", "fortress", "xstest"]
# Tasks evaluated under both the standard and the adversarial-prefilling setting.
PREFILL_TASKS = ["strongreject", "safechain"]


def get_model_config(model_family):
    with open(MODEL_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)[model_family]


def get_model(model_path, max_new_tokens, tp, gpu_memory_utilization, trust_remote_code,
              temperature=0.6, top_p=0.95, top_k=None):
    """Load a vLLM engine and its sampling parameters."""
    input_tokens_len = 10240 if "safeguard" in model_path else 2048
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        max_model_len=max_new_tokens + input_tokens_len,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=trust_remote_code,
    )
    if top_k is not None:
        # Gemma-4 needs its special tokens to locate the end of the thought channel.
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
            top_k=top_k, skip_special_tokens=False,
        )
    else:
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
            skip_special_tokens=True,
        )
    return llm, llm.get_tokenizer(), sampling_params


def split_reasoning_and_final(response, model_family):
    """Split a generation into (final_answer, reasoning)."""
    if model_family == "gpt-oss":
        end_reasoning = "<|end|><|start|>assistant<|channel|>final<|message|>"
    elif "Gemma" in model_family:
        end_reasoning = "<channel|>"
    else:
        end_reasoning = "</think>"

    idx_end = response.rfind(end_reasoning)
    if idx_end == -1:
        print("Warning: end-of-reasoning tag not found; the model may not have finished reasoning.")
        return response, response

    reasoning = response[:idx_end].strip()
    final_answer = response[idx_end + len(end_reasoning):].strip()
    if idx_end == 0:
        print("Warning: the model produced no reasoning after the prefill.")
        return final_answer, final_answer
    return final_answer, reasoning


def parse_safeguard_output(text):
    """Extract the final label from a gpt-oss-safeguard generation."""
    for marker in ("assistantfinal", ".final"):
        idx = text.rfind(marker)
        if idx != -1:
            return text[idx + len(marker):].strip()
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def result_path(output_dir, task, prefill):
    return os.path.join(output_dir, f"{task}_{'prefill' if prefill else 'no_prefill'}.csv")


def update_summary(output_dir, key, metrics):
    """Merge `metrics` into <output_dir>/eval_summary.json under `key`."""
    path = os.path.join(output_dir, "eval_summary.json")
    summary = {}
    if os.path.exists(path):
        with open(path, "r") as f:
            summary = json.load(f)
    summary.setdefault(key, {}).update(metrics)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
