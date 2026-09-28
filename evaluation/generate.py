"""Step 1: generate reasoning traces and final answers with the model under evaluation.

For every task, writes <output_dir>/<task>_{no_prefill,prefill}.csv with columns
`prompt`, (`prefill`), `reasoning`, `final`. StrongReject and SafeChain are run in both
the standard and the adversarial-prefilling setting; the other tasks only in the standard one.
"""

import argparse
import json
import os
import random

import pandas as pd
from datasets import load_from_disk

from common import (DATA_DIR, JAILBREAK_TASKS, OVER_REFUSAL_TASKS, PREFILL_TASKS,
                    get_model, get_model_config, result_path, split_reasoning_and_final)

ICL_NUM_SHOTS = 16


def build_chat_prompts(questions, tokenizer, model_family, demo_set=None):
    chat_prompts = []
    for question in questions:
        messages = []
        if demo_set is not None:
            # 16-shot ICL attack: harmful questions paired with compliant answers.
            for demo in random.sample(demo_set, ICL_NUM_SHOTS):
                messages.append({"role": "user", "content": demo["prompt"]})
                messages.append({"role": "assistant", "content": demo["rejected"]})
        messages.append({"role": "user", "content": question})

        kwargs = {"enable_thinking": True} if "Gemma" in model_family else {}
        chat_prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs))
    return chat_prompts


def add_prefill(chat_prompt, prefill, model_family):
    """Open the model's reasoning channel and force it to start with `prefill`."""
    if "R1" in model_family:
        return chat_prompt + "<think>\n" + prefill
    if "gpt-oss" in model_family:
        return chat_prompt + "<|channel|>analysis<|message|>" + prefill
    if "Gemma" in model_family:
        return chat_prompt + "<|channel>thought\n" + prefill
    return chat_prompt + prefill


def run_generation(llm, tokenizer, sampling_params, dataset, task, prefill, model_family, save_path, demo_set=None):
    question_key = "instruction" if task == "safechain" else "prompt"
    questions = dataset[question_key]
    prompts = build_chat_prompts(questions, tokenizer, model_family, demo_set)
    if prefill:
        prefills = dataset["prefill_prompts"]
        prompts = [add_prefill(p, pre, model_family) for p, pre in zip(prompts, prefills)]

    outputs = llm.generate(prompts, sampling_params)
    finals, reasonings = [], []
    for output in outputs:
        text = output.outputs[0].text if output.outputs else ""
        final, reasoning = split_reasoning_and_final(text, model_family)
        finals.append(final)
        reasonings.append(reasoning)

    columns = {"prompt": questions}
    if prefill:
        columns["prefill"] = prefills
    columns.update({"reasoning": reasonings, "final": finals})
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pd.DataFrame(columns).to_csv(save_path, index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate responses for the safety / helpfulness benchmarks.")
    parser.add_argument("--model_path", required=True, help="HF model id or local checkpoint path")
    parser.add_argument("--model_family", required=True, help="Key in configs/model_config.yaml")
    parser.add_argument("--output_dir", required=True, help="Where the generation CSVs are written")
    parser.add_argument("--tasks", nargs="+", default=JAILBREAK_TASKS + OVER_REFUSAL_TASKS)
    parser.add_argument("--max_new_tokens", type=int, default=10240)
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling ICL demonstrations")
    return parser.parse_args()


def main(args):
    random.seed(args.seed)
    cfg = get_model_config(args.model_family)
    llm = tokenizer = sampling_params = None

    for task in args.tasks:
        dataset = load_from_disk(os.path.join(DATA_DIR, task))
        demo_set = None
        if task == "icl":
            with open(os.path.join(DATA_DIR, "icl_demos.json"), "r") as f:
                demo_set = json.load(f)

        for prefill in ([False, True] if task in PREFILL_TASKS else [False]):
            save_path = result_path(args.output_dir, task, prefill)
            if os.path.exists(save_path):
                print(f"[skip] {save_path} already exists")
                continue
            if llm is None:
                llm, tokenizer, sampling_params = get_model(
                    args.model_path, args.max_new_tokens, args.tp, args.gpu_memory_utilization,
                    args.trust_remote_code, cfg["temp"], cfg["top_p"], cfg.get("top_k"))
            print(f"Generating {task} (prefill={prefill})")
            run_generation(llm, tokenizer, sampling_params, dataset, task, prefill,
                           args.model_family, save_path, demo_set)


if __name__ == "__main__":
    main(parse_args())
