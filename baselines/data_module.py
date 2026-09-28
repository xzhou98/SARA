"""SFT dataset for the off-policy baselines (SafeChain, STAR-1, SafePath)."""

import os
import random

import torch
import yaml
from datasets import load_from_disk
from torch.utils.data import Dataset

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_CONFIG_PATH = os.path.join(REPO_ROOT, "configs", "model_config.yaml")

SAFEPATH_PREFIX = "Let’s think about safety first. "


def get_model_config(model_family):
    with open(MODEL_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)[model_family]


def build_sft_example(tokenizer, max_length, question, answer, model_config):
    """Tokenize (question, answer) into fixed-length input_ids / labels / attention_mask.

    Prompt tokens are masked with -100 so the loss is computed on the response only.
    """
    message = []
    if model_config["system_prompt"]:
        message.append({"role": "system", "content": model_config["system_prompt"]})
    message.append({"role": "user", "content": question})
    question_prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    answer_prompt = f"{model_config['begin_think']}{model_config['begin_answer']}{answer}{model_config['end_tag']}"

    num_question_tokens = len(tokenizer.tokenize(
        question_prompt, add_special_tokens=True, max_length=max_length, truncation=True))
    tokenized = tokenizer(question_prompt + answer_prompt, add_special_tokens=True,
                          max_length=max_length, truncation=True)

    pad_length = max_length - len(tokenized.input_ids)
    input_ids = tokenized["input_ids"] + [tokenizer.pad_token_id] * pad_length
    attention_mask = tokenized["attention_mask"] + [0] * pad_length
    labels = tokenized["input_ids"] + [-100] * pad_length
    for i in range(num_question_tokens):
        labels[i] = -100

    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_mask),
    }


class SFTDataset(Dataset):
    """Reads `instruction` / `response` pairs from a HF dataset saved with `save_to_disk`.

    With loss_type="safepath", the response is prefixed with the SafePath "safety primer"
    for a random half of the samples.
    """

    def __init__(self, data_path, tokenizer, model_family, max_length, loss_type,
                 question_key="instruction", answer_key="response"):
        self.data = load_from_disk(data_path)
        self.model_config = get_model_config(model_family)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.loss_type = loss_type
        self.qk = question_key
        self.ak = answer_key
        self.rng = random.Random(42)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answer = self.data[idx][self.ak]
        if self.loss_type == "safepath" and self.rng.random() < 0.5:
            answer = SAFEPATH_PREFIX + answer
        return build_sft_example(self.tokenizer, self.max_length, question, answer, self.model_config)


def collate_fn(samples):
    return {key: torch.stack([s[key] for s in samples]) for key in samples[0]}
