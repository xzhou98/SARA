"""Supervised fine-tuning for the off-policy baselines.

  SafeChain : --loss_type SFT      --data_path data/train/safechain_sft_2k
  STAR-1    : --loss_type SFT      --data_path data/train/star1_1k
  SafePath  : --loss_type safepath --data_path data/train/sara_train_2k
"""

import argparse
import logging
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from transformers.training_args import TrainingArguments

from data_module import REPO_ROOT, SFTDataset, collate_fn, get_model_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]


class SFTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        return (outputs.loss, outputs) if return_outputs else outputs.loss


def parse_args():
    parser = argparse.ArgumentParser(description="SFT baselines (SafeChain / STAR-1 / SafePath)")
    parser.add_argument("--model_family", type=str, default="R1-Qwen-14B", help="Key in configs/model_config.yaml")
    parser.add_argument("--data_path", type=str, default=os.path.join(REPO_ROOT, "data/train/safechain_sft_2k"))
    parser.add_argument("--loss_type", type=str, default="SFT", choices=["SFT", "safepath"])
    parser.add_argument("--save_dir", type=str, default=None, help="Defaults to saves/<model_family>/<run name>")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=3072, help="Max sequence length")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_r", type=int, default=0, help="LoRA rank (0 = full fine-tuning)")
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    return parser.parse_args()


def set_random_seed(seed):
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    set_random_seed(args.seed)

    if args.save_dir is None:
        run_name = (f"{args.loss_type}_{os.path.basename(os.path.normpath(args.data_path))}"
                    f"_lr-{args.learning_rate}_epochs-{args.num_epochs}")
        if args.lora_r != 0:
            run_name += "_lora"
        args.save_dir = os.path.join(REPO_ROOT, "saves", args.model_family, run_name)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    model_config = get_model_config(args.model_family)
    model_id = model_config["hf_key"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    train_dataset = SFTDataset(args.data_path, tokenizer, args.model_family, args.max_length, args.loss_type)

    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, trust_remote_code=True)
    if args.lora_r != 0:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        model = model.to(torch.device(f"cuda:{local_rank}"))

    training_args = TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        warmup_steps=10,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant_with_warmup",
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        optim="adamw_torch",
        bf16=True,
        bf16_full_eval=True,
        logging_steps=1,
        logging_dir=f"{args.save_dir}/logs",
        output_dir=args.save_dir,
        save_strategy="epoch",
        save_only_model=True,
        save_total_limit=1,
        eval_strategy="no",
        ddp_find_unused_parameters=False,
    )

    if model_config["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()
        if args.lora_r != 0:
            model.enable_input_require_grads()

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )
    model.config.use_cache = False

    logger.info(f"Start training: {datetime.now():%Y-%m-%d %H:%M:%S}")
    trainer.train()
    logger.info(f"Finish training: {datetime.now():%Y-%m-%d %H:%M:%S}")
    tokenizer.save_pretrained(args.save_dir)


if __name__ == "__main__":
    main()
