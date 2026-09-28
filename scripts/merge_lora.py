"""Merge a LoRA adapter into its base model so the result can be served by vLLM.

Examples:
  # verl (SARA / RECAP) checkpoint
  python scripts/merge_lora.py --base_model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
      --adapter <ckpt_dir>/global_step_62/actor/lora_adapter --output_dir <ckpt_dir>/merged

  # SFT baseline checkpoint
  python scripts/merge_lora.py --base_model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
      --adapter saves/R1-Qwen-14B/<run>/checkpoint-<step> --output_dir saves/R1-Qwen-14B/<run>/merged
"""

import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base_model", required=True, help="HF id of the base model")
    parser.add_argument("--adapter", required=True, help="Folder with adapter_config.json + adapter weights")
    parser.add_argument("--output_dir", required=True, help="Where to save the merged model")
    args = parser.parse_args()

    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype="bfloat16")
    model = PeftModel.from_pretrained(base_model, args.adapter).merge_and_unload()
    model.save_pretrained(args.output_dir)
    AutoTokenizer.from_pretrained(args.base_model).save_pretrained(args.output_dir)
    print(f"Merged model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
