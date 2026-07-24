"""
RECAP dataset: extends RLHFDataset to pass prefill text alongside raw prompts.

The training data has fields:
  - instruction: the user query (string)
  - label: category label (string)
  - response: reference response (string, unused during RL)
  - prefill_prompts: the prefill continuation after <think> (string)

This dataset converts `instruction` into the standard chat message format
and passes `prefill_prompts` through non_tensor_batch so the agent loop
can prepend it to the model's generation.
"""

import logging
import traceback
from typing import Optional

import datasets
import numpy as np
import torch
from omegaconf import DictConfig

from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.tokenizer import normalize_token_ids

logger = logging.getLogger(__name__)

THINK_TAG = "<think>"


class RECAPDataset(RLHFDataset):
    """Dataset for RECAP that yields raw_prompt + prefill_prompts.

    Expects Arrow/Parquet/JSON data with at least:
      - instruction (str): the user message
      - prefill_prompts (str): text to inject after <think> as a generation prefix

    Optional fields:
      - label (str): forwarded as data_source for reward routing
      - reward_model.ground_truth: forwarded for reward computation
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        self.prefill_key = config.get("prefill_key", "prefill_prompts")
        self.instruction_key = config.get("instruction_key", "instruction")
        super().__init__(
            data_files=data_files,
            tokenizer=tokenizer,
            config=config,
            processor=processor,
            max_samples=max_samples,
        )

    def _read_files_and_tokenize(self):
        """Load arrow/parquet/json and optionally filter long prompts."""
        dataframes = []
        for parquet_file in self.data_files:
            if parquet_file.endswith(".parquet"):
                dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            elif parquet_file.endswith(".json") or parquet_file.endswith(".jsonl"):
                dataframe = datasets.load_dataset("json", data_files=parquet_file)["train"]
            else:
                dataframe = datasets.load_from_disk(parquet_file)
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        total = len(self.dataframe)
        print(f"RECAP dataset len: {total}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {total}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        """Filter prompts that exceed max_prompt_length.

        Builds the full prompt (chat template + <think> + prefill) to measure
        the actual token count the agent loop will produce.
        """
        if not self.filter_overlong_prompts:
            return dataframe

        tokenizer = self.tokenizer
        instruction_key = self.instruction_key
        prefill_key = self.prefill_key
        max_prompt_length = self.max_prompt_length
        apply_chat_template_kwargs = dict(self.apply_chat_template_kwargs)
        apply_chat_template_kwargs.pop("tokenize", None)
        apply_chat_template_kwargs.pop("return_dict", None)
        apply_chat_template_kwargs.pop("return_tensors", None)

        def doc2len(doc) -> int:
            try:
                messages = [{"role": "user", "content": doc[instruction_key]}]
                prompt_ids = normalize_token_ids(
                    tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True, **apply_chat_template_kwargs
                    )
                )
                prefill_text = doc.get(prefill_key, "")
                prefill_str = THINK_TAG + prefill_text if prefill_text else THINK_TAG
                prefill_ids = normalize_token_ids(tokenizer.encode(prefill_str, add_special_tokens=False))
                return len(prompt_ids) + len(prefill_ids)
            except Exception:
                print("Error processing one of the samples, skipping...")
                traceback.print_exc()
                return max_prompt_length + 1

        dataframe = dataframe.filter(
            lambda doc: doc2len(doc) <= max_prompt_length,
            num_proc=self.num_workers,
            desc=f"Filtering prompts longer than {max_prompt_length} tokens",
        )
        print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]

        instruction = row_dict.get(self.instruction_key, "")
        raw_prompt = [{"role": "user", "content": instruction}]
        row_dict["raw_prompt"] = raw_prompt

        row_dict["prefill_text"] = row_dict.get(self.prefill_key, "")

        if "data_source" not in row_dict:
            row_dict["data_source"] = row_dict.get("label", "recap")

        if "reward_model" not in row_dict:
            row_dict["reward_model"] = {"ground_truth": row_dict.get("response", ""), "style": "rule"}

        row_dict["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)

        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = {}
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index %s, data source: %s", index, row_dict.get("data_source"))
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict
