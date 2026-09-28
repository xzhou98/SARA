"""
SARA agent loop: single-turn agent loop with prefill injection.

After applying the chat template, this loop appends tokenized "<think>" + prefill_text
to the prompt_ids so the model generates continuing from the prefill.
The prefill tokens are part of the prompt (not trained on).
"""

import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

THINK_TAG = "<think>"


@register("sara_agent")
class SARAAgentLoop(AgentLoopBase):
    """Single-turn agent loop that injects <think> + prefill text into the prompt."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        prefill_text = kwargs.get("prefill_text", "")

        # 1. extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        # 2. apply chat template and tokenize the base prompt
        prompt_ids = await self.apply_chat_template(
            messages,
            images=images,
            videos=videos,
        )

        # 3. append <think> + prefill_text as part of the prompt
        # DeepSeek-R1-0528-Qwen3-8B needs an explicit <think> tag to enter
        # reasoning mode; other models (e.g. R1-Distill) already handle it.
        needs_think_tag = "DeepSeek-R1-0528-Qwen3-8B" in getattr(self.tokenizer, "name_or_path", "")

        if needs_think_tag:
            prefill_str = THINK_TAG + prefill_text if prefill_text else THINK_TAG
        else:
            prefill_str = prefill_text if prefill_text else ""

        if prefill_str:
            prefill_ids = normalize_token_ids(
                self.tokenizer.encode(prefill_str, add_special_tokens=False)
            )
            prompt_ids = prompt_ids + prefill_ids

        # 4. generate sequences (model continues from after the prefill)
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        response_mask = [1] * len(output.token_ids)

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )

        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})

        return output
