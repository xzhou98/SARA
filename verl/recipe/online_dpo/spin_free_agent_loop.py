"""
spin_free agent loop: single-turn agent loop that consumes a fully-formatted
prompt text supplied by the trainer.

The trainer builds two distinct prompt strings per training sample:
  - prompt_1 = chat_template(instruction) + "<think>\n" + prefill_prompts
  - prompt_2 = prompt_1 + "\n" + reflection

This loop simply tokenizes the supplied prompt text (no extra chat template
processing) and asks the rollout server to generate one continuation.

This mirrors recipe.recap.recap_agent_loop.RECAPAgentLoop, which appends
prefill text to prompt_ids manually instead of relying on apply_chat_template
to format an assistant turn (which would corrupt the prefill behaviour).
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


@register("spin_free_agent")
class SpinFreeAgentLoop(AgentLoopBase):
    """Single-turn agent loop for spin_free reference-free online DPO.

    Expects ``spin_free_prompt_text`` (str) in the per-sample kwargs supplied
    by the trainer. The text is tokenized as-is and used directly as
    ``prompt_ids``; the model generates a single continuation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        # The trainer always provides spin_free_prompt_text. raw_prompt is kept
        # only so the postprocessor (which expects it) does not crash.
        prompt_text = kwargs.get("spin_free_prompt_text", None)
        if prompt_text is None:
            raise KeyError(
                "spin_free_agent requires 'spin_free_prompt_text' in non_tensor_batch. "
                "Make sure RaySpinFreeTrainer._build_spin_free_generation_batch was used."
            )

        # Tokenize the fully-formatted prompt text directly (no chat template).
        prompt_ids = normalize_token_ids(
            self.tokenizer.encode(prompt_text, add_special_tokens=False)
        )

        # Left-truncate so prompt_ids fit into the configured prompt window.
        if len(prompt_ids) > self.prompt_length:
            prompt_ids = prompt_ids[-self.prompt_length :]

        metrics: dict[str, Any] = {}
        with simple_timer("generate_sequences", metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=None,
                video_data=None,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

        response_mask = [1] * len(output.token_ids)

        loop_output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data={},
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )
        loop_output.extra_fields.update({"turn_scores": [], "tool_rewards": []})
        return loop_output
