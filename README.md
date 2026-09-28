<div align='center'>
  
# Towards Mitigating Deceptive Safety Alignment in Large Reasoning Models (NeurIPS-2026)

[![Venue: NeurIPS 2026](https://img.shields.io/badge/Venue-NeurIPS%202026-green)](https://neurips.cc/)
[![issues](https://img.shields.io/badge/Issues-Welcome!-yellow)](https://github.com/xzhou98/SARA/issues)
[![GitHub repo size](https://img.shields.io/github/repo-size/xzhou98/SARA)](https://github.com/xzhou98/SARA)
[![GitHub top language](https://img.shields.io/github/languages/top/xzhou98/SARA)](https://github.com/xzhou98/SARA)
[![GitHub stars](https://img.shields.io/github/stars/xzhou98/SARA)](https://github.com/xzhou98/SARA)
</div>

This is the official code repository for AAAI 2026 paper "Towards Mitigating Deceptive Safety Alignment in Large Reasoning Models"  by [Xiangyu Zhou](https://xzhou98.github.io/), [Saleh Zare Zade](https://scholar.google.com/citations?user=O3X_iagAAAAJ&hl=en&oi=ao), [Rafi Ibn Sultan](https://rafiibnsultan.github.io/), [Alexander Kotov](https://rusillini.github.io/), [Dongxiao Zhu](https://dongxiaozhu.github.io/)

<table align="center">
  <tr>
    <td align="center"> 
      <img src="images/illustration.png" alt="Teaser" style="width: 1100px;"/> 
      <br>
      <em style="font-size: 18px;">  <strong style="font-size: 18px;">Figure 1:</strong> Illustration of the proposed TIF framework.</em>
    </td>
  </tr>
</table>




Code and data for **DSAR** (Deceptive Safety Alignment Rate), a metric for how often a reasoning model's
chain of thought and final answer send contradictory safety signals, and **SARA** (Safety-Aware
Reasoning Alignment), an RL method that rewards both safety-aware reasoning and safe final answers.

- **DSAR evaluation** (Sec. 2.3): the reasoning trace is judged sentence by sentence for safety awareness
  and as a whole for harmfulness. The final answer is judged for safety. A generation is *deceptive* when
  the two verdicts disagree.
- **SARA** (Sec. 3): DAPO-style RL on 2K prompts (1K harmful from SafeChain, 1K benign from FalseReject),
  half of them augmented with counter-aligned prefilled reasoning. For harmful prompts the reward is

  $$R^{harm} = \tfrac{1}{2}\, R^{cot} \cdot R^{SA} + \tfrac{1}{2}\, R^{ans}, \qquad R^{SA} = 1 - k^\star / N$$

  where $k^\star$ is the index of the first safety-aware sentence among the $N$ reasoning sentences.
  For benign prompts the reward is $1 - \text{refusal score}/10$.

## Repository structure

```
SARA/
├── configs/
│   ├── model_config.yaml         # chat tags + sampling params per model family
│   └── deepspeed_zero2.yaml      # accelerate config for the SFT baselines
├── data/
│   ├── train/                    # training sets (HF datasets, load with `load_from_disk`)
│   └── eval/                     # benchmarks, including the pre-generated adversarial prefills
├── evaluation/                   # DSAR / SAR / SS / HS evaluation pipeline
├── baselines/                    # off-policy SFT baselines: SafeChain, STAR-1, SafePath
├── scripts/
│   ├── serve_reward_models.sh    # vLLM servers for the reward models used during RL
│   └── merge_lora.py             # merge a LoRA adapter into its base model for evaluation
└── verl/                         # verl RL framework (upstream) + our two recipes:
    └── recipe/
        ├── sara/                 # SARA (our method)
        └── recap/                # RECAP baseline (answer-only reward)
```

Everything under `verl/` other than `recipe/sara/` and `recipe/recap/` is the upstream
[verl](https://github.com/volcengine/verl) library.

## Data

All datasets are stored in Hugging Face `save_to_disk` format.

| Path | Size | Used for | Fields |
|---|---|---|---|
| `data/train/sara_train_2k` | 2,000 | SARA, RECAP, SafePath | `instruction`, `label` (`vanilla_harmful` / `adversarial_harmful` / `benign`), `response`, `prefill_prompts` |
| `data/train/safechain_sft_2k` | 2,000 | SafeChain baseline | `instruction`, `label`, `response` |
| `data/train/star1_1k` | 1,000 | STAR-1 baseline | `instruction`, `label`, `response` |
| `data/eval/strongreject` | 313 | Safety, adv. prefilling | `prompt`, `prefill_prompts` |
| `data/eval/safechain` | 500 | Safety, standard | `instruction`, `prefill_prompts` |
| `data/eval/icl` + `data/eval/icl_demos.json` | 400 | Unseen attack: 16-shot ICL | `prompt` |
| `data/eval/h-cot` | 50 | Unseen attack: H-CoT | `prompt` |
| `data/eval/or_bench` | 1,319 | Helpfulness (OR-Bench-hard) | `prompt` |
| `data/eval/fortress` | 500 | Helpfulness | `prompt` |
| `data/eval/xstest` | 450 | Helpfulness | `prompt` |

- **`sara_train_2k`**: the harmful half comes from SafeChain and the benign half from FalseReject.
  `prefill_prompts` is non-empty for exactly half of each label (Appendix B.2). Harmful prompts get an
  unsafe prefill and benign prompts get an over-cautious one. SARA and RECAP are trained on this same
  dataset and differ only in the reward function.
- **`safechain_sft_2k`**: the same 1K harmful prompts, plus 1K benign samples from the SafeChain dataset.
- **Adversarial prefills** (`prefill_prompts`) start with
  *"Okay, so I am trying to figure out how to do it and give a detailed plan. First,"*, followed by a
  continuation generated by DeepSeek-R1-Distill-Qwen-1.5B. The same prefills are used for training and
  evaluation.

---

## Running the experiments

The instructions below assume **one node with 4 × 80GB GPUs (e.g. H100)**:

| GPU | During RL training | During evaluation |
|---|---|---|
| 0, 1 | policy training (actor, rollout, reference) | model under test + judges |
| 2 | reward server: DeepSeek-R1-Distill-Qwen-32B (over-refusal judge) | unused |
| 3 | reward servers: Granite-Guardian-3.3-8B (safety guard) + gpt-oss-safeguard-20b (sentence-level judge) | unused |

### Step 0: Environment and Hugging Face login

We use two conda environments: one for RL training with verl, and one for evaluation and the SFT baselines.

```bash
# RL training (SARA / RECAP) and the reward servers
conda create -n verl python=3.10 -y && conda activate verl
cd verl && pip install -e . && pip install vllm && cd ..

# Evaluation + SFT baselines
conda create -n sara-eval python=3.10 -y && conda activate sara-eval
pip install -r requirements.txt
```

> **Hugging Face access (required).** No access token is included in this repository. Several of the
> models are gated or require an account (e.g. `Qwen/Qwen3Guard-Gen-8B`, `openai/gpt-oss-safeguard-20b`,
> `ibm-granite/granite-guardian-3.3-8b`). In **each** environment, and before running anything, either
> log in with
> ```bash
> huggingface-cli login
> ```
> or export your own token in the shell you run from: `export HF_TOKEN=<your_token>`.
> Accept each gated model's license on its Hugging Face page first.

All commands below are run from the repository root unless stated otherwise.

### Step 1: Start the reward servers (GPUs 2 and 3)

RL training queries three reward models over HTTP. Start them in their own terminal (or a `tmux`
session) and leave them running for the whole training run:

```bash
conda activate verl
bash scripts/serve_reward_models.sh sara     # for SARA: all three models
# bash scripts/serve_reward_models.sh recap  # for RECAP: guard + refusal judge only
```

| Port | GPU | Model | Role in the reward |
|---|---|---|---|
| 8003 | 2 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | over-refusal score (0–10) → benign reward |
| 8002 | 3 | `ibm-granite/granite-guardian-3.3-8b` | P(safe) of reasoning / answer → $R^{cot}$, $R^{ans}$ |
| 8001 | 3 | `openai/gpt-oss-safeguard-20b` | sentence-level safety awareness → $R^{SA}$ (SARA only) |

- The script waits until every server is healthy and prints `All reward servers are up`.
- Logs go to `logs/reward_servers/`. Loading all three models the first time can take a while.
- The two models on GPU 3 each take 45% of its memory. The script starts them one after the other,
  which vLLM needs when two servers share a GPU.
- Different GPUs: `REFUSAL_JUDGE_GPU=2 GUARD_GPU=3 SENTENCE_JUDGE_GPU=3 bash scripts/serve_reward_models.sh sara`.
- Check the servers by hand: `curl http://127.0.0.1:8001/health` (likewise for 8002 and 8003).

### Step 2: Train SARA (GPUs 0 and 1)

In a second terminal:

```bash
conda activate verl
cd verl
bash recipe/sara/run_sara_lora.sh                                                    # DeepSeek-R1-0528-Qwen3-8B
MODEL_PATH=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B bash recipe/sara/run_sara_lora.sh
```

- The script checks that all reward servers respond, then trains on `TRAIN_GPUS` (default `0,1`).
- Settings: LoRA r=8 / α=16 on all linear layers, lr 3e-5, 1 epoch, 32 prompts per batch, 4 rollouts
  per prompt, clip ε = 0.2 / 0.28, no KL. All hyperparameters are at the top of the script.
- Checkpoints are saved every 10 steps to
  `verl/ckpts/Deceptive_Alignment/SARA-<model>-LoRA_lr-3e-5_epoch-1/global_step_<N>/`, and training
  resumes automatically from the latest one if restarted.
- Training curves are logged to Weights & Biases (`wandb login` first, or edit `trainer.logger` in the
  script). Per-component rewards are logged as `reasoning_reward`, `answer_reward`, `sac_reward` (= $R^{SA}$)
  and `refusal_reward`.
- If the reward servers run on another machine: `REWARD_HOST=<ip> bash recipe/sara/run_sara_lora.sh`.
  The ports can be changed with `GUARD_PORT`, `SENTENCE_JUDGE_PORT` and `REFUSAL_JUDGE_PORT`.

### Step 3: Train the baselines

**RECAP** (on-policy RL, answer-only reward). Same data, hyperparameters and GPU layout as SARA; only
the reward function differs. The reward servers from Step 1 can stay up (RECAP just doesn't use port
8001).

```bash
conda activate verl
cd verl
bash recipe/recap/run_recap_lora.sh
```

**SafeChain / STAR-1 / SafePath** (off-policy SFT, LoRA r=8, 2 GPUs × 4 per device × 4 accumulation
= batch 32). These don't need the reward servers:

```bash
conda activate sara-eval
CUDA_VISIBLE_DEVICES=0,1 bash baselines/run_sft.sh R1-0528        # or R1-Qwen-14B
```

Checkpoints are written to `saves/<model_family>/`.

### Step 4: Merge LoRA weights

vLLM evaluates full models, so merge each LoRA adapter into its base model first:

```bash
conda activate sara-eval

# SARA / RECAP (verl checkpoint)
python scripts/merge_lora.py --base_model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
    --adapter verl/ckpts/Deceptive_Alignment/SARA-DeepSeek-R1-0528-Qwen3-8B-LoRA_lr-3e-5_epoch-1/global_step_<N>/actor/lora_adapter \
    --output_dir verl/ckpts/Deceptive_Alignment/SARA-DeepSeek-R1-0528-Qwen3-8B-LoRA_lr-3e-5_epoch-1/merged

# SFT baselines
python scripts/merge_lora.py --base_model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
    --adapter saves/R1-0528/<run>/checkpoint-<step> --output_dir saves/R1-0528/<run>/merged
```

### Step 5: Evaluate

Evaluation doesn't need the reward servers; stop them (Ctrl-C in their terminal) to free GPUs 2 and 3.
Each evaluation step loads its model locally with vLLM on the GPU(s) in `CUDA_VISIBLE_DEVICES`:

```bash
conda activate sara-eval
CUDA_VISIBLE_DEVICES=0 bash evaluation/run_eval.sh <model_path> <model_family> [output_dir]

# e.g. the original model and a SARA checkpoint
CUDA_VISIBLE_DEVICES=0 bash evaluation/run_eval.sh deepseek-ai/DeepSeek-R1-0528-Qwen3-8B R1-0528
CUDA_VISIBLE_DEVICES=1 bash evaluation/run_eval.sh \
    verl/ckpts/Deceptive_Alignment/SARA-DeepSeek-R1-0528-Qwen3-8B-LoRA_lr-3e-5_epoch-1/merged R1-0528 results/R1-0528/SARA
```

`model_family` is a key of `configs/model_config.yaml`: `R1-0528`, `R1-Qwen-14B`, `R1-Llama-8B`,
`Gemma-4` or `gpt-oss`. To shard a large model across GPUs, set `TP` to the number of GPUs, e.g.
`CUDA_VISIBLE_DEVICES=0,1 TP=2 bash evaluation/run_eval.sh ...`. Results go to
`results/<model_family>/<model_name>/`: one CSV per benchmark and setting, plus all metrics in
`eval_summary.json`. Steps whose output already exists are skipped, so an interrupted run can be resumed.

The pipeline has four steps. Each step is a separate script, so each loads only one vLLM model:

| Step | Script | Model | Output |
|---|---|---|---|
| 1 | `generate.py` | model under test | `<task>_{no_prefill,prefill}.csv` with `reasoning` and `final`. StrongReject and SafeChain are run with and without the adversarial prefill. |
| 2 | `judge_guard.py` | `Qwen/Qwen3Guard-Gen-8B` | `is_cot_safe` = $r_{safe}(y_{cot})$, `is_ans_safe` = $s(y_{ans})$ → **SS** |
| 3 | `judge_safety_aware.py` | `openai/gpt-oss-safeguard-20b` | `safety_aware_sent_idx` (first safety-aware sentence, −1 if none) → **SAR**, **DSAR** |
| 4 | `judge_helpfulness.py` | `openai/gpt-oss-safeguard-20b` | `is_helpful` on OR-Bench / Fortress / XSTest → **HS** |

Metric definitions (Sec. 2.3 and Appendix B.3), as they appear in `eval_summary.json`:

- **SAR** = fraction of reasoning traces with at least one safety-aware sentence ($r_{SA}=1$)
- **SS** = fraction of final answers judged safe
- **DSAR** = fraction of generations with $r(y_{cot}) \neq s(y_{ans})$, where
  $r(y_{cot}) = \mathbb{1}[r_{SA}=1 \lor r_{safe}=1]$. Table 2 reports 1 − DSAR.
- **HS** = fraction of benign-prompt answers classified as non-refusals

In the paper, StrongReject is reported under adversarial prefilling (`strongreject_prefill`) and
SafeChain under the standard setting (`safechain_no_prefill`). `evaluation/common.py` holds the shared
helpers and `evaluation/prompts.py` the judge instructions. If gpt-oss-safeguard fails to load under the
vLLM V1 engine, set `VLLM_USE_V1_ENGINE=0`.

**Utility** (GSM8K, MMLU-Pro) is evaluated with
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness):

```bash
CUDA_VISIBLE_DEVICES=0 lm_eval --model vllm \
  --model_args pretrained=<model_path>,dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.9,enable_thinking=True,think_end_token="</think>" \
  --tasks gsm8k_cot --num_fewshot 4 --batch_size auto --apply_chat_template \
  --gen_kwargs do_sample=True,temperature=0.6,top_p=0.95,max_new_tokens=10240 \
  --output_path results/lm_eval/gsm8k_cot --log_samples

CUDA_VISIBLE_DEVICES=0 lm_eval --model vllm \
  --model_args pretrained=<model_path>,dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.95,enable_thinking=True,think_end_token="</think>" \
  --tasks mmlu_pro --batch_size auto --apply_chat_template --fewshot_as_multiturn \
  --gen_kwargs do_sample=True,temperature=0.6,top_p=0.95,max_new_tokens=20480,max_gen_toks=15240 \
  --output_path results/lm_eval/mmlu_pro
```

---

## Code reference

### `verl/recipe/sara/` (SARA)

| File | Purpose |
|---|---|
| `run_sara_lora.sh` | launch script and all hyperparameters |
| `main_sara.py` | entry point; builds the dataset and runs verl's `RayDAPOTrainer` |
| `sara_dataset.py` | loads `sara_train_2k`, turns `instruction` into a chat prompt and passes `prefill_prompts` through |
| `sara_agent_loop.py` | appends `<think>` + prefill to the prompt, so the policy continues from the prefill (prefill tokens are not trained on) |
| `sara_reward.py` | **the SARA reward** (Eq. 2–5): routes by `label` and queries the three reward servers |
| `config/constants.py` | judge instructions: sentence-level safety awareness, and the 0–10 over-refusal rubric |
| `config/*.yaml` | Hydra config that wires the dataset, agent loop and reward into verl |

### `verl/recipe/recap/` (RECAP baseline)

Same file layout as `sara/`. The only functional difference is `recap_reward.py`: harmful prompts are
rewarded with $R^{ans}$ alone, with no reasoning or safety-awareness term.

### `baselines/`

| File | Purpose |
|---|---|
| `finetune.py` | SFT training loop. `--loss_type SFT` for SafeChain / STAR-1, `--loss_type safepath` for SafePath |
| `data_module.py` | tokenization, with loss on response tokens only. SafePath prefixes half of the responses with *"Let’s think about safety first."* |
| `run_sft.sh` | runs all three baselines with the paper's settings |

### `scripts/`

| File | Purpose |
|---|---|
| `serve_reward_models.sh` | starts the reward servers for RL training (Step 1) |
| `merge_lora.py` | merges a LoRA adapter into the base model (Step 4) |

## Acknowledgements

The RL code is built on [verl](https://github.com/volcengine/verl). The benchmarks come from
SafeChain, FalseReject, StrongReject, STAR-1, OR-Bench, Fortress, XSTest and H-CoT; please cite the
original works when you use them.
