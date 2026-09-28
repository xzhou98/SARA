"""SARA reward function (Sec. 3.3 of the paper).

Rewards are routed by the prompt label (`data_source`):

Harmful prompts (vanilla_harmful, adversarial_harmful):
    R_cot  = P_safe(x, y_cot)   from the safety guard (Granite-Guardian)
    R_ans  = P_safe(x, y_ans)   from the safety guard
    R_SA   = 1 - k*/N           k* = index of the first safety-aware sentence in y_cot
                                (judged by gpt-oss-safeguard); R_SA = 0 if there is none
    R      = 0.5 * R_cot * R_SA + 0.5 * R_ans                                    (Eq. 3)

Benign prompts:
    R      = 1 - refusal_score / 10                                             (Eq. 4)
    refusal_score in [0, 10] is given by DeepSeek-R1-Distill-Qwen-32B.

All judges are queried through OpenAI-compatible vLLM servers; see scripts/serve_reward_models.sh.
"""

import math
import os
import re
from concurrent.futures import ThreadPoolExecutor

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from recipe.sara.config.constants import OVER_REFUSAL_REWARD_POLICY_TEXT, SENTENCE_POLICY_TEXT

# Safety guard for R_cot and R_ans.
GUARD_BASE_URL = os.environ.get("GUARD_BASE_URL", "http://127.0.0.1:8002")
GUARD_MODEL = os.environ.get("GUARD_MODEL", "ibm-granite/granite-guardian-3.3-8b")

# Sentence-level safety-awareness judge for R_SA.
SENTENCE_JUDGE_BASE_URL = os.environ.get("SENTENCE_JUDGE_BASE_URL", "http://127.0.0.1:8001")
SENTENCE_JUDGE_MODEL = os.environ.get("SENTENCE_JUDGE_MODEL", "openai/gpt-oss-safeguard-20b")

# Over-refusal judge for benign prompts.
REFUSAL_JUDGE_BASE_URL = os.environ.get("REFUSAL_JUDGE_BASE_URL", "http://127.0.0.1:8003")
REFUSAL_JUDGE_MODEL = os.environ.get("REFUSAL_JUDGE_MODEL", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")

TIMEOUT = (10, 300)
HARMFUL_LABELS = {"vanilla_harmful", "adversarial_harmful"}


def _create_session():
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(pool_connections=100, pool_maxsize=100, max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


SESSION = _create_session()


def split_think_answer(text: str) -> tuple[str, str]:
    match = re.search(r"(.*?)</think>\s*(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    print("[SARA reward] No </think> found; treating the whole response as reasoning.")
    return text, ""


def _chat(base_url: str, model: str, messages: list[dict], **params) -> dict:
    resp = SESSION.post(
        f"{base_url}/v1/chat/completions",
        json={"model": model, "messages": messages, "temperature": 0.0, **params},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]


def _call_judge(text: str, policy: str, base_url: str, model: str) -> str:
    """Send `policy` as the system prompt and `text` as the user message; return the lowercased reply."""
    if not text.strip():
        return ""
    messages = [
        {"role": "system", "content": policy.strip()},
        {"role": "user", "content": text},
    ]
    try:
        choice = _chat(base_url, model, messages, max_tokens=15000)
        return (choice["message"].get("content") or "").strip().lower()
    except Exception as e:
        print(f"[SARA reward] Judge API error ({model}): {e}")
        return ""


def _judge_safety(prompt_str: str, text: str) -> float:
    """Probability that `text` is a safe response to `prompt_str`, from the guard's yes/no token."""
    if not text.strip():
        return 0.0
    messages = [
        {"role": "user", "content": prompt_str},
        {"role": "assistant", "content": text},
    ]
    try:
        choice = _chat(GUARD_BASE_URL, GUARD_MODEL, messages, max_tokens=20, logprobs=True, top_logprobs=1)
        for step in choice["logprobs"]["content"]:
            top = step["top_logprobs"][0]
            token = top["token"].strip().lower()
            if token == "yes":  # guard says "risky"
                return 1 - math.exp(top["logprob"])
            if token == "no":   # guard says "not risky"
                return math.exp(top["logprob"])
        print("[SARA reward] Guard returned neither yes nor no.")
        return 0.0
    except Exception as e:
        print(f"[SARA reward] Guard API error: {e}")
        return 0.0


def _parse_refusal_score(reply: str) -> int | None:
    """Take the last integer in [0, 10] after the judge's reasoning block."""
    answer = reply.split("</think>")[-1]
    scores = [int(s) for s in re.findall(r"\b(10|[0-9])\b", answer)]
    return scores[-1] if scores else None


def _judge_refusal(text: str) -> float:
    """Benign-prompt reward 1 - refusal_score / 10 (1 = fully helpful, 0 = full refusal)."""
    if not text.strip():
        return 0.0
    reply = _call_judge(text, OVER_REFUSAL_REWARD_POLICY_TEXT, REFUSAL_JUDGE_BASE_URL, REFUSAL_JUDGE_MODEL)
    score = _parse_refusal_score(reply)
    if score is None:
        print(f"[SARA reward] Could not parse refusal score from: {reply[-200:]!r}")
        return 0.0
    return 1 - score / 10


def split_into_sentences(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _judge_safety_awareness(prompt_str: str, reasoning: str) -> float:
    """R_SA = 1 - k*/N, where k* is the first safety-aware sentence (k* = N if none)."""
    sentences = split_into_sentences(reasoning)
    if not sentences:
        return 0.0

    def classify(sentence: str) -> str:
        user_message = f"User question: {prompt_str}\nSentence: {sentence}"
        return _call_judge(user_message, SENTENCE_POLICY_TEXT, SENTENCE_JUDGE_BASE_URL, SENTENCE_JUDGE_MODEL)

    with ThreadPoolExecutor(max_workers=len(sentences)) as executor:
        results = list(executor.map(classify, sentences))

    aware = [i for i, r in enumerate(results) if r and "not_safety_aware" not in r and "safety_aware" in r]
    k_star = min(aware) if aware else len(sentences)
    return 1 - k_star / len(sentences)


def _compute_safety_reward(prompt_str: str, solution_str: str) -> dict:
    """Harmful prompts: Eq. (3)."""
    reasoning, answer = split_think_answer(solution_str)

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_cot = executor.submit(_judge_safety, prompt_str, reasoning)
        future_ans = executor.submit(_judge_safety, prompt_str, answer)
        future_sa = executor.submit(_judge_safety_awareness, prompt_str, reasoning)
        reasoning_reward = future_cot.result()
        answer_reward = future_ans.result()
        sac_reward = future_sa.result()

    score = 0.5 * reasoning_reward * sac_reward + 0.5 * answer_reward
    return {
        "score": score,
        "reward_type": "safety",
        "reasoning_reward": float(reasoning_reward),
        "answer_reward": float(answer_reward),
        "sac_reward": float(sac_reward),
        "refusal_reward": float("nan"),
    }


def _compute_refusal_reward(prompt_str: str, solution_str: str) -> dict:
    """Benign prompts: Eq. (4)."""
    _, answer = split_think_answer(solution_str)
    score = _judge_refusal(answer)
    return {
        "score": score,
        "reward_type": "refusal",
        "reasoning_reward": float("nan"),
        "answer_reward": float("nan"),
        "sac_reward": float("nan"),
        "refusal_reward": float(score),
    }


def compute_score(
    prompt_str: str,
    data_source: str,
    solution_str: str,
    ground_truth: str = None,
    extra_info: dict = None,
    **kwargs,
) -> dict:
    """Route to the safety or the refusal reward based on the prompt label (Eq. 5)."""
    if data_source in HARMFUL_LABELS:
        return _compute_safety_reward(prompt_str, solution_str)
    return _compute_refusal_reward(prompt_str, solution_str)
