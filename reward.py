"""Generated integer-addition prompts with exact-match and relative-distance rewards."""

import random
import re

import torch

_INT = re.compile(r"-?\d+")
REWARD_MODES = ("exact", "relative_distance")


def make_arithmetic_batch(num_prompts, seed=None, max_operand=99):
    """Return (prompts: list[str], answers: list[int]), one per prompt."""
    rng = random.Random(seed)
    prompts, answers = [], []
    for _ in range(num_prompts):
        a, b = rng.randint(0, max_operand), rng.randint(0, max_operand)
        prompts.append(
            "Compute the sum. Reply with ONLY the integer, nothing else.\n"
            f"{a} + {b} ="
        )
        answers.append(a + b)
    return prompts, answers


def _last_integer(text):
    found = _INT.findall(text)
    return int(found[-1]) if found else None


def arithmetic_reward(completions, answers_expanded, mode="exact"):
    """
    completions       : list[str] length B (completion text only, no prompt echo)
    answers_expanded  : list[int] length B, prompt-major -- each prompt's answer repeated
                        num_generations times so it aligns row-for-row with completions.
    ``exact`` is 1 only when the last integer equals the answer. ``relative_distance`` is a
    deterministic, verifier-only training signal: ``max(0, 1 - |prediction-answer|/|answer|)``.
    Missing integers receive zero and exact answers always receive one. Evaluation remains exact.
    """
    if mode not in REWARD_MODES:
        raise ValueError(f"unknown arithmetic reward mode {mode!r}")
    rewards = []
    for text, ans in zip(completions, answers_expanded):
        prediction = _last_integer(text)
        answer = int(ans)
        if prediction is None:
            reward = 0.0
        elif mode == "exact":
            reward = 1.0 if prediction == answer else 0.0
        else:
            scale = max(abs(answer), 1)
            reward = max(0.0, 1.0 - abs(prediction - answer) / scale)
        rewards.append(reward)
    return torch.tensor(rewards, dtype=torch.float32)


def expand_answers(answers, num_generations):
    """[a0, a1, ...] -> [a0]*n + [a1]*n + ...  (prompt-major, matches rollout_client B order)."""
    return [a for a in answers for _ in range(num_generations)]
