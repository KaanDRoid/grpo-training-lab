"""Deterministic, leakage-checked arithmetic curriculum for GRPO experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import random


PROMPT_TEMPLATE = "Compute the sum. Reply with ONLY the integer, nothing else.\n{a} + {b} ="


@dataclass(frozen=True)
class ArithmeticExample:
    a: int
    b: int

    @property
    def prompt(self) -> str:
        return PROMPT_TEMPLATE.format(a=self.a, b=self.b)

    @property
    def answer(self) -> int:
        return self.a + self.b

    @property
    def key(self) -> tuple[int, int]:
        return self.a, self.b


@dataclass(frozen=True)
class ArithmeticCurriculum:
    """A fixed-width progression whose samples are stable across restarts.

    Stage selection depends only on the absolute optimizer step, never the final requested run
    length. Consequently a checkpoint created for 12 steps can safely be extended to 24 steps.
    """

    seed: int = 17
    operand_digits: tuple[int, ...] = (4, 5, 6)
    steps_per_stage: int = 8

    def __post_init__(self) -> None:
        if not self.operand_digits:
            raise ValueError("operand_digits must not be empty")
        if any(digit < 1 for digit in self.operand_digits):
            raise ValueError("operand_digits must contain positive integers")
        if self.steps_per_stage < 1:
            raise ValueError("steps_per_stage must be positive")

    def to_config(self) -> dict:
        value = asdict(self)
        value["operand_digits"] = list(self.operand_digits)
        return value

    def stage_for_step(self, step: int) -> int:
        if step < 0:
            raise ValueError("step must be non-negative")
        return min(step // self.steps_per_stage, len(self.operand_digits) - 1)

    def digit_for_step(self, step: int) -> int:
        return self.operand_digits[self.stage_for_step(step)]

    def batch_for_step(self, step: int, num_prompts: int) -> list[ArithmeticExample]:
        if num_prompts < 1:
            raise ValueError("num_prompts must be positive")
        stage = self.stage_for_step(step)
        local_step = step - stage * self.steps_per_stage
        start = local_step * num_prompts
        return self._stage_examples(stage, start + num_prompts)[start:]

    def training_examples(self, steps: int, prompts_per_step: int) -> list[ArithmeticExample]:
        if steps < 0:
            raise ValueError("steps must be non-negative")
        return [
            example
            for step in range(steps)
            for example in self.batch_for_step(step, prompts_per_step)
        ]

    def heldout_examples(
        self,
        prompts_per_stage: int,
        *,
        exclude: list[ArithmeticExample] | tuple[ArithmeticExample, ...] = (),
    ) -> dict[int, list[ArithmeticExample]]:
        """Return a fixed evaluation split, explicitly rejecting every training pair."""
        if prompts_per_stage < 1:
            raise ValueError("prompts_per_stage must be positive")
        excluded = {example.key for example in exclude}
        result: dict[int, list[ArithmeticExample]] = {}
        for stage, digits in enumerate(self.operand_digits):
            result[digits] = self._unique_examples(
                digits=digits,
                count=prompts_per_stage,
                namespace=f"eval-stage-{stage}",
                exclude=excluded,
            )
        return result

    def _stage_examples(self, stage: int, count: int) -> list[ArithmeticExample]:
        return self._unique_examples(
            digits=self.operand_digits[stage],
            count=count,
            namespace=f"train-stage-{stage}",
            exclude=set(),
        )

    def _unique_examples(
        self,
        *,
        digits: int,
        count: int,
        namespace: str,
        exclude: set[tuple[int, int]],
    ) -> list[ArithmeticExample]:
        lower = 0 if digits == 1 else 10 ** (digits - 1)
        upper = 10**digits - 1
        population = (upper - lower + 1) ** 2
        if count + len(exclude) > population:
            raise ValueError("requested split is larger than the operand-pair population")

        # Domain-separated SHA-256 avoids dependence on process-global RNG state and keeps the
        # schedule identical on Windows and Linux/Python restarts.
        material = f"arithmetic-curriculum-v1:{self.seed}:{namespace}".encode("utf-8")
        rng = random.Random(int.from_bytes(hashlib.sha256(material).digest(), "big"))
        examples: list[ArithmeticExample] = []
        seen = set(exclude)
        while len(examples) < count:
            example = ArithmeticExample(rng.randint(lower, upper), rng.randint(lower, upper))
            if example.key in seen:
                continue
            seen.add(example.key)
            examples.append(example)
        return examples


def unpack_examples(examples: list[ArithmeticExample]) -> tuple[list[str], list[int]]:
    return [example.prompt for example in examples], [example.answer for example in examples]
