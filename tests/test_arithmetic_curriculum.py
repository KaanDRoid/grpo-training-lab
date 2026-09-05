import pytest

from arithmetic_curriculum import ArithmeticCurriculum, unpack_examples


def test_stage_progression_is_absolute_step_based():
    curriculum = ArithmeticCurriculum(seed=17, operand_digits=(4, 5, 6), steps_per_stage=2)
    assert [curriculum.digit_for_step(step) for step in range(8)] == [4, 4, 5, 5, 6, 6, 6, 6]


def test_batches_are_deterministic_unique_and_restart_safe():
    first = ArithmeticCurriculum(seed=17, steps_per_stage=3)
    second = ArithmeticCurriculum(seed=17, steps_per_stage=3)
    uninterrupted = first.training_examples(8, 2)
    restarted = [
        example
        for step in range(4, 8)
        for example in second.batch_for_step(step, 2)
    ]
    assert uninterrupted[8:] == restarted
    assert len({example.key for example in uninterrupted}) == len(uninterrupted)


def test_different_seed_changes_schedule():
    left = ArithmeticCurriculum(seed=17).training_examples(4, 1)
    right = ArithmeticCurriculum(seed=18).training_examples(4, 1)
    assert left != right


def test_heldout_split_is_disjoint_and_has_each_difficulty():
    curriculum = ArithmeticCurriculum(seed=17, steps_per_stage=2)
    training = curriculum.training_examples(8, 2)
    heldout = curriculum.heldout_examples(5, exclude=training)
    training_keys = {example.key for example in training}
    assert set(heldout) == {4, 5, 6}
    for digits, examples in heldout.items():
        assert len(examples) == 5
        assert not training_keys.intersection(example.key for example in examples)
        assert all(len(str(example.a)) == digits and len(str(example.b)) == digits for example in examples)


def test_unpack_examples_preserves_alignment():
    examples = ArithmeticCurriculum(seed=17).batch_for_step(0, 2)
    prompts, answers = unpack_examples(examples)
    assert len(prompts) == len(answers) == 2
    assert all(str(answer) not in prompt for prompt, answer in zip(prompts, answers))
    assert answers == [example.a + example.b for example in examples]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"operand_digits": ()}, "operand_digits"),
        ({"operand_digits": (0,)}, "operand_digits"),
        ({"steps_per_stage": 0}, "steps_per_stage"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ArithmeticCurriculum(**kwargs)
