import pytest

from aggregate_curriculum_runs import student_t_95_interval


def test_student_t_interval_for_three_runs():
    interval = student_t_95_interval([0.0, -4 / 96, -1 / 96])
    assert interval["runs"] == 3
    assert interval["mean"] == pytest.approx(-5 / 288)
    assert interval["ci95_low"] == pytest.approx(-0.0712272, abs=1e-6)
    assert interval["ci95_high"] == pytest.approx(0.0365050, abs=1e-6)


def test_confidence_interval_requires_independent_replication():
    with pytest.raises(ValueError, match="at least two"):
        student_t_95_interval([0.1])
