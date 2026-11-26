import numpy as np
import pytest

from src.selection_policy import select_with_capacity


def test_selects_highest_scoring_eligible_applicants_up_to_capacity():
    selected, target = select_with_capacity(
        np.array([0.9, 0.8, 0.7, 0.6]),
        np.array([True, False, True, True]),
        capacity_fraction=0.5,
    )

    assert target == 2
    assert selected.tolist() == [True, False, True, False]


def test_capacity_selection_uses_stable_input_order_for_ties():
    selected, target = select_with_capacity(
        np.array([0.8, 0.8, 0.8]),
        np.array([True, True, True]),
        capacity_fraction=0.34,
    )

    assert target == 1
    assert selected.tolist() == [True, False, False]


def test_fewer_eligible_applicants_leave_seats_unfilled():
    selected, target = select_with_capacity(
        np.array([0.9, 0.8, 0.7]),
        np.array([True, False, True]),
        capacity_fraction=1.0,
    )

    assert target == 3
    assert selected.tolist() == [True, False, True]


@pytest.mark.parametrize("fraction", [-0.01, 1.01])
def test_invalid_capacity_fraction_raises(fraction):
    with pytest.raises(ValueError, match="capacity_fraction"):
        select_with_capacity(np.array([0.8]), np.array([True]), fraction)


def test_invalid_scores_raise():
    with pytest.raises(ValueError, match="finite"):
        select_with_capacity(np.array([np.nan]), np.array([True]))
