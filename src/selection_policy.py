"""Capacity-limited selection of eligible applicants."""

import math
from typing import Tuple

import numpy as np

DEFAULT_CAPACITY_FRACTION = 0.20


def select_with_capacity(
    scores: np.ndarray,
    eligible: np.ndarray,
    capacity_fraction: float = DEFAULT_CAPACITY_FRACTION,
) -> Tuple[np.ndarray, int]:
    """Select the highest-scoring eligible applicants up to a rounded seat target.

    Equal scores are resolved by input order. If fewer applicants are eligible
    than the target, all eligible applicants are selected and seats remain open.
    """
    scores = np.asarray(scores, dtype=float)
    eligible = np.asarray(eligible, dtype=bool)
    if scores.ndim != 1 or eligible.ndim != 1 or len(scores) != len(eligible):
        raise ValueError("scores and eligible must be one-dimensional arrays of equal length")
    if not np.isfinite(scores).all():
        raise ValueError("scores must contain only finite values")
    if not 0.0 <= capacity_fraction <= 1.0:
        raise ValueError("capacity_fraction must be between 0 and 1")

    target_seats = int(math.floor(len(scores) * capacity_fraction + 0.5))
    eligible_indices = np.flatnonzero(eligible)
    selected = np.zeros(len(scores), dtype=bool)
    seats_to_fill = min(target_seats, len(eligible_indices))
    if seats_to_fill:
        ranked_positions = np.argsort(-scores[eligible_indices], kind="stable")
        selected[eligible_indices[ranked_positions[:seats_to_fill]]] = True
    return selected, target_seats
