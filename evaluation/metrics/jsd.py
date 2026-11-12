"""Jensen-Shannon divergence and the histograms it is computed on."""
from __future__ import annotations

import numpy as np


def jsd(p_counts, q_counts) -> float:
    """Jensen-Shannon divergence in log base 2 between two count vectors over the same bins.

    JSD(P, Q) = 1/2 KL(P || M) + 1/2 KL(Q || M) with M = (P + Q) / 2, after turning the counts into probabilities.
    Base 2 keeps it between 0 (same distribution) and 1 (no overlap at all).
    """
    p = np.asarray(p_counts, dtype=float)
    q = np.asarray(q_counts, dtype=float)
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    return float(0.5 * _kl(p, m) + 0.5 * _kl(q, m))


def _kl(p: np.ndarray, m: np.ndarray) -> float:
    """KL(P || M) in base 2, with 0 * log(0) = 0."""
    keep = p > 0
    return float(np.sum(p[keep] * np.log2(p[keep] / m[keep])))


def histogram(values, bins: int, max_value: float) -> np.ndarray:
    """Count values in `bins` equal bins from 0 to max_value, plus one overflow bin for larger values.

    Returns bins + 1 counts: the overflow bin keeps long trajectories instead of dropping them, as the spec asks.
    """
    values = np.asarray(values, dtype=float)
    index = np.floor(values / max_value * bins).astype(np.int64)
    return np.bincount(np.clip(index, 0, bins), minlength=bins + 1)


def counts_by_key(keys_a, keys_b) -> tuple[np.ndarray, np.ndarray]:
    """Count two sets of category keys (e.g. cell or OD numbers) over the categories either one uses."""
    keys_a, keys_b = np.asarray(keys_a), np.asarray(keys_b)
    categories, inverse = np.unique(np.concatenate([keys_a, keys_b]), return_inverse=True)
    inverse = inverse.ravel()
    a = np.bincount(inverse[:len(keys_a)], minlength=len(categories))
    b = np.bincount(inverse[len(keys_a):], minlength=len(categories))
    return a, b
