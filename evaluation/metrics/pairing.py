"""Which generated trajectory is compared with which real one, for the trajectory metrics (5-8).

1. ID pairing: a generated trajectory with a ref_id is paired with that real test trajectory.
2. Shared fallback for models without ref_id (Diff-RNTraj and the real-train floor), set by fallback_pairing.rule:
   same_cell  a real test trajectory that starts and ends in the same cells of the canonical 0.005 degree grid; among
              several, the one with the smallest 32-point privacy distance. Generated trajectories without one are
              left out, and the share that has one is reported.
   nearest    the closest real test trajectory: of the `candidates` with the most similar start, end and length, the
              one with the smallest 32-point privacy distance.
"""
from __future__ import annotations

import numpy as np

from evaluation.geo import haversine_m, project_m
from evaluation.metrics.grid import Grid
from evaluation.metrics.privacy import resample
from evaluation.metrics.trajset import TrajSet, sample_index


def choose_pairs(generated: TrajSet, real: TrajSet, settings: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (generated index, real index, info) for the pairs the trajectory metrics use.

    Uses ID pairing when any generated trajectory has a ref_id in the real set, otherwise the shared fallback. At most
    settings["max_pairs"] pairs are kept, sampled with the seed from the generated trajectories that have a match;
    info says how they were paired and what share has a match.
    """
    real_row = {traj_id: i for i, traj_id in enumerate(real.ids)}
    with_ref = np.array([ref is not None and ref in real_row for ref in generated.refs])
    how = "id" if with_ref.any() else settings["fallback_pairing"]["rule"]
    if how == "id":
        eligible = np.flatnonzero(with_ref)
    elif how == "same_cell":
        gen_keys, real_keys = _od_keys(generated, settings), _od_keys(real, settings)
        eligible = np.flatnonzero(np.isin(gen_keys, real_keys))
    elif how == "nearest":
        eligible = np.arange(len(generated))
    else:
        raise ValueError(f"fallback_pairing.rule must be same_cell or nearest, not {how!r}")

    gen_index = eligible[sample_index(len(eligible), settings["max_pairs"], settings["seed"])]
    if how == "id":
        real_index = np.array([real_row[generated.refs[i]] for i in gen_index], dtype=int)
    elif how == "same_cell":
        real_index = same_cell_pairs(generated.take(gen_index), real, settings)
    else:
        real_index = nearest_pairs(generated.take(gen_index), real, settings)
    info = {"pairing": how, "pairs": int(len(gen_index)),
            "paired_share": float(len(eligible) / len(generated)) if len(generated) else float("nan")}
    return gen_index, real_index, info


def same_cell_pairs(generated: TrajSet, real: TrajSet, settings: dict) -> np.ndarray:
    """For each generated trajectory, the closest real one with the same start cell and end cell.

    Every generated trajectory given here must have at least one such real trajectory.
    """
    n_points = settings["privacy"]["points"]
    gen_keys, real_keys = _od_keys(generated, settings), _od_keys(real, settings)
    gen_lat, gen_lng = resample(generated, n_points)
    real_lat, real_lng = resample(real, n_points)
    order = np.argsort(real_keys, kind="stable")
    sorted_keys = real_keys[order]
    best = np.empty(len(generated), dtype=int)
    for i, key in enumerate(gen_keys):
        lo, hi = np.searchsorted(sorted_keys, key, "left"), np.searchsorted(sorted_keys, key, "right")
        candidates = order[lo:hi]
        d = haversine_m(gen_lat[i], gen_lng[i], real_lat[candidates], real_lng[candidates]).mean(axis=1)
        best[i] = candidates[np.argmin(d)]
    return best


def nearest_pairs(generated: TrajSet, real: TrajSet, settings: dict, chunk: int = 500) -> np.ndarray:
    """For each generated trajectory, the index of the closest real trajectory (the `nearest` rule)."""
    k = min(settings["fallback_pairing"]["candidates"], len(real))
    n_points = settings["privacy"]["points"]
    gen_features, real_features = _features_km(generated), _features_km(real)
    gen_lat, gen_lng = resample(generated, n_points)
    real_lat, real_lng = resample(real, n_points)

    best = np.empty(len(generated), dtype=int)
    for start in range(0, len(generated), chunk):
        rows = slice(start, start + chunk)
        feature_distance = ((gen_features[rows, None, :] - real_features[None]) ** 2).sum(axis=2)
        candidates = np.argpartition(feature_distance, k - 1, axis=1)[:, :k]
        # 32-point privacy distance from each generated trajectory to each of its candidates
        d = haversine_m(gen_lat[rows, None, :], gen_lng[rows, None, :],
                        real_lat[candidates], real_lng[candidates]).mean(axis=2)
        best[rows] = candidates[np.arange(len(candidates)), np.argmin(d, axis=1)]
    return best


def _od_keys(trajectories: TrajSet, settings: dict) -> np.ndarray:
    """One number per trajectory for its (start cell, end cell) pair on the canonical grid."""
    grid = settings["grid"]
    g = Grid.from_settings(grid["cell_deg"], grid["lng_bounds"], grid["lat_bounds"])
    start, end = g.cells(*trajectories.first_points()), g.cells(*trajectories.last_points())
    return (start + 1) * (g.size + 1) + (end + 1)


def _features_km(trajectories: TrajSet) -> np.ndarray:
    """Start x, y, end x, y and path length of each trajectory, all in km (columns of an (n, 5) array)."""
    start_x, start_y = project_m(*trajectories.first_points())
    end_x, end_y = project_m(*trajectories.last_points())
    return np.column_stack([start_x, start_y, end_x, end_y, trajectories.lengths_m()]) / 1000.0
