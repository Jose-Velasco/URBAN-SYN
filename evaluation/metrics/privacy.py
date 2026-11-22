"""Metrics 9-10: membership inference (ROC-AUC) and uniqueness, on the 32-point privacy representation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.geo import EARTH_RADIUS_M, haversine_m
from evaluation.metrics.trajset import TrajSet, sample_index


def resample(trajectories: TrajSet, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Resample every trajectory to n_points evenly spaced along its path.

    Returns lat and lng arrays of shape (trajectories, n_points). Points are placed at equal Haversine distances along
    the path by linear interpolation between the original points. A trajectory that never moves (one point, or all
    points the same) becomes n_points copies of its first point.
    """
    lat_out = np.empty((len(trajectories), n_points))
    lng_out = np.empty((len(trajectories), n_points))
    for k in range(len(trajectories)):
        lat, lng = trajectories.points(k)
        steps = haversine_m(lat[:-1], lng[:-1], lat[1:], lng[1:])
        moved = np.concatenate([[True], steps > 0])  # drop repeated points so the distances increase
        lat, lng = lat[moved], lng[moved]
        along = np.concatenate([[0.0], np.cumsum(steps[steps > 0])])
        if along[-1] == 0:
            lat_out[k], lng_out[k] = lat[0], lng[0]
            continue
        targets = np.linspace(0.0, along[-1], n_points)
        lat_out[k] = np.interp(targets, along, lat)
        lng_out[k] = np.interp(targets, along, lng)
    return lat_out, lng_out


def unit_vectors(lat: np.ndarray, lng: np.ndarray) -> np.ndarray:
    """Turn (n, P) resampled lat/lng arrays into (n, P, 3) points on the unit sphere."""
    lat, lng = np.radians(lat), np.radians(lng)
    return np.stack([np.cos(lat) * np.cos(lng), np.cos(lat) * np.sin(lng), np.sin(lat)], axis=-1)


def _mean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """d_priv between unit-vector trajectories a and b (broadcast over leading axes), in metres.

    The great-circle distance 2R * arcsin(chord / 2) is exactly the Haversine distance, in a cheaper form.
    """
    chord = np.sqrt(((a - b) ** 2).sum(axis=-1))
    return (2 * EARTH_RADIUS_M * np.arcsin(np.clip(chord / 2, 0, 1))).mean(axis=-1)


def nearest_distance(queries: np.ndarray, pool: np.ndarray, exclude: dict | None = None,
                     batch: int = 128, candidates: int = 256) -> np.ndarray:
    """Exact d_priv (m) from every query trajectory to its closest trajectory in the pool.

    queries (Q, P, 3) and pool (N, P, 3) are unit vectors (unit_vectors); exclude maps a query row to pool rows it must
    skip. Comparing every query with every pool trajectory is too slow for whole datasets, so the search uses a lower
    bound: R times the distance between two trajectories' mean vectors is never larger than d_priv (the same bound as
    privacy_metrics.py). For each query, the `candidates` pool trajectories with the smallest bound are compared
    exactly; only if one outside them could still be closer is the rest searched, nearest bound first. The result is
    exact, not an approximation.
    """
    exclude = exclude or {}
    q_mean, p_mean = queries.mean(axis=1), pool.mean(axis=1)
    p_sq = (p_mean ** 2).sum(axis=1)
    k = min(candidates, len(pool))
    out = np.empty(len(queries))
    for start in range(0, len(queries), batch):
        q = slice(start, start + batch)
        bound2 = (q_mean[q] ** 2).sum(axis=1)[:, None] + p_sq[None] - 2 * q_mean[q] @ p_mean.T
        bound = EARTH_RADIUS_M * np.sqrt(np.clip(bound2, 0, None)) - 1e-3  # 1 mm margin for rounding
        for row in range(*q.indices(len(queries))):
            if row in exclude:
                bound[row - start, exclude[row]] = np.inf
        near = np.argpartition(bound, k - 1, axis=1)[:, :k]
        rows = np.arange(len(near))[:, None]
        d = _mean_distance(queries[q][:, None], pool[near])
        d[~np.isfinite(bound[rows, near])] = np.inf
        best = d.min(axis=1)
        out[q] = best
        if k < len(pool):  # pool trajectories outside `near` all have a bound >= the largest inside
            unsure = np.flatnonzero(best > bound[rows, near].max(axis=1))
            for i in unsure:
                out[start + i] = _search(queries[start + i], pool, bound[i])
    return out


def _search(query: np.ndarray, pool: np.ndarray, bound: np.ndarray, block: int = 512) -> float:
    """Exact nearest d_priv for one query, visiting the pool in order of its lower bound."""
    order = np.argsort(bound)
    best = np.inf
    for start in range(0, len(order), block):
        rows = order[start:start + block]
        if bound[rows[0]] >= best:
            break
        rows = rows[np.isfinite(bound[rows])]
        if len(rows):
            best = min(best, float(_mean_distance(query[None], pool[rows]).min()))
    return best


def roc_auc(labels, scores) -> float:
    """ROC-AUC of `scores` for the binary `labels` (1 = member), with tied scores counted as half."""
    labels = np.asarray(labels, dtype=bool)
    ranks = pd.Series(np.asarray(scores, dtype=float)).rank(method="average").to_numpy()
    n_pos, n_neg = labels.sum(), (~labels).sum()
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def privacy_scores(generated: TrajSet, train: TrajSet, test: TrajSet, settings: dict) -> tuple[list[dict], dict]:
    """Score metric 9 (MIA ROC-AUC) and metric 10 (uniqueness and its threshold sweep).

    settings["privacy"]["sample"] is null to use every trajectory (the team's choice), or a number to sample the
    generated, train and test sets to that size with fixed seeds. For a test trajectory, the generated trajectory
    conditioned on it (ref_id) is left out of its nearest-neighbour search, so a conditional model's attack is not
    helped by its own conditioning.
    """
    privacy, seed = settings["privacy"], settings["seed"]
    size, n_points = privacy["sample"], privacy["points"]
    gen = generated.take(sample_index(len(generated), size, seed))
    members = train.take(sample_index(len(train), size, seed + 1))
    non_members = test.take(sample_index(len(test), size, seed + 2))
    gen_u, member_u, non_member_u = (unit_vectors(*resample(t, n_points)) for t in (gen, members, non_members))

    gen_cols = {}
    for j, ref in enumerate(gen.refs):
        if ref is not None:
            gen_cols.setdefault(ref, []).append(j)
    exclude = {i: gen_cols[t] for i, t in enumerate(non_members.ids) if t in gen_cols}

    member_d = nearest_distance(member_u, gen_u)                     # D_G(x) of every member
    non_member_d = nearest_distance(non_member_u, gen_u, exclude)    # D_G(x) of every non-member
    labels = np.concatenate([np.ones(len(members)), np.zeros(len(non_members))])
    scores = -np.concatenate([member_d, non_member_d])
    rows = [{"metric": "mia_auc", "setting": f"points={n_points}", "main_setting": True,
             "value": roc_auc(labels, scores), "std": np.nan, "n": len(gen),
             "note": f"{len(members)} train vs {len(non_members)} test"}]

    nearest_train = nearest_distance(gen_u, member_u)                # D_NN(g) of every generated trajectory
    for tau in sorted(set(privacy["uniqueness_sweep_m"]) | {privacy["uniqueness_m"]}):
        rows.append({"metric": "uniqueness", "setting": f"tau={tau}m", "main_setting": tau == privacy["uniqueness_m"],
                     "value": float(np.mean(nearest_train > tau)), "std": np.nan, "n": len(gen),
                     "note": f"nearest of {len(members)} train"})
    diagnostics = {"privacy_sample": {"generated": len(gen), "train": len(members), "test": len(non_members)},
                   "mia_self_matches_excluded": int(sum(len(v) for v in exclude.values()))}
    return rows, diagnostics
