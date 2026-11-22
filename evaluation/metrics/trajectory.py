"""Metrics 5-8: Hausdorff, normalized DTW, EDR and grid-cell Jaccard, averaged over real/generated pairs."""
from __future__ import annotations

import numpy as np

from evaluation.geo import EARTH_RADIUS_M
from evaluation.metrics.distribution import grids_from_settings
from evaluation.metrics.pairing import choose_pairs
from evaluation.metrics.trajset import TrajSet

# Pairs are processed in batches of similar length; a batch holds at most BATCH_CELLS (pair x point) cells per row of
# the dynamic programme, and at most MAX_BATCH pairs, which bounds memory use.
BATCH_CELLS = 15_000
MAX_BATCH = 512


def trajectory_scores(generated: TrajSet, real: TrajSet, settings: dict) -> tuple[list[dict], dict]:
    """Pair the generated set with the real test set and score metrics 5-8 (with the EDR and Jaccard sweeps)."""
    gen_index, real_index, info = choose_pairs(generated, real, settings)
    if len(gen_index) == 0:
        return [], {**info, "real_trajectories_used": 0}
    eps_list = sorted(set(settings["edr"]["sweep_m"]) | {settings["edr"]["eps_m"]})
    per_pair = pair_metrics(real, generated, real_index, gen_index, eps_list, grids_from_settings(settings))

    n = len(gen_index)
    note = f"{info['pairing']} pairs" + ("" if info["pairing"] == "id" else
                                         f", {100 * info['paired_share']:.1f}% of the set has a match")
    rows = [
        {"metric": "hausdorff_km", "setting": "", "main_setting": True, "value": float(per_pair["hausdorff_km"].mean())},
        {"metric": "ndtw_km", "setting": "", "main_setting": True, "value": float(per_pair["ndtw_km"].mean())},
    ]
    for eps in eps_list:
        rows.append({"metric": "edr", "setting": f"eps={eps}m", "main_setting": eps == settings["edr"]["eps_m"],
                     "value": float(per_pair["edr"][eps].mean())})
    for size, values in per_pair["jaccard"].items():
        rows.append({"metric": "jaccard", "setting": f"grid={size}",
                     "main_setting": size == settings["grid"]["cell_deg"], "value": float(values.mean())})
    for row in rows:
        row.update(std=np.nan, n=n, note=note)
    diagnostics = {**info, "real_trajectories_used": int(len(np.unique(real_index)))}
    return rows, diagnostics


def pair_metrics(real: TrajSet, generated: TrajSet, real_index, gen_index, eps_list, grids) -> dict:
    """Hausdorff (km), nDTW (km), EDR per threshold and Jaccard per grid for every pair.

    Pair p is real trajectory real_index[p] with generated trajectory gen_index[p].
    Returns {"hausdorff_km": (P,), "ndtw_km": (P,), "edr": {eps: (P,)}, "jaccard": {cell size: (P,)}}.
    """
    real_index, gen_index = np.asarray(real_index), np.asarray(gen_index)
    n, m = real.counts[real_index], generated.counts[gen_index]
    hausdorff, ndtw = np.empty(len(n)), np.empty(len(n))
    edr = np.empty((len(eps_list), len(n)))

    order = np.argsort(n + m, kind="stable")  # similar lengths together, so batches need little padding
    for batch in _batches(order, n, m):
        r_pts = [real.points(k) for k in real_index[batch]]
        g_pts = [generated.points(k) for k in gen_index[batch]]
        h, cost, steps, edit = _dynamic_programme(r_pts, g_pts, eps_list)
        hausdorff[batch] = h / 1000.0
        ndtw[batch] = cost / steps / 1000.0
        edr[:, batch] = edit / np.maximum(n[batch], m[batch])

    jaccard = {size: _jaccard(real, generated, real_index, gen_index, grid) for size, grid in grids.items()}
    return {"hausdorff_km": hausdorff, "ndtw_km": ndtw,
            "edr": {eps: edr[e] for e, eps in enumerate(eps_list)}, "jaccard": jaccard}


def _batches(order: np.ndarray, n: np.ndarray, m: np.ndarray):
    """Split the pair order into batches whose padded size stays within BATCH_CELLS."""
    batch, longest = [], 0
    for p in order:
        longest_if_added = max(longest, n[p], m[p])
        if batch and (len(batch) + 1) * (longest_if_added + 1) > BATCH_CELLS or len(batch) == MAX_BATCH:
            yield np.array(batch)
            batch, longest_if_added = [], max(n[p], m[p])
        batch.append(p)
        longest = longest_if_added
    if batch:
        yield np.array(batch)


def _dynamic_programme(r_pts: list, g_pts: list, eps_list) -> tuple[np.ndarray, ...]:
    """Run Hausdorff, DTW and EDR for a batch of pairs in one pass over the distance matrix.

    r_pts and g_pts are lists of (lat, lng) arrays. The (n x m) Haversine distance matrix is never stored: cells are
    visited one anti-diagonal (i + j = s) at a time, so numpy handles a whole diagonal of every pair at once. Shorter
    pairs are padded; padded cells never feed a real cell, since a cell only depends on cells with smaller i and j.

    DTW:  D(i, j) = d(i, j) + min(D(i-1, j), D(i, j-1), D(i-1, j-1)), D(0, 0) = 0. The warping path length |W| is
          carried along; when predecessors tie on cost, the shortest path is kept.
    EDR:  E(i, j) = min(E(i, j-1) + 1, E(i-1, j) + 1, E(i-1, j-1) + [d(i, j) >= eps]), E(i, 0) = i, E(0, j) = j
          (a match needs d < eps, as in TS-TrajGen).
    Hausdorff: running minimum of every row and column of the distance matrix.

    Returns Hausdorff (m), DTW cost (m), DTW path length and EDR distance (eps x pairs).
    """
    B = len(r_pts)
    n = np.array([len(lat) for lat, _ in r_pts])
    m = np.array([len(lat) for lat, _ in g_pts])
    N, M = int(n.max()), int(m.max())
    r_lat, r_lng = _padded(r_pts, N)
    g_lat, g_lng = _padded(g_pts, M)
    # The generated points are stored back to front, so one diagonal (i up, j down) is a plain slice of both arrays.
    g_lat, g_lng = g_lat[:, ::-1].copy(), g_lng[:, ::-1].copy()
    r_cos, g_cos = np.cos(r_lat), np.cos(g_lat)
    eps = np.asarray(eps_list, dtype=float)[:, None, None]
    E = len(eps_list)

    # Three rotating buffers per quantity hold diagonals s-2, s-1 and s, indexed by i = 0..N.
    cost = np.full((3, B, N + 1), np.inf)
    steps = np.zeros((3, B, N + 1))
    edit = np.full((3, E, B, N + 1), np.inf)
    cost[0, :, 0] = 0.0                          # diagonal 0: cell (0, 0)
    edit[0, :, :, 0] = 0.0
    edit[1, :, :, 0] = edit[1, :, :, 1] = 1.0    # diagonal 1: cells (0, 1) and (1, 0)
    row_min = np.full((B, N), np.inf)
    col_min = np.full((B, M), np.inf)            # also back to front, like the generated points
    out_cost, out_steps, out_edit = np.empty(B), np.empty(B), np.empty((E, B))
    finishing = {}
    for b, s in enumerate(n + m):
        finishing.setdefault(int(s), []).append(b)

    for s in range(2, N + M + 1):
        two, one, cur = (s - 2) % 3, (s - 1) % 3, s % 3
        lo, hi = max(1, s - M), min(N, s - 1)    # cells (i, s - i) for i = lo..hi
        here, before = slice(lo, hi + 1), slice(lo - 1, hi)            # index i and i - 1
        g = slice(M - s + lo, M - s + hi + 1)                          # j - 1 = s - i - 1, back to front
        d = _haversine(r_lat[:, before], r_lng[:, before], r_cos[:, before],
                       g_lat[:, g], g_lng[:, g], g_cos[:, g])         # (B, cells on this diagonal)

        up, left, diag = cost[one][:, before], cost[one][:, here], cost[two][:, before]
        best = np.minimum(np.minimum(up, left), diag)
        length = np.where(diag == best, steps[two][:, before], np.inf)
        length = np.minimum(length, np.where(up == best, steps[one][:, before], np.inf))
        length = np.minimum(length, np.where(left == best, steps[one][:, here], np.inf))
        cost[cur][:, 0] = np.inf                 # boundary cell (0, s)
        cost[cur][:, here] = best + d
        steps[cur][:, here] = length + 1
        if s <= N:
            cost[cur][:, s] = np.inf             # boundary cell (s, 0)

        edit[cur][:, :, 0] = s
        edit[cur][:, :, here] = np.minimum(np.minimum(edit[one][:, :, before], edit[one][:, :, here]) + 1,
                                           edit[two][:, :, before] + (d[None] >= eps))
        if s <= N:
            edit[cur][:, :, s] = s

        i = np.arange(lo, hi + 1)
        inside = (i[None, :] <= n[:, None]) & ((s - i)[None, :] <= m[:, None])  # not a padded cell
        d_inside = np.where(inside, d, np.inf)
        np.minimum(row_min[:, before], d_inside, out=row_min[:, before])
        np.minimum(col_min[:, g], d_inside, out=col_min[:, g])

        done = finishing.get(s)
        if done is not None:  # pairs whose last cell (n, m) is on this diagonal
            done = np.array(done)
            out_cost[done] = cost[cur][done, n[done]]
            out_steps[done] = steps[cur][done, n[done]]
            out_edit[:, done] = edit[cur][:, done, n[done]]

    col_min = col_min[:, ::-1]
    row_max = np.where(np.arange(N)[None, :] < n[:, None], row_min, -np.inf).max(axis=1)
    col_max = np.where(np.arange(M)[None, :] < m[:, None], col_min, -np.inf).max(axis=1)
    return np.maximum(row_max, col_max), out_cost, out_steps, out_edit


def _padded(points: list, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Stack (lat, lng) arrays into (batch, length) arrays in radians, padding with zeros."""
    lat = np.zeros((len(points), length))
    lng = np.zeros((len(points), length))
    for b, (la, ln) in enumerate(points):
        lat[b, :len(la)] = la
        lng[b, :len(ln)] = ln
    return np.radians(lat), np.radians(lng)


def _haversine(lat1, lng1, cos1, lat2, lng2, cos2) -> np.ndarray:
    """Haversine distance in metres for coordinates already in radians (same formula as geo.haversine_m)."""
    h = np.sin((lat2 - lat1) / 2) ** 2 + cos1 * cos2 * np.sin((lng2 - lng1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def _jaccard(real: TrajSet, generated: TrajSet, real_index, gen_index, grid) -> np.ndarray:
    """Grid-cell Jaccard of each pair: shared unique cells over all unique cells visited by either."""
    real_cells = grid.cells(real.lat, real.lng)
    gen_cells = grid.cells(generated.lat, generated.lng)
    values = np.empty(len(real_index))
    for p, (r, g) in enumerate(zip(real_index, gen_index)):
        a = np.unique(real_cells[real.offsets[r]:real.offsets[r + 1]])
        b = np.unique(gen_cells[generated.offsets[g]:generated.offsets[g + 1]])
        shared = len(np.intersect1d(a, b, assume_unique=True))
        values[p] = shared / (len(a) + len(b) - shared)
    return values
