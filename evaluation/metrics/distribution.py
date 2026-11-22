"""Metrics 1-4: global geographic, OD flow, trip distance and radius distributions, each scored with JSD.

Two extra metrics split the OD flow into its two halves: the start-point and end-point distributions over the grid cells
(start_jsd, end_jsd). The joint OD metric has about 61 million possible (start, end) cell pairs on the 0.005 degree
grid, so a few thousand trajectories rarely share one and even real data scores near 1; each half has only the grid's
cells, so it shows where trajectories begin and end.
"""
from __future__ import annotations

import numpy as np

from evaluation.metrics.grid import Grid
from evaluation.metrics.jsd import counts_by_key, histogram, jsd
from evaluation.metrics.trajset import TrajSet


def grids_from_settings(settings: dict) -> dict[float, Grid]:
    """Build one Grid per cell size in the grid sweep (the canonical size is always included)."""
    grid = settings["grid"]
    sizes = sorted(set(grid["sweep_deg"]) | {grid["cell_deg"]})
    return {size: Grid.from_settings(size, grid["lng_bounds"], grid["lat_bounds"]) for size in sizes}


def equal_size_draws(n_a: int, n_b: int, draws: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Index pairs that make two sets the same size before a JSD is computed.

    If the sets are already the same size, there is one draw with everything. Otherwise the larger set is sampled down
    to the smaller size `draws` times (without replacement, seeded), so a model is not helped or hurt by generating more
    or fewer trajectories than the real set has.
    """
    if n_a == n_b:
        return [(np.arange(n_a), np.arange(n_b))]
    n = min(n_a, n_b)
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(draws):
        a = np.sort(rng.choice(n_a, n, replace=False)) if n_a > n else np.arange(n_a)
        b = np.sort(rng.choice(n_b, n, replace=False)) if n_b > n else np.arange(n_b)
        result.append((a, b))
    return result


class _Features:
    """Everything the distribution metrics need from one TrajSet, computed once."""

    def __init__(self, trajectories: TrajSet, grids: dict[float, Grid]):
        """Compute point cells, start and end cells, OD keys, trip distances and radii for every trajectory."""
        self.counts = trajectories.counts
        self.point_cells = {size: g.cells(trajectories.lat, trajectories.lng) for size, g in grids.items()}
        first, last = trajectories.first_points(), trajectories.last_points()
        self.od, self.start, self.end = {}, {}, {}
        for size, g in grids.items():
            origin, destination = g.cells(*first), g.cells(*last)
            self.start[size], self.end[size] = origin, destination  # cell of each trajectory's first and last point
            self.od[size] = (origin + 1) * (g.size + 1) + (destination + 1)  # one number per OD pair, outside = 0
        self.distance_km = trajectories.lengths_m() / 1000.0
        self.radius_km = trajectories.radius_m() / 1000.0

    def point_mask(self, index: np.ndarray) -> np.ndarray:
        """Boolean mask over all points that keeps the points of the chosen trajectories."""
        chosen = np.zeros(len(self.counts), dtype=bool)
        chosen[index] = True
        return np.repeat(chosen, self.counts)


def distribution_scores(generated: TrajSet, real: TrajSet, settings: dict) -> tuple[list[dict], dict]:
    """Score metrics 1-4 for one generated set against the real test set.

    Returns score rows (main grid plus the grid sweep for the geographic, OD, start-point and end-point metrics) and
    diagnostics (out-of-grid rates). With equal-size draws, `value` and `std` are the mean and std over the draws.
    """
    grids = grids_from_settings(settings)
    canonical = settings["grid"]["cell_deg"]
    gen, ref = _Features(generated, grids), _Features(real, grids)
    draws = equal_size_draws(len(generated), len(real), settings["jsd_draws"], settings["seed"])
    trip, radius = settings["trip_distance"], settings["radius"]

    values = {}  # (metric, setting) -> list of JSD values, one per draw
    for gen_index, real_index in draws:
        gen_points, real_points = gen.point_mask(gen_index), ref.point_mask(real_index)
        for size, g in grids.items():
            p = np.bincount(ref.point_cells[size][real_points] + 1, minlength=g.size + 1)
            q = np.bincount(gen.point_cells[size][gen_points] + 1, minlength=g.size + 1)
            values.setdefault(("geo_jsd", size), []).append(jsd(p, q))
            values.setdefault(("od_jsd", size), []).append(jsd(*counts_by_key(ref.od[size][real_index],
                                                                              gen.od[size][gen_index])))
            for metric, real_cells, gen_cells in (("start_jsd", ref.start[size], gen.start[size]),
                                                  ("end_jsd", ref.end[size], gen.end[size])):
                p = np.bincount(real_cells[real_index] + 1, minlength=g.size + 1)  # outside the grid = bin 0
                q = np.bincount(gen_cells[gen_index] + 1, minlength=g.size + 1)
                values.setdefault((metric, size), []).append(jsd(p, q))
        values.setdefault(("distance_jsd", None), []).append(jsd(
            histogram(ref.distance_km[real_index], trip["bins"], trip["max_km"]),
            histogram(gen.distance_km[gen_index], trip["bins"], trip["max_km"])))
        values.setdefault(("radius_jsd", None), []).append(jsd(
            histogram(ref.radius_km[real_index], radius["bins"], radius["max_km"]),
            histogram(gen.radius_km[gen_index], radius["bins"], radius["max_km"])))

    n = len(draws[0][0])
    note = f"mean of {len(draws)} random samples" if len(draws) > 1 else "same size, no sampling"
    rows = []
    for (metric, size), draw_values in values.items():
        rows.append({
            "metric": metric,
            "setting": f"grid={size}" if size is not None else _bins_setting(metric, settings),
            "main_setting": size is None or size == canonical,
            "value": float(np.mean(draw_values)),
            "std": float(np.std(draw_values, ddof=1)) if len(draw_values) > 1 else np.nan,
            "n": n,
            "note": note,
        })
    diagnostics = {
        "out_of_grid_points_generated": float(np.mean(gen.point_cells[canonical] < 0)),
        "out_of_grid_points_real": float(np.mean(ref.point_cells[canonical] < 0)),
        "distance_overflow_generated": float(np.mean(gen.distance_km >= trip["max_km"])),
        "radius_overflow_generated": float(np.mean(gen.radius_km >= radius["max_km"])),
    }
    return rows, diagnostics


def _bins_setting(metric: str, settings: dict) -> str:
    """Describe the histogram used by the trip-distance or radius metric, e.g. "bins=1000,max=40.71km"."""
    hist = settings["trip_distance"] if metric == "distance_jsd" else settings["radius"]
    return f"bins={hist['bins']},max={hist['max_km']}km"
