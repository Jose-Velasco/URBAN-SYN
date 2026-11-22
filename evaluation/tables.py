"""The common format: one GPS table and one road table per source, plus simple checks."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from evaluation.geo import haversine_m

# GPS table: one row per point.
#   traj_id  unique trajectory ID, "<model>:<native id>" for generated data
#   group    run group, e.g. a MobilityGPT checkpoint/prompt/seed; "main" for a single run, "real_test" for the test set
#   ref_id   traj_id of the real test trajectory this one was generated from (None if unconditional)
#   seq      point order within the trajectory, starting at 0
#   lat, lng WGS84 coordinates
#   t_s      seconds since the trajectory's first point (NaN when the model has no time)
GPS_COLUMNS = ["traj_id", "group", "ref_id", "seq", "lat", "lng", "t_s"]

# Road table: one row per road visit, in order.
#   edge_id  road ID on the network named in `network`
#   network  which road network edge_id belongs to (until everything is mapped onto one network)
ROAD_COLUMNS = ["traj_id", "group", "seq", "edge_id", "network"]


@dataclass
class Converted:
    """Everything one converter returns for one model."""

    model: str
    generated: pd.DataFrame                      # GPS table of the generated trajectories
    real: pd.DataFrame                           # GPS table of the real test set, at the model's unit
    generated_roads: pd.DataFrame | None = None  # road table; None until FMM matches exist (GPS models)
    real_roads: pd.DataFrame | None = None
    train: pd.DataFrame | None = None            # GPS table of the real training set (group "real_train")
    unit: str = ""                               # what one trajectory is (trip, move, piece, section)
    native: str = "gps"                          # what the model writes: "gps" points or "road" IDs
    notes: dict = field(default_factory=dict)    # counts and caveats printed in the summary


def gps_table(trajectories) -> pd.DataFrame:
    """Build a GPS table from (traj_id, group, ref_id, points, t_s) tuples.

    points is an (n, 2) array of lat, lng; t_s is a length-n array of seconds, or None. Trajectories with no points are
    skipped. The table is built in one go at the end, which is much faster than one small table per trajectory.
    """
    ids, groups, refs, counts, points, times = [], [], [], [], [], []
    for traj_id, group, ref_id, pts, t_s in trajectories:
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        if len(pts) == 0:
            continue
        if t_s is not None and len(t_s) != len(pts):
            raise ValueError(f"{traj_id}: {len(pts)} points but {len(t_s)} times")
        ids.append(traj_id)
        groups.append(group)
        refs.append(ref_id)
        counts.append(len(pts))
        points.append(pts)
        times.append(np.full(len(pts), np.nan) if t_s is None else np.asarray(t_s, dtype=float))
    if not ids:
        return pd.DataFrame(columns=GPS_COLUMNS)
    counts = np.asarray(counts)
    points = np.concatenate(points)
    return pd.DataFrame({
        "traj_id": _repeat(ids, counts),
        "group": _repeat(groups, counts),
        "ref_id": _repeat(refs, counts),
        "seq": _sequence_numbers(counts),
        "lat": points[:, 0],
        "lng": points[:, 1],
        "t_s": np.concatenate(times),
    })


def road_table(sequences, network: str) -> pd.DataFrame:
    """Build a road table from (traj_id, group, edge_ids) tuples on one named network."""
    ids, groups, counts, edges = [], [], [], []
    for traj_id, group, edge_ids in sequences:
        edge_ids = np.asarray(edge_ids, dtype="int64")
        if len(edge_ids) == 0:
            continue
        ids.append(traj_id)
        groups.append(group)
        counts.append(len(edge_ids))
        edges.append(edge_ids)
    if not ids:
        return pd.DataFrame(columns=ROAD_COLUMNS)
    counts = np.asarray(counts)
    return pd.DataFrame({
        "traj_id": _repeat(ids, counts),
        "group": _repeat(groups, counts),
        "seq": _sequence_numbers(counts),
        "edge_id": np.concatenate(edges),
        "network": network,
    })


def collapse_repeats(values) -> tuple[list, list[int]]:
    """Remove consecutive duplicates, e.g. one road ID per GPS point -> one entry per road.

    Returns the kept values and the index of each in the input (so matching timestamps can be picked out).
    """
    kept, index = [], []
    for i, value in enumerate(values):
        if not kept or value != kept[-1]:
            kept.append(value)
            index.append(i)
    return kept, index


def validate(converted: Converted) -> list[str]:
    """Run basic checks on a converter's output and return a list of problems (empty = all good)."""
    problems = []
    for name in ("generated", "real", "train"):
        table = getattr(converted, name)
        if table is not None:
            problems += [f"{name}: {p}" for p in _check_gps(table)]
    for name in ("generated_roads", "real_roads"):
        table = getattr(converted, name)
        if table is not None and list(table.columns) != ROAD_COLUMNS:
            problems.append(f"{name}: columns are {list(table.columns)}, expected {ROAD_COLUMNS}")
    missing = unmatched_refs(converted)
    if missing:
        problems.append(f"{missing} generated trajectories point to a ref_id that is not in the real test set")
    elif converted.generated["ref_id"].notna().any():
        distance_km = paired_start_distance_km(converted)
        if distance_km > MAX_PAIRED_START_KM:
            problems.append(
                f"generated trajectories start a median {distance_km:.1f} km from the real trajectory they are "
                f"paired with (limit {MAX_PAIRED_START_KM:g} km): the generated and real test files are probably "
                "from different data splits or runs. Regenerate the model's output for the current test set, "
                "or point eval_config.yaml at the matching real files")
    return problems


# A generated trajectory starts near the real one it was generated from (TS-TrajGen and MobilityGPT on the same road,
# DiffTraj in the same region). If the median start distance of the ID pairs is above this, the pairing is almost
# certainly broken.
MAX_PAIRED_START_KM = 5.0


def paired_start_distance_km(converted: Converted) -> float:
    """Median distance (km) between the first point of each ID-paired generated trajectory and its real one."""
    gen = converted.generated[converted.generated["ref_id"].notna()]
    gen_first = gen.loc[gen.groupby("traj_id", sort=False)["seq"].idxmin()]
    real_first = converted.real.loc[converted.real.groupby("traj_id", sort=False)["seq"].idxmin()].set_index("traj_id")
    real_first = real_first.loc[gen_first["ref_id"].astype(str)]
    distance = haversine_m(gen_first["lat"].to_numpy(), gen_first["lng"].to_numpy(),
                           real_first["lat"].to_numpy(), real_first["lng"].to_numpy())
    return float(np.median(distance)) / 1000.0


def unmatched_refs(converted: Converted) -> int:
    """Count generated trajectories whose ref_id is set but not found among the real traj_ids."""
    refs = converted.generated.drop_duplicates("traj_id")["ref_id"].dropna()
    return int((~refs.isin(set(converted.real["traj_id"]))).sum())


def summarize(table: pd.DataFrame | None) -> str:
    """One-line description of a table: trajectories, rows and groups."""
    if table is None:
        return "not available yet"
    groups = table["group"].nunique()
    return (f"{table['traj_id'].nunique():,} trajectories, {len(table):,} rows"
            + (f", {groups} groups" if groups > 1 else ""))


def _check_gps(table: pd.DataFrame) -> list[str]:
    """Check one GPS table: columns, coordinates and point order."""
    if list(table.columns) != GPS_COLUMNS:
        return [f"columns are {list(table.columns)}, expected {GPS_COLUMNS}"]
    problems = []
    if table[["lat", "lng"]].isna().any().any():
        problems.append("some points have no coordinates")
    if not table["lat"].between(-90, 90).all() or not table["lng"].between(-180, 180).all():
        problems.append("some coordinates are out of range (lat/lng swapped?)")
    if not (table.groupby("traj_id", sort=False).cumcount().to_numpy() == table["seq"].to_numpy()).all():
        problems.append("seq does not run 0, 1, 2, ... within every trajectory")
    return problems


def _repeat(values: list, counts: np.ndarray) -> np.ndarray:
    """Repeat each trajectory-level value once per row of that trajectory."""
    return np.repeat(np.asarray(values, dtype=object), counts)


def _sequence_numbers(counts: np.ndarray) -> np.ndarray:
    """Return 0, 1, ..., n-1 for each trajectory, all in one array."""
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    return np.arange(counts.sum()) - starts
