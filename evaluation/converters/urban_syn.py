"""URBAN-SYN (GNN road encoder + Diff-RNTraj): unconditional road-network trajectories, as road ID + position."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.geo import haversine_m, point_on_polyline, tstrajgen_road_polylines
from evaluation.tables import Converted, gps_table, road_table

MODEL = "urban_syn"
NETWORK = "tstrajgen"  # edge_id of the shared nyc.gpkg, which TS-TrajGen's nyc.geo numbers the same way (geo_id)

GENERATED_COLUMNS = ["trajectory_id", "step", "edge_id", "position_rate", "lng", "lat"]
SEQUENCE_COLUMNS = ["fmm_id", "edge_ids", "rates"]
# The generator places each point on its road's nyc.gpkg geometry; placed again on nyc.geo (built from the same file),
# it must land in the same spot. A larger gap means the model used a different road network.
MAX_REBUILD_M = 1.0


def convert(files: dict, options: dict) -> Converted:
    """Read URBAN-SYN's generated runs and its real test and training pieces and return the common format.

    Each file key starting with "generated_" is one generation run (the generated/*.parquet of one experiment run) and
    becomes one group named after the rest of the key: generated_finetune -> group "finetune". A generated file has one
    row per point (trajectory_id, step, model_road_id, edge_id, rate, position_rate, lng, lat); edge_id is the shared
    road network's ID, so the road tables use it directly, and lat/lng are the points the generator placed on those
    roads. The real pieces are the prepared road sequences the model was trained and tested on (fmm_id, edge_ids,
    rates); each point is placed on its road's nyc.geo geometry with Diff-RNTraj's rate encoding, as the generator does.
    The model is unconditional, so generated trajectories have no ref_id.
    """
    runs = {key[len("generated_"):]: Path(path) for key, path in files.items() if key.startswith("generated_")}
    if not runs:
        raise ValueError("urban_syn: list at least one generation run as generated_<name> under files:")
    generated = {group: _read_generated(path) for group, path in runs.items()}
    real_ids, real_roads, real_positions = _read_sequences(files["real_test"])
    train_ids, train_roads, train_positions = _read_sequences(files["real_train"])

    used = set(np.concatenate([g["edge_id"].to_numpy() for g in generated.values()]).tolist())
    used |= {e for roads in real_roads + train_roads for e in roads.tolist()}
    polylines = tstrajgen_road_polylines(files["geo"], only=used)
    unknown = used - set(polylines)
    if unknown:
        raise ValueError(
            f"urban_syn: {len(unknown):,} road IDs are not in nyc.geo (e.g. {sorted(unknown)[:5]}): the model's road "
            "network is not the shared network TS-TrajGen was built on. Rebuild one of them from the same nyc.gpkg.")

    notes, gen_gps, gen_roads = {}, [], []
    for group, frame in generated.items():
        gap_m = _rebuild_gap_m(frame, polylines)
        if gap_m.max() > MAX_REBUILD_M:
            raise ValueError(
                f"urban_syn ({group}): {int((gap_m > MAX_REBUILD_M).sum()):,} of {len(gap_m):,} generated points are "
                f"more than {MAX_REBUILD_M:g} m from where their edge_id and position_rate put them on nyc.geo "
                f"(largest {gap_m.max():,.0f} m): the model generated on a different road network than the one "
                "TS-TrajGen was built on. Use the same nyc.gpkg for both.")
        for trajectory_id, part in _trajectories(frame):
            traj_id = f"{MODEL}:{group}:{trajectory_id}"
            gen_gps.append((traj_id, group, None, part[["lat", "lng"]].to_numpy(float), None))
            gen_roads.append((traj_id, group, part["edge_id"].to_numpy()))
        notes[f"{group}: generated trajectories, points"] = f"{frame['trajectory_id'].nunique():,}, {len(frame):,}"
        notes[f"{group}: largest gap to nyc.geo"] = f"{gap_m.max():.3f} m"

    return Converted(
        model=MODEL,
        generated=gps_table(gen_gps),
        real=_gps(real_ids, real_roads, real_positions, "real_test", polylines),
        generated_roads=road_table(gen_roads, NETWORK),
        real_roads=road_table(((t, "real_test", r) for t, r in zip(real_ids, real_roads)), NETWORK),
        train=_gps(train_ids, train_roads, train_positions, "real_train", polylines),
        unit="piece",
        native="road",
        notes={**notes,
               "time": "none (not written in the generated file)",
               "pairing": "unconditional: no ref_id, needs the shared fallback pairing"},
    )


def _read_generated(path: Path) -> pd.DataFrame:
    """Read one generated Parquet file, sorted by trajectory and step."""
    frame = pd.read_parquet(path)
    missing = [c for c in GENERATED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; expected the generator's {GENERATED_COLUMNS}")
    frame = frame[GENERATED_COLUMNS].sort_values(["trajectory_id", "step"], kind="stable").reset_index(drop=True)
    frame["edge_id"] = frame["edge_id"].astype("int64")
    if frame.empty:
        raise ValueError(f"{path}: no generated trajectories")
    return frame


def _trajectories(frame: pd.DataFrame):
    """(trajectory_id, rows) for each generated trajectory of a sorted frame, in order."""
    ids = frame["trajectory_id"].to_numpy()
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    for start, end in zip(starts, np.r_[starts[1:], len(frame)]):
        yield ids[start], frame.iloc[start:end]


def _rebuild_gap_m(frame: pd.DataFrame, polylines: dict) -> np.ndarray:
    """Distance (m) from each generated point to where its edge_id and position_rate put it on nyc.geo."""
    rebuilt = np.array([point_on_polyline(polylines[int(e)], float(p))
                        for e, p in zip(frame["edge_id"], frame["position_rate"])])
    return haversine_m(frame["lat"].to_numpy(float), frame["lng"].to_numpy(float), rebuilt[:, 0], rebuilt[:, 1])


def _read_sequences(path) -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
    """Read prepared road sequences (fmm_id, edge_ids, rates) into IDs, road IDs and positions along each road.

    Rates use Diff-RNTraj's encoding: on a new road the rate is the position p on it, and on the same road again it is
    the share of the remaining road, so p = p_prev + (1 - p_prev) * rate.
    """
    frame = pd.read_parquet(path, columns=SEQUENCE_COLUMNS)
    ids, roads, positions = [], [], []
    for fmm_id, edge_ids, rates in frame.itertuples(index=False, name=None):
        edge_ids = np.asarray(edge_ids, dtype="int64")
        rates = np.asarray(rates, dtype=float)
        if len(edge_ids) != len(rates):
            raise ValueError(f"{path}: trajectory {fmm_id} has {len(edge_ids)} roads but {len(rates)} rates")
        position, previous, decoded = 0.0, None, np.empty(len(rates))
        for k, (edge_id, rate) in enumerate(zip(edge_ids, rates)):
            position = position + (1 - position) * rate if edge_id == previous else rate
            previous = edge_id
            decoded[k] = position
        ids.append(str(fmm_id))
        roads.append(edge_ids)
        positions.append(decoded)
    return ids, roads, positions


def _gps(ids, roads, positions, group: str, polylines: dict) -> pd.DataFrame:
    """GPS table of a real set: each point placed at its position along its road's nyc.geo geometry."""
    return gps_table(
        (t, group, None, [point_on_polyline(polylines[int(e)], p) for e, p in zip(r, pos)], None)
        for t, r, pos in zip(ids, roads, positions))
