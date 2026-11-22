"""TS-TrajGen: road sequences, one per test trip, generated from that trip's start, destination and length."""
from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.geo import tstrajgen_road_midpoints
from evaluation.tables import Converted, collapse_repeats, gps_table, road_table

MODEL = "tstrajgen"
NETWORK = "tstrajgen"  # geo_id of baselines/gan/ts_trajgen/datasets/nyc/nyc.geo


def convert(files: dict, options: dict) -> Converted:
    """Read TS-TrajGen's generated CSV and its real test and training CSVs and return the common format.

    All three files have traj_id, rid_list and time_list; a generated row has the same traj_id as its real trip, so
    ref_id = traj_id. Roads are nyc.geo geo_ids, and each becomes one GPS point at its midpoint.
    """
    generated = pd.read_csv(files["generated"])
    gen_ids = generated["traj_id"].astype(str).to_numpy()
    gen_roads = [_ids(r) for r in generated["rid_list"]]
    gen_times = [_minutes_to_seconds(t) for t in generated["time_list"]]
    real_ids, real_roads, real_times = _read_real(files["real_test"])
    train_ids, train_roads, train_times = _read_real(files["real_train"])

    used = {g for roads in gen_roads + real_roads + train_roads for g in roads}
    midpoints = tstrajgen_road_midpoints(files["geo"], only=used)

    real_destination = {t: roads[-1] for t, roads in zip(real_ids, real_roads)}
    failed = sum(roads[-1] != real_destination.get(t) for t, roads in zip(gen_ids, gen_roads))
    return Converted(
        model=MODEL,
        generated=gps_table(
            (f"{MODEL}:{t}", "main", t, [midpoints[g] for g in roads], s)
            for t, roads, s in zip(gen_ids, gen_roads, gen_times)
        ),
        real=_midpoint_table(real_ids, real_roads, real_times, "real_test", midpoints),
        generated_roads=road_table(((f"{MODEL}:{t}", "main", r) for t, r in zip(gen_ids, gen_roads)), NETWORK),
        real_roads=road_table(((t, "real_test", r) for t, r in zip(real_ids, real_roads)), NETWORK),
        train=_midpoint_table(train_ids, train_roads, train_times, "real_train", midpoints),
        unit="move",
        native="road",
        notes={"did not reach the real destination road": failed,
               "real test and train": "map-matched roads (repeats collapsed), one midpoint per road",
               "time": "minute slots in the generated file, timestamps in the real files"},
    )


def _read_real(path) -> tuple[np.ndarray, list[list[int]], list[np.ndarray]]:
    """Read a real CSV (nyc_mm_test.csv or nyc_mm_train.csv) into IDs, roads and times.

    These files have one road per GPS point; consecutive repeats are collapsed, keeping each road's first timestamp.
    """
    real = pd.read_csv(path)
    ids = real["traj_id"].astype(str).to_numpy()
    roads, times = [], []
    for rid_list, time_list in zip(real["rid_list"], real["time_list"]):
        kept, index = collapse_repeats(_ids(rid_list))
        roads.append(kept)
        times.append(_timestamps_to_seconds(time_list, index))
    return ids, roads, times


def _midpoint_table(ids, roads, times, group: str, midpoints: dict):
    """Build the GPS table of a real set, one midpoint per road."""
    return gps_table((t, group, None, [midpoints[g] for g in r], s) for t, r, s in zip(ids, roads, times))


def _ids(text: str) -> list[int]:
    """Split a "1,2,3" road list into integers."""
    return [int(x) for x in str(text).split(",")]


def _minutes_to_seconds(text: str) -> np.ndarray:
    """Turn TS-TrajGen's minute-of-day slots into seconds since the first road (wraps at midnight)."""
    minutes = np.array([int(x) for x in str(text).split(",")])
    return ((minutes - minutes[0]) % 1440) * 60.0


def _timestamps_to_seconds(text: str, keep: list[int]) -> np.ndarray:
    """Turn ISO timestamps into seconds since the first one, keeping only the entries in `keep`."""
    times = np.array(str(text).split(","), dtype="datetime64[ms]")[keep]
    return (times - times[0]).astype("timedelta64[ms]").astype(float) / 1000.0
