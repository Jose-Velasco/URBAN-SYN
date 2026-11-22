"""MobilityGPT: road sequences, one per test section, started from the section's first 1 or 3 roads."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from evaluation.geo import tstrajgen_road_midpoints
from evaluation.tables import Converted, gps_table, road_table

MODEL = "mobilitygpt"
NETWORK = "tstrajgen"  # MobilityGPT was prepared from TS-TrajGen's nyc.geo, so its roads are geo_ids


def convert(files: dict, options: dict) -> Converted:
    """Read MobilityGPT's generated JSON lines and its real test and training sections.

    Road IDs in all files are the workflow's prepared IDs; id_mapping.csv maps them back to nyc.geo geo_ids. Each
    generated record names the test section it started from (trajectory_id, e.g. "10948:0"), which becomes ref_id.
    Records are grouped as <checkpoint>_p<prompt roads>_s<seed>; options["groups"] picks which to keep (default "all").
    """
    mapping = pd.read_csv(files["id_mapping"])
    to_geo = dict(zip(mapping["new_id"].astype(int), mapping["original_id"].astype(int)))

    records = _read_records(files["generated"], options.get("groups", "all"))
    gen_roads = [[to_geo[int(x)] for x in rec["roads"]] for rec in records]
    real_ids, real_roads, real_times = _read_real(files["real_test"], to_geo)
    train_ids, train_roads, train_times = _read_real(files["real_train"], to_geo)

    used = {g for roads in gen_roads + real_roads + train_roads for g in roads}
    midpoints = tstrajgen_road_midpoints(files["geo"], only=used)
    gen_ids = [f"{MODEL}:{rec['group']}:{rec['trajectory_id']}" for rec in records]

    return Converted(
        model=MODEL,
        generated=gps_table(
            (t, rec["group"], rec["trajectory_id"], [midpoints[g] for g in roads], None)
            for t, rec, roads in zip(gen_ids, records, gen_roads)
        ),
        real=gps_table(
            (t, "real_test", None, [midpoints[g] for g in roads], s)
            for t, roads, s in zip(real_ids, real_roads, real_times)
        ),
        generated_roads=road_table(((t, rec["group"], r) for t, rec, r in zip(gen_ids, records, gen_roads)), NETWORK),
        real_roads=road_table(((t, "real_test", r) for t, r in zip(real_ids, real_roads)), NETWORK),
        train=gps_table(
            (t, "real_train", None, [midpoints[g] for g in roads], s)
            for t, roads, s in zip(train_ids, train_roads, train_times)
        ),
        unit="section",
        native="road",
        notes={"stopped right after the prompt (immediate_stop)": sum(bool(rec.get("immediate_stop")) for rec in records),
               "prompt roads": "included at the start of each generated trajectory",
               "time": "none in the generated file"},
    )


def _read_records(path, groups) -> list[dict]:
    """Read the generated JSON lines, add each record's group name and keep the requested groups."""
    keep = None if groups == "all" else set(groups)
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            rec["group"] = f"{rec['checkpoint']}_p{rec['prompt_length']}_s{rec['seed']}"
            if keep is None or rec["group"] in keep:
                records.append(rec)
    if not records:
        raise ValueError(f"No generated records left after choosing groups {groups}")
    return records


def _read_real(path, to_geo: dict) -> tuple[list[str], list[list[int]], list[np.ndarray]]:
    """Read a real sections CSV (test or train) into IDs, nyc.geo road IDs and times."""
    real = pd.read_csv(path)
    roads = [[to_geo[int(x)] for x in str(r).split(",")] for r in real["rid_list"]]
    times = [_timestamps_to_seconds(t) for t in real["time_list"]]
    return real["traj_id"].astype(str).to_list(), roads, times


def _timestamps_to_seconds(text: str) -> np.ndarray:
    """Turn the ISO timestamps of a real section's roads into seconds since its first road."""
    times = np.array(str(text).split(","), dtype="datetime64[ms]")
    return (times - times[0]).astype("timedelta64[ms]").astype(float) / 1000.0
