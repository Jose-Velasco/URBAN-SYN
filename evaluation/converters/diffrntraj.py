"""Diff-RNTraj: unconditional road-network trajectories, as road ID + position."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

from evaluation.geo import point_on_polyline, prepare_polyline
from evaluation.tables import Converted, gps_table, road_table

MODEL = "diffrntraj"
NETWORK = "diffrntraj_drive"  # road tables store the FMM fid of nyc_drive.gpkg


def convert(files: dict, options: dict) -> Converted:
    """Read Diff-RNTraj's generated runs and its real test and training pieces.

    Each file key starting with "generated_" is one generation run (a folder of <length>.txt files) and becomes one
    group named after the rest of the key: generated_train_length -> group "train_length". Road IDs in all files are the
    model's own (1..N); new2raw_rid.json maps them to the FMM fid of nyc_drive.gpkg, which the road table stores. The
    model is unconditional, so generated trajectories have no ref_id.
    """
    new2raw = {int(k): int(v) for k, v in _read_json(files["new2raw"]).items()}
    rn_dict = _read_json(files["rn_dict"])
    runs = {key[len("generated_"):]: Path(path) for key, path in files.items() if key.startswith("generated_")}
    if not runs:
        raise ValueError("diffrntraj: list at least one generation run as generated_<name> under files:")
    generated, notes = [], {}
    for group, folder in runs.items():
        trajectories, dropped, hit = _read_generated(folder, new2raw, group)
        emptied = sum(1 for _, points, _ in trajectories if not points)
        if dropped and not options.get("allow_invalid_points", False):
            raise ValueError(
                f"diffrntraj ({group}): {dropped:,} generated points are at (0, 0) or on unknown road IDs, in {hit:,} "
                f"of {len(trajectories):,} trajectories ({emptied:,} with no valid point left). Scoring only the valid "
                "points would hide these failures: fix the generation, or add options: {allow_invalid_points: true} "
                "to diffrntraj in eval_config.yaml to score the valid points anyway (the counts stay in run.json).")
        generated += [(group, *t) for t in trajectories]
        notes[f"{group}: generated, invalid points (in trajectories, left empty)"] = \
            f"{len(trajectories):,}, {dropped:,} ({hit:,}, {emptied:,})"
    real = _read_pieces(files["real_test_eids"], files["real_test_rates"], rn_dict, new2raw)
    train = _read_pieces(files["real_train_eids"], files["real_train_rates"], rn_dict, new2raw)

    return Converted(
        model=MODEL,
        generated=gps_table((t, group, None, pts, None) for group, t, pts, _ in generated),
        real=gps_table((t, "real_test", None, pts, None) for t, pts, _ in real),
        generated_roads=road_table(((t, group, roads) for group, t, _, roads in generated), NETWORK),
        real_roads=road_table(((t, "real_test", roads) for t, _, roads in real), NETWORK),
        train=gps_table((t, "real_train", None, pts, None) for t, pts, _ in train),
        unit="piece",
        native="road",
        notes={**notes,
               "time": "none (not written in the generated files)",
               "pairing": "unconditional: no ref_id, needs the shared fallback pairing"},
    )


def _read_generated(folder: Path, new2raw: dict, group: str) -> tuple[list, int, int]:
    """Parse every <length>.txt file of one run into (traj_id, points, raw road IDs) tuples.

    Each trajectory starts with a "#, trajectory num :k" line (k restarts every batch, so it is not an ID), followed by
    one "lat, lng, rid, rate" line per point. Points at (0, 0) or on unknown road IDs are dropped; also returns how many
    were dropped and how many trajectories lost at least one.
    """
    trajectories, dropped, hit = [], 0, set()
    for path in sorted(folder.glob("*.txt"), key=lambda p: int(p.stem)):
        number = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("#"):
                    trajectories.append((f"{MODEL}:{group}:{path.stem}_{number}", [], []))
                    number += 1
                    continue
                lat, lng, rid, _rate = (float(x) for x in line.split(","))
                if (lat == 0 and lng == 0) or int(rid) not in new2raw:
                    dropped += 1
                    hit.add(len(trajectories) - 1)
                    continue
                trajectories[-1][1].append((lat, lng))
                trajectories[-1][2].append(new2raw[int(rid)])
    return trajectories, dropped, len(hit)


def _read_pieces(eids_path, rates_path, rn_dict: dict, new2raw: dict) -> list:
    """Decode real pieces (gen_test/ or gen_train/) into (traj_id, points, raw road IDs) tuples.

    eid_seqs.bin and rate_seqs.bin are pickled {length: [sequences]}. Rates use Diff-RNTraj's encoding: on a new road
    the rate is the position p on it, and on the same road again it is the share of the remaining road, so
    p = p_prev + (1 - p_prev) * rate. Each point is placed at p along its road's rn_dict.json geometry, as in rate2gps.
    """
    with open(eids_path, "rb") as handle:
        eids_by_length = pickle.load(handle)
    with open(rates_path, "rb") as handle:
        rates_by_length = pickle.load(handle)
    roads = {}  # road ID -> prepared polyline, built on first use

    trajectories = []
    for length in sorted(eids_by_length):
        for k, (eids, rates) in enumerate(zip(eids_by_length[length], rates_by_length[length])):
            points, previous, position = [], None, 0.0
            for eid, rate in zip(eids, rates):
                position = position + (1 - position) * rate if eid == previous else rate
                previous = eid
                if eid not in roads:
                    roads[eid] = prepare_polyline(rn_dict[str(eid)]["coords"])
                points.append(point_on_polyline(roads[eid], position))
            trajectories.append((f"{length}_{k}", points, [new2raw[e] for e in eids]))
    return trajectories


def _read_json(path) -> dict:
    """Read a JSON file."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
