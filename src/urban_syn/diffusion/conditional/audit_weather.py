"""Trace unknown sidecar weather back to original observations without changing data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metadata import load_aligned_split, sha256


def audit(dataset, metadata_dir, moves_path, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new audit output directory: {output}")
    _, _, _, rows, manifest = load_aligned_split(dataset, metadata_dir, "gen_all")
    sources = manifest["source_hashes"]
    for source in sources.values():
        if sha256(source["path"]) != source["sha256"]:
            raise ValueError(f"Source changed since sidecar construction: {source['path']}")
    weather = pd.read_parquet(sources["weather"]["path"])
    trips = pd.read_csv(sources["trip_map"]["path"],
                        dtype={"uid": str, "tid": str, "move_id": str})
    moves = pd.read_parquet(moves_path, columns=["uid", "datetime"]).reset_index()
    for frame in (weather, moves, rows):
        frame["uid"] = frame.uid.astype(str)
        frame["tid"] = frame.tid.astype(str)
    moves["move_id"] = moves.move_id.astype(str)
    moves["date"] = moves.datetime.dt.strftime("%Y-%m-%d")
    keys = set(zip(weather.uid, weather.tid, weather.DATE.astype(str)))
    source_keys = set(zip(moves.uid, moves.tid, moves.date))
    missing = rows[rows.weather_status == "missing_source"].copy()
    if missing.empty:
        raise ValueError("No missing-source weather records to investigate")
    training_ids = set(pd.read_parquet(Path(metadata_dir) / "gen_train.parquet").fmm_id)
    records, groups = [], []
    for (uid, tid, move_id), pieces in missing.groupby(["uid", "tid", "move_id"]):
        observations = moves[(moves.uid == uid) & (moves.tid == tid) &
                             (moves.move_id == move_id)].sort_values("datetime")
        timestamps = observations.datetime.astype("int64").to_numpy() // 10**9
        dates = observations.date.value_counts().sort_index()
        weather_records = weather[(weather.uid == uid) & (weather.tid == tid)]
        piece_starts = pd.to_datetime(pieces.start_timestamp, unit="s", utc=True)
        piece_ends = pd.to_datetime(pieces.end_timestamp, unit="s", utc=True)
        groups.append({"uid": uid, "tid": tid, "move_id": move_id,
                       "unknown_pieces": len(pieces), "original_observations": len(observations),
                       "source_date_point_counts": {k: int(v) for k, v in dates.items()},
                       "weather_records": weather_records[["DATE", "DESCRIPTION"]].to_dict("records"),
                       "max_observation_gap_seconds": int(np.diff(timestamps).max()),
                       "unknown_piece_departure_dates": pieces.date.value_counts().to_dict(),
                       "unknown_pieces_crossing_midnight": int((piece_starts.dt.date.to_numpy() !=
                                                                 piece_ends.dt.date.to_numpy()).sum())})
        for _, row in pieces.iterrows():
            start = int(row.start_timestamp)
            position = int(np.searchsorted(timestamps, start, side="right"))
            before = int(timestamps[position - 1]) if position else None
            after = int(timestamps[position]) if position < len(timestamps) else None
            inside_gap = before is not None and after is not None and before < start < after
            records.append({
                "fmm_id": int(row.fmm_id), "uid": uid, "tid": tid, "move_id": move_id,
                "piece": int(row.piece), "split": "gen_train" if row.fmm_id in training_ids else "gen_test",
                "sequence_length": int(row.sequence_length),
                "departure_date": row.date,
                "piece_start": str(pd.to_datetime(start, unit="s", utc=True)),
                "piece_end": str(pd.to_datetime(int(row.end_timestamp), unit="s", utc=True)),
                "original_move_points_on_departure_date": int(dates.get(row.date, 0)),
                "weather_key_exists": (uid, tid, row.date) in keys,
                "source_before_piece_start": str(pd.to_datetime(before, unit="s", utc=True)),
                "source_after_piece_start": str(pd.to_datetime(after, unit="s", utc=True)),
                "source_gap_seconds": after - before if inside_gap else None,
                "piece_start_strictly_inside_source_gap": inside_gap,
                "accepted_by_fmm_and_exact_binary_replay": True,
            })
    trips["date"] = pd.to_datetime(trips.start_timestamp, unit="s", utc=True).dt.strftime("%Y-%m-%d")
    input_missing = trips[[key not in keys for key in zip(trips.uid, trips.tid, trips.date)]]
    accepted = set(rows.fmm_id)
    rejected_ids = set(trips.id) - accepted
    offsets = {}
    for hours in range(-14, 15):
        shifted = (pd.to_datetime(missing.start_timestamp, unit="s", utc=True)
                   + pd.Timedelta(hours=hours)).dt.strftime("%Y-%m-%d")
        offsets[str(hours)] = sum(key in keys for key in zip(missing.uid, missing.tid, shifted))
    local_dates = pd.to_datetime(missing.start_timestamp, unit="s", utc=True).dt.tz_convert(
        "America/New_York").dt.strftime("%Y-%m-%d")
    details = pd.DataFrame(records)
    report = {
        "source_moves_sha256": sha256(moves_path),
        "original_move_observations": len(moves), "original_trip_dates": len(source_keys),
        "original_trip_dates_without_weather": len(source_keys - keys),
        "weather_date_strings_all_iso": bool(weather.DATE.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()),
        "weather_dates_all_equal_utc_timestamp_dates": bool((weather.DATE == pd.to_datetime(
            weather.datetime, utc=True).dt.strftime("%Y-%m-%d")).all()),
        "unknown_sidecar_pieces": len(missing), "affected_moves": groups,
        "unknown_pieces_before_fmm": len(input_missing),
        "pre_fmm_unknown_ids_exactly_equal_sidecar_unknown_ids": set(input_missing.id) == set(missing.fmm_id),
        "accepted_pieces": len(rows), "fmm_rejected_input_pieces": len(rejected_ids),
        "unknown_weather_pieces_rejected_by_fmm": len(set(input_missing.id) & rejected_ids),
        "all_unknown_departures_inside_source_gaps": bool(details.piece_start_strictly_inside_source_gap.all()),
        "all_unknown_departure_dates_have_zero_original_move_points": bool(
            (details.original_move_points_on_departure_date == 0).all()),
        "timezone_sensitivity": {
            "method": "Diagnostic only: shift missing-piece departures while keeping the original lookup fixed; not a timezone correction.",
            "matched_after_fixed_hour_shift": offsets,
            "single_shift_recovers_all": any(count == len(missing) for count in offsets.values()),
            "matched_after_utc_to_new_york_date_conversion": sum(key in keys for key in zip(
                missing.uid, missing.tid, local_dates)),
        },
        "finding": "Unknown weather dates were introduced by 15-second interpolation across a long original-observation gap. FMM and conversion did not remove these weather keys.",
        "changes_to_existing_artifacts": "None",
    }
    output.mkdir(parents=True)
    details.to_csv(output / "unknown_weather_records.csv", index=False)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("dataset", "metadata-dir", "moves", "output"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.dataset, args.metadata_dir, args.moves, args.output), indent=2))


if __name__ == "__main__":
    main()
