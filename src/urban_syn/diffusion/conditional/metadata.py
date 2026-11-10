"""Weather-only v1 predecessor; use sparse_metadata.py for current multi-field sidecars."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from .conversion import encode_rates, parse_float_sequence, parse_int_sequence

SPLITS = ("gen_all", "gen_train", "gen_test")
UNKNOWN = "<unknown>"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_sequences(dataset, split):
    # These are the user's trusted local converter outputs, not arbitrary uploads.
    def read(name):
        with (Path(dataset) / split / name).open("rb") as stream:
            return pickle.load(stream)
    return read("eid_seqs.bin"), read("rate_seqs.bin")


def replay_alignment(dataset, trip_map, fmm_matches, test_trips):
    """Replay stored road IDs and raw rates before training normalization, or fail closed.

    Split, length bucket, and bucket index are preserved; enrichment and encoder
    preprocessing happen only after this comparison succeeds.
    """
    dataset = Path(dataset)
    identities = pd.read_csv(trip_map, dtype={"uid": str, "tid": str, "move_id": str})
    required = {"id", "uid", "tid", "move_id", "piece",
                "start_timestamp", "end_timestamp", "num_points"}
    if not required.issubset(identities.columns) or identities[list(required)].isna().any().any():
        raise ValueError("Trajectory map lacks complete required identity fields")
    if identities.id.duplicated().any() or identities.duplicated(["uid", "tid", "move_id", "piece"]).any():
        raise ValueError("Duplicate trajectory identity in source map")
    identities = identities.set_index("id").to_dict("index")
    test = pd.read_csv(test_trips, dtype={"uid": str, "tid": str})
    if test.empty or test[["uid", "tid"]].isna().any().any():
        raise ValueError("A nonempty, complete held-out trip list is required")
    test_keys = set(zip(test.uid, test.tid))
    mapping = json.loads((dataset / "extra_file/raw2new_rid.json").read_text())
    valid_fids = set(json.loads((dataset / "extra_file/raw_rn_dict.json").read_text()))
    with Path(fmm_matches).open() as stream:
        header = stream.readline()
    delimiter = next((d for d in (";", "\t", "|") if d in header), ",")
    matches = pd.read_csv(fmm_matches, sep=delimiter)
    expected = {split: defaultdict(list) for split in SPLITS}
    skipped = Counter()
    seen = set()
    for _, match in matches.iterrows():
        raw = parse_int_sequence(match.opath)
        offsets = parse_float_sequence(match.offset)
        lengths = parse_float_sequence(match.length)
        if not raw:
            skipped["empty_opath"] += 1
            continue
        if not len(raw) == len(offsets) == len(lengths):
            skipped["length_mismatch"] += 1
            continue
        if any(str(fid) not in valid_fids for fid in raw):
            skipped["unknown_fid"] += 1
            continue
        source_id = int(match.id)
        if source_id in seen or source_id not in identities:
            raise ValueError(f"Duplicate or missing accepted identity: {source_id}")
        seen.add(source_id)
        meta = identities[source_id]
        if len(raw) != int(meta["num_points"]):
            raise ValueError(f"Source/FMM point count mismatch: {source_id}")
        # Use the saved road mapping: never regenerate road IDs or graph artifacts.
        ids = [mapping[str(fid)] for fid in raw]
        rates = encode_rates(raw, offsets, lengths)
        row = {"fmm_id": source_id, **meta}
        row["date"] = pd.to_datetime(int(meta["start_timestamp"]), unit="s", utc=True).strftime("%Y-%m-%d")
        split = "gen_test" if (meta["uid"], meta["tid"]) in test_keys else "gen_train"
        expected["gen_all"][len(raw)].append((ids, rates, row))
        expected[split][len(raw)].append((ids, rates, row))

    aligned = {}
    parent_keys = {}
    for split, buckets in expected.items():
        stored_ids, stored_rates = read_sequences(dataset, split)
        if buckets.keys() != stored_ids.keys() or buckets.keys() != stored_rates.keys():
            raise ValueError(f"{split}: replay length buckets differ from stored binaries")
        rows = []
        for length in sorted(buckets):
            records = buckets[length]
            if len(records) != len(stored_ids[length]) or len(records) != len(stored_rates[length]):
                raise ValueError(f"{split}/{length}: replay bucket count differs")
            for index, (ids, rates, meta) in enumerate(records):
                if ids != stored_ids[length][index] or rates != stored_rates[length][index]:
                    raise ValueError(f"{split}/{length}/{index}: exact conversion replay failed")
                rows.append({"split": split, "sequence_length": length,
                             "index_within_length": index, **meta})
        if not rows:
            raise ValueError(f"{split}: empty split")
        aligned[split] = pd.DataFrame(rows)
        parent_keys[split] = set(zip(aligned[split].uid, aligned[split].tid))
    if parent_keys["gen_train"] & parent_keys["gen_test"]:
        raise ValueError("Parent trip crosses train/test boundary")
    if len(aligned["gen_all"]) != len(aligned["gen_train"]) + len(aligned["gen_test"]):
        raise ValueError("Combined split count differs")
    return aligned, dict(skipped)


def join_weather(rows, weather):
    """Select departure-date weather using the upstream stored-date convention."""
    keys = ["uid", "tid", "date"]
    weather = weather.copy()
    if weather[["uid", "tid", "DATE"]].isna().any().any():
        raise ValueError("Weather identity/date contains missing values")
    weather["uid"] = weather.uid.astype(str)
    weather["tid"] = weather.tid.astype(str)
    weather["date"] = weather.DATE.astype(str)
    if weather.duplicated(keys).any():
        raise ValueError("Weather keys (uid, tid, DATE) are not unique")
    if "datetime" in weather:
        dates = pd.to_datetime(weather["datetime"], utc=True).dt.strftime("%Y-%m-%d")
        if not (dates == weather.date).all():
            raise ValueError("Weather DATE differs from its UTC timestamp date")
    joined = rows.merge(weather[keys + ["DESCRIPTION"]], on=keys, how="left",
                        validate="many_to_one", sort=False)
    joined["weather_category"] = joined.DESCRIPTION.astype("string").str.strip().replace("", pd.NA)
    joined["weather_status"] = "matched"
    joined.loc[joined.weather_category.isna(), "weather_status"] = "missing_source"
    joined["weather_category"] = joined.weather_category.fillna(UNKNOWN)
    return joined.drop(columns="DESCRIPTION")


def build_sidecars(dataset, trip_map, fmm_matches, test_trips, weather_path, output):
    """Write schema-v1 weather sidecars with a training-only vocabulary and hashes."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new sidecar output directory: {output}")
    aligned, skipped = replay_alignment(dataset, trip_map, fmm_matches, test_trips)
    weather = pd.read_parquet(weather_path)
    joined = {split: join_weather(rows, weather) for split, rows in aligned.items()}
    categories = sorted(set(joined["gen_train"].weather_category) - {UNKNOWN})
    if len(categories) < 2:
        raise ValueError("Pilot needs at least two observed training weather categories")
    vocabulary = {UNKNOWN: 0, **{category: i + 1 for i, category in enumerate(categories)}}
    manifest = {
        "schema_version": 1, "field": "weather_category", "vocabulary": vocabulary,
        "join_keys": ["uid", "tid", "departure_date"],
        "date_policy": "Departure date from upstream naive-as-epoch timestamps, interpreted as UTC to match weather DATE. No NYC-local conversion or original-timezone claim.",
        "missing_policy": "Retain every trajectory; code 0 means missing weather or category unseen in training.",
        "scope": "Weather on piece departure date; a piece crossing midnight keeps departure-date weather.",
        "alignment": "Exact road-ID and rate replay for all stored trajectories, then identical split and bucket indices.",
        "skipped_fmm_rows": skipped, "dataset_hashes": {}, "source_hashes": {},
        "sidecar_hashes": {}, "coverage": {},
    }
    sources = {"trip_map": trip_map, "fmm_matches": fmm_matches,
               "test_trips": test_trips, "weather": weather_path}
    manifest["source_hashes"] = {name: {"path": str(Path(path).resolve()), "sha256": sha256(path)}
                                 for name, path in sources.items()}
    for relative in ["extra_file/raw2new_rid.json", "extra_file/raw_rn_dict.json"] + [
        f"{split}/{name}" for split in SPLITS for name in ("eid_seqs.bin", "rate_seqs.bin")
    ]:
        manifest["dataset_hashes"][relative] = sha256(Path(dataset) / relative)
    output.mkdir(parents=True)
    for split, rows in joined.items():
        unseen = (rows.weather_status == "matched") & ~rows.weather_category.isin(vocabulary)
        rows.loc[unseen, "weather_status"] = "unseen_training_category"
        rows["weather_code"] = rows.weather_category.map(vocabulary).fillna(0).astype("int64")
        rows.to_parquet(output / f"{split}.parquet", index=False)
        manifest["sidecar_hashes"][f"{split}.parquet"] = sha256(output / f"{split}.parquet")
        manifest["coverage"][split] = {"trajectories": len(rows),
            "status_counts": {str(k): int(v) for k, v in rows.weather_status.value_counts().items()},
            "category_counts": {str(k): int(v) for k, v in rows.weather_category.value_counts().items()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_aligned_split(dataset, metadata_dir, split="gen_train"):
    """Load a verified v1 split as aligned IDs, raw rates, weather codes, and rows."""
    metadata_dir = Path(metadata_dir)
    manifest = json.loads((metadata_dir / "manifest.json").read_text())
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported metadata schema")
    for relative, expected in manifest["dataset_hashes"].items():
        if sha256(Path(dataset) / relative) != expected:
            raise ValueError(f"Sidecar belongs to different dataset: {relative}")
    filename = f"{split}.parquet"
    if sha256(metadata_dir / filename) != manifest["sidecar_hashes"][filename]:
        raise ValueError("Sidecar checksum differs")
    rows = pd.read_parquet(metadata_dir / filename)
    eids, rates = read_sequences(dataset, split)
    if len(rows) != sum(map(len, eids.values())):
        raise ValueError("Sidecar count differs from trajectory count")
    weather = {}
    for length, ids in eids.items():
        group = rows[rows.sequence_length == length].sort_values("index_within_length")
        if group.index_within_length.tolist() != list(range(len(ids))):
            raise ValueError(f"Missing/duplicate sidecar bucket indices: {length}")
        weather[length] = group.weather_code.astype(int).tolist()
    return eids, rates, weather, rows, manifest


def iter_batches(eids, rates, weather, batch_size, seed=0):
    """Shuffle complete aligned batches, including each length's remainder."""
    if batch_size < 1 or eids.keys() != rates.keys() or eids.keys() != weather.keys():
        raise ValueError("Positive batch size and identical length buckets required")
    batches = []
    for length, sequences in eids.items():
        if len(sequences) != len(rates[length]) or len(sequences) != len(weather[length]):
            raise ValueError(f"Metadata/trajectory count mismatch in bucket {length}")
        for start in range(0, len(sequences), batch_size):
            stop = start + batch_size
            batches.append((sequences[start:stop], rates[length][start:stop], weather[length][start:stop]))
    random.Random(seed).shuffle(batches)
    yield from batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "trip-map", "fmm-matches", "test-trips", "weather", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_sidecars(args.dataset, args.trip_map, args.fmm_matches,
                              args.test_trips, args.weather, args.output)
    print(json.dumps(manifest["coverage"], indent=2))


if __name__ == "__main__":
    main()
