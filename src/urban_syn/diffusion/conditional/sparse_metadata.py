"""Version 2 sidecars: field-wise missingness and training-only preprocessing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .conditioning import encode_requests
from .metadata import SPLITS, UNKNOWN, join_weather, read_sequences, replay_alignment, sha256

DEFAULT_FIELDS = ("weather_category", "transportation_mode", "temperature")


def prepare_rows(rows, weather):
    weather = weather.copy()
    # Missing feature columns are valid: identity/date columns remain required for joining.
    if "DESCRIPTION" not in weather:
        weather["DESCRIPTION"] = pd.NA
    if "TAVG_C" not in weather:
        weather["TAVG_C"] = np.nan
    joined = join_weather(rows, weather)
    joined["transportation_mode"] = joined.get("label", pd.Series(pd.NA, index=joined.index))
    joined["transportation_mode"] = joined.transportation_mode.astype("string").str.strip().replace("", pd.NA)
    temperature = weather[["uid", "tid", "DATE", "TAVG_C"]].copy()
    temperature["uid"] = temperature.uid.astype(str)
    temperature["tid"] = temperature.tid.astype(str)
    temperature["date"] = temperature.DATE.astype(str)
    joined = joined.merge(temperature[["uid", "tid", "date", "TAVG_C"]],
                          on=["uid", "tid", "date"], how="left", validate="many_to_one", sort=False)
    # Non-finite measurements are absent; malformed nonnumeric values are input errors.
    joined["temperature"] = pd.to_numeric(joined.TAVG_C, errors="raise").replace([np.inf, -np.inf], np.nan)
    return joined.drop(columns=["TAVG_C", "weather_code", "weather_status"], errors="ignore")


def fit_fields(training_rows, names):
    result = []
    for name in names:
        if name not in DEFAULT_FIELDS:
            raise ValueError(f"No source mapping defined for field {name}")
        if name == "temperature":
            observed = training_rows[name].dropna().astype(float)
            mean = float(observed.mean()) if len(observed) else 0.0
            scale = float(observed.std(ddof=0)) if len(observed) else 1.0
            if not np.isfinite(scale) or scale == 0:
                scale = 1.0
            result.append({"name": name, "kind": "continuous", "units": "degrees Celsius",
                           "source": "weather_enrichment.parquet:TAVG_C",
                           "scope": "Departure-date enrichment value; not instantaneous point temperature",
                           "normalization": {"mean": mean, "scale": scale,
                                             "observed_training_rows": len(observed), "ddof": 0},
                           "missing_contribution": "exact_zero"})
        else:
            labels = sorted(set(training_rows[name].dropna().astype(str)) - {UNKNOWN})
            result.append({"name": name, "kind": "categorical",
                           "source": "trajectory_map:label" if name == "transportation_mode"
                           else "weather_enrichment.parquet:DESCRIPTION",
                           "vocabulary": {UNKNOWN: 0, **{label: i + 1 for i, label in enumerate(labels)}},
                           "missing_contribution": "exact_zero"})
    return result


def transform_rows(rows, fields):
    rows = rows.copy()
    for field in fields:
        name = field["name"]
        if field["kind"] == "categorical":
            raw_known = rows[name].notna() & (rows[name] != UNKNOWN)
            codes = rows[name].map(field["vocabulary"]).fillna(0).astype("int64")
            available = raw_known & (codes != 0)
            rows[f"{name}_code"] = codes
            status = np.where(raw_known & ~available, "unseen_training_category", "missing_source")
        else:
            raw_known = rows[name].notna()
            stats = field["normalization"]
            available = raw_known & (stats["observed_training_rows"] > 0)
            rows[f"{name}_normalized"] = ((rows[name] - stats["mean"]) / stats["scale"]).where(available, 0.0)
            status = np.where(raw_known & ~available, "no_training_support", "missing_source")
        rows[f"{name}_available"] = available.astype(bool)
        rows[f"{name}_status"] = np.where(available, "known", status)
    return rows


def build_sparse_sidecars(dataset, trip_map, fmm_matches, test_trips, weather_path, output,
                          names=DEFAULT_FIELDS):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new sparse sidecar directory: {output}")
    if not names or len(set(names)) != len(names):
        raise ValueError("Select at least one field without duplicates")
    aligned, skipped = replay_alignment(dataset, trip_map, fmm_matches, test_trips)
    weather = pd.read_parquet(weather_path)
    prepared = {split: prepare_rows(rows, weather) for split, rows in aligned.items()}
    fields = fit_fields(prepared["gen_train"], names)
    manifest = {
        "schema_version": 2, "fields": fields,
        "join_keys": ["uid", "tid", "departure_date"],
        "date_policy": "Preserve upstream naive-as-epoch departure dates, interpreted as UTC to match weather DATE; original timezone remains unverified.",
        "alignment": "Exact replay of every stored road ID and rate, followed by identical split/bucket indices",
        "missing_policy": "Retain all trajectories; each absent field has a false mask and exactly zero encoder contribution. Unknown is a valid absence of conditioning.",
        "partial_policy": "Known fields remain active when other fields are missing; all missing fields select the learned zero-condition path.",
        "training_dropout": {"all_fields_probability": 0.1, "per_field_probability": 0.1,
                             "scope": "Independent of observed metadata; training only; never at inference",
                             "limitation": "Makes zero/partial conditions trainable; does not guarantee unbiased marginal or typical behavior under informative missingness"},
        "preprocessing_fit": "Known gen_train trajectory records only; never gen_test or gen_all",
        "skipped_fmm_rows": skipped, "coverage": {}, "dataset_hashes": {},
        "source_hashes": {}, "sidecar_hashes": {},
    }
    for name, path in {"trip_map": trip_map, "fmm_matches": fmm_matches,
                       "test_trips": test_trips, "weather": weather_path}.items():
        manifest["source_hashes"][name] = {"path": str(Path(path).resolve()), "sha256": sha256(path)}
    for relative in ["extra_file/raw2new_rid.json", "extra_file/raw_rn_dict.json"] + [
        f"{split}/{name}" for split in SPLITS for name in ("eid_seqs.bin", "rate_seqs.bin")
    ]:
        manifest["dataset_hashes"][relative] = sha256(Path(dataset) / relative)
    output.mkdir(parents=True)
    for split, rows in prepared.items():
        rows = transform_rows(rows, fields)
        rows.to_parquet(output / f"{split}.parquet", index=False)
        manifest["sidecar_hashes"][f"{split}.parquet"] = sha256(output / f"{split}.parquet")
        manifest["coverage"][split] = {"trajectories": len(rows), "fields": {}}
        for field in fields:
            name = field["name"]
            manifest["coverage"][split]["fields"][name] = {
                str(status): int(count) for status, count in rows[f"{name}_status"].value_counts().items()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def load_sparse_split(dataset, metadata_dir, split="gen_train"):
    metadata_dir = Path(metadata_dir)
    manifest = json.loads((metadata_dir / "manifest.json").read_text())
    if manifest["schema_version"] != 2:
        raise ValueError("Expected sparse metadata schema version 2")
    for relative, expected in manifest["dataset_hashes"].items():
        if sha256(Path(dataset) / relative) != expected:
            raise ValueError(f"Sparse sidecar/dataset checksum mismatch: {relative}")
    filename = f"{split}.parquet"
    if sha256(metadata_dir / filename) != manifest["sidecar_hashes"][filename]:
        raise ValueError("Sparse sidecar checksum mismatch")
    rows = pd.read_parquet(metadata_dir / filename)
    eids, rates = read_sequences(dataset, split)
    if len(rows) != sum(map(len, eids.values())):
        raise ValueError("Sparse sidecar count mismatch")
    by_length = {}
    for length, trajectories in eids.items():
        group = rows[rows.sequence_length == length].sort_values("index_within_length")
        if group.index_within_length.tolist() != list(range(len(trajectories))):
            raise ValueError("Sparse sidecar bucket indices mismatch")
        requests = []
        for _, row in group.iterrows():
            requests.append({field["name"]: row[field["name"]] if row[f"{field['name']}_available"] else None
                             for field in manifest["fields"]})
        by_length[length] = encode_requests(requests, manifest["fields"])
    return eids, rates, by_length, rows, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("dataset", "trip-map", "fmm-matches", "test-trips", "weather", "output"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    parser.add_argument("--fields", nargs="+", choices=DEFAULT_FIELDS, default=list(DEFAULT_FIELDS))
    args = parser.parse_args()
    manifest = build_sparse_sidecars(args.dataset, args.trip_map, args.fmm_matches,
                                     args.test_trips, args.weather, args.output, args.fields)
    print(json.dumps(manifest["coverage"], indent=2))


if __name__ == "__main__":
    main()
