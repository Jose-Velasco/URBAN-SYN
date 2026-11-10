"""Build v2 sparse sidecars for weather, mode, temperature, and opt-in POI context."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .conditioning import encode_requests
from .metadata import SPLITS, UNKNOWN, join_weather, read_sequences, replay_alignment, sha256

DEFAULT_FIELDS = ("weather_category", "transportation_mode", "temperature")
POI_FIELD = "previous_stop_poi_category"
POI_MAX_GAP_SECONDS = 900
POI_MAX_DISTANCE = 50
POI_DISTANCE_CRS = "EPSG:3857"
POI_DISTANCE_METRIC = "planar"
POI_DISTANCE_UNITS = "projected_metres"
POI_REASON_NO_STOP = "no_preceding_stop"
POI_REASON_GAP = "excessive_gap"
POI_REASON_AMBIGUOUS = "ambiguous_association"
POI_REASON_UNAVAILABLE = "unavailable_candidate"
POI_REASON_UNSEEN = "unseen_training_category"


def _timestamp_seconds(values, name):
    # The complete trajectory map stores Unix seconds; Parquet enrichment stores
    # datetime values. Preserve both conventions explicitly.
    if pd.api.types.is_numeric_dtype(values):
        parsed = pd.to_numeric(values, errors="raise")
        if parsed.isna().any():
            raise ValueError(f"{name} contains missing timestamps")
        return parsed.astype("int64")
    else:
        parsed = pd.to_datetime(values, errors="raise", utc=True)
    if parsed.isna().any():
        raise ValueError(f"{name} contains missing timestamps")
    # pandas may retain microsecond-resolution Parquet timestamps; timestamp()
    # avoids assuming nanosecond storage.
    return parsed.map(lambda value: int(value.timestamp())).astype("int64")


def _poi_identity(row):
    """Stable candidate identity, independent of input row order."""
    index_poi = row.get("index_poi")
    element_type = row.get("element_type")
    osmid = row.get("osmid")
    if pd.isna(index_poi) and pd.isna(osmid):
        return None
    return (str(index_poi) if not pd.isna(index_poi) else None,
            str(element_type) if not pd.isna(element_type) else None,
            str(osmid) if not pd.isna(osmid) else None)


def _numeric_osmid(value):
    if pd.isna(value):
        return float("inf")
    try:
        return int(value)
    except (TypeError, ValueError):
        return float("inf")


def _validate_parent_disjointness(aligned):
    train = aligned["gen_train"]
    test = aligned["gen_test"]
    train_parents = set(zip(train.uid.astype(str), train.tid.astype(str)))
    test_parents = set(zip(test.uid.astype(str), test.tid.astype(str)))
    if train_parents & test_parents:
        raise ValueError("Training and test parent trips overlap")
    train_users = set(train.uid.astype(str))
    test_users = set(test.uid.astype(str))
    if train_users & test_users:
        raise ValueError("Training and test users overlap")


def _prepare_poi_assignments(aligned, trip_map, poi, stops):
    """Assign one POI category/Unknown reason to every complete-map move.

    Move departure comes from the earliest complete-map timestamp. Stop and POI
    identity, coordinates, dates, distances, and context gaps remain provenance;
    only the resulting category and mask are encoder inputs.
    """
    required_trip = {"uid", "tid", "move_id", "start_timestamp"}
    required_stop = {"uid", "tid", "datetime", "leaving_datetime"}
    required_poi = required_stop | {"category", "distance"}
    if not required_trip.issubset(trip_map.columns):
        raise ValueError(f"Trajectory map lacks POI departure columns: {sorted(required_trip - set(trip_map))}")
    if not required_stop.issubset(stops.columns):
        raise ValueError(f"Stops lack required identity/timestamp columns: {sorted(required_stop - set(stops))}")
    if not required_poi.issubset(poi.columns):
        raise ValueError(f"POI enrichment lacks required columns: {sorted(required_poi - set(poi))}")

    trip_map = trip_map.copy()
    stops = stops.copy()
    poi = poi.copy()
    for frame in (trip_map, stops, poi):
        frame["uid"] = frame.uid.astype(str)
        frame["tid"] = frame.tid.astype(str)
    trip_map["move_departure"] = _timestamp_seconds(trip_map.start_timestamp, "trip map start_timestamp")
    move_departures = (trip_map.groupby(["uid", "tid", "move_id"], sort=False, dropna=False)
                       .move_departure.min().rename("move_departure").reset_index())

    stops["stop_departure"] = _timestamp_seconds(stops.leaving_datetime, "stops leaving_datetime")
    stops["stop_arrival"] = _timestamp_seconds(stops.datetime, "stops datetime")
    if (stops.stop_departure < stops.stop_arrival).any():
        raise ValueError("Stop departure precedes stop arrival")
    stops["_stop_key"] = list(zip(stops.uid, stops.tid, stops.stop_arrival, stops.stop_departure))
    stop_groups = {key: group for key, group in stops.groupby("_stop_key", sort=False, dropna=False)}

    poi["poi_stop_key"] = list(zip(poi.uid, poi.tid,
                                    _timestamp_seconds(poi.datetime, "POI datetime"),
                                    _timestamp_seconds(poi.leaving_datetime, "POI leaving_datetime")))
    poi_groups = {key: group for key, group in poi.groupby("poi_stop_key", sort=False, dropna=False)}
    assignments = []
    for move in move_departures.itertuples(index=False):
        parent = (move.uid, move.tid)
        candidates_stops = stops[(stops.uid == move.uid) & (stops.tid == move.tid) &
                                 (stops.stop_departure <= move.move_departure)]
        base = {"uid": move.uid, "tid": move.tid, "move_id": move.move_id,
                POI_FIELD: UNKNOWN, f"{POI_FIELD}_reason": POI_REASON_NO_STOP,
                "selected_stop_id": None, "selected_poi_id": None,
                "selected_poi_distance": np.nan, "previous_stop_gap_seconds": np.nan}
        if candidates_stops.empty:
            assignments.append(base)
            continue
        latest_departure = candidates_stops.stop_departure.max()
        latest = candidates_stops[candidates_stops.stop_departure == latest_departure]
        if len(latest) != 1:
            base[f"{POI_FIELD}_reason"] = POI_REASON_AMBIGUOUS
            assignments.append(base)
            continue
        stop = latest.iloc[0]
        gap = int(move.move_departure - stop.stop_departure)
        base["previous_stop_gap_seconds"] = gap
        if gap < 0 or gap > POI_MAX_GAP_SECONDS:
            base[f"{POI_FIELD}_reason"] = POI_REASON_GAP
            assignments.append(base)
            continue
        stop_key = stop._stop_key
        base["selected_stop_id"] = (str(stop.get("stop_id")) if "stop_id" in stops and not pd.isna(stop.get("stop_id"))
                                     else "|".join(map(str, stop_key)))
        enriched = poi_groups.get(stop_key, poi.iloc[0:0])
        if len(enriched) == 0:
            base[f"{POI_FIELD}_reason"] = POI_REASON_UNAVAILABLE
            assignments.append(base)
            continue
        # A source row must point back to exactly one stop with the same parent and timestamps.
        matching_stops = stop_groups.get(stop_key, stops.iloc[0:0])
        if len(matching_stops) != 1:
            base[f"{POI_FIELD}_reason"] = POI_REASON_AMBIGUOUS
            assignments.append(base)
            continue
        candidate_rows = []
        conflicting = False
        for identity, duplicate_rows in enriched.groupby(enriched.apply(_poi_identity, axis=1), dropna=False):
            if identity is None:
                conflicting = True
                break
            normalized = duplicate_rows.assign(
                _category=duplicate_rows.category.astype("string"),
                _distance=pd.to_numeric(duplicate_rows.distance, errors="coerce"))
            signatures = set(zip(normalized._category.astype(str), normalized._distance.astype(str)))
            if len(signatures) > 1:
                conflicting = True
                break
            candidate_rows.append(normalized.iloc[0])
        if conflicting:
            base[f"{POI_FIELD}_reason"] = POI_REASON_AMBIGUOUS
            assignments.append(base)
            continue
        usable = [row for row in candidate_rows
                  if pd.notna(row.category) and str(row.category) != UNKNOWN and
                  pd.notna(row._distance) and np.isfinite(row._distance) and
                  0 <= float(row._distance) <= POI_MAX_DISTANCE]
        if not usable:
            base[f"{POI_FIELD}_reason"] = POI_REASON_UNAVAILABLE
            assignments.append(base)
            continue
        usable.sort(key=lambda row: (float(row._distance),
                                     "" if pd.isna(row.element_type) else str(row.element_type),
                                     _numeric_osmid(row.osmid), str(row.category)))
        chosen = usable[0]
        base[POI_FIELD] = str(chosen.category)
        base[f"{POI_FIELD}_reason"] = "known"
        base["selected_poi_id"] = "|".join(map(str, _poi_identity(chosen)))
        base["selected_poi_distance"] = float(chosen._distance)
        assignments.append(base)
    return pd.DataFrame(assignments)


def prepare_rows(rows, weather, poi_assignments=None):
    """Join weather and optional move-level POI assignments without dropping rows."""
    weather = weather.copy()
    # Missing feature columns are valid: identity/date columns remain required for joining.
    if "DESCRIPTION" not in weather:
        weather["DESCRIPTION"] = pd.NA
    if "TAVG_C" not in weather:
        weather["TAVG_C"] = np.nan
    # WEATHER: categorical DESCRIPTION from weather_enrichment.parquet.
    joined = join_weather(rows, weather)
    # TRANSPORTATION MODE: move-level label preserved in the trajectory-ID map.
    joined["transportation_mode"] = joined.get("label", pd.Series(pd.NA, index=joined.index))
    joined["transportation_mode"] = joined.transportation_mode.astype("string").str.strip().replace("", pd.NA)
    temperature = weather[["uid", "tid", "DATE", "TAVG_C"]].copy()
    temperature["uid"] = temperature.uid.astype(str)
    temperature["tid"] = temperature.tid.astype(str)
    temperature["date"] = temperature.DATE.astype(str)
    joined = joined.merge(temperature[["uid", "tid", "date", "TAVG_C"]],
                          on=["uid", "tid", "date"], how="left", validate="many_to_one", sort=False)
    # Non-finite measurements are absent; malformed nonnumeric values are input errors.
    # TEMPERATURE: continuous TAVG_C from weather_enrichment.parquet, joined by
    # user, trip, and departure date.
    joined["temperature"] = pd.to_numeric(joined.TAVG_C, errors="raise").replace([np.inf, -np.inf], np.nan)
    joined = joined.drop(columns=["TAVG_C", "weather_code", "weather_status"], errors="ignore")
    if poi_assignments is not None:
        joined = joined.merge(poi_assignments, on=["uid", "tid", "move_id"],
                              how="left", validate="many_to_one", sort=False)
        joined[POI_FIELD] = joined[POI_FIELD].fillna(UNKNOWN)
        joined[f"{POI_FIELD}_reason"] = joined[f"{POI_FIELD}_reason"].fillna(POI_REASON_NO_STOP)
    return joined


def fit_fields(training_rows, names):
    """Fit weather/mode/POI vocabularies and temperature normalization on gen_train only.

    Categorical codebooks reserve code 0 for Unknown; temperature gets
    training-only Celsius mean/scale statistics.
    """
    result = []
    for name in names:
        if name not in DEFAULT_FIELDS and name != POI_FIELD:
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
        elif name == POI_FIELD:
            labels = sorted(set(training_rows[name].dropna().astype(str)) - {UNKNOWN})
            result.append({"name": name, "kind": "categorical",
                           "source": "POI enrichment:category for latest preceding stop",
                           "scope": "Latest same-parent stop departing no more than 900 seconds before move departure",
                           "vocabulary": {UNKNOWN: 0, **{label: i + 1 for i, label in enumerate(labels)}},
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
    """Materialize declared field values, codes/normalization, masks, and statuses.

    Stop/POI provenance remains in sidecar rows and is not passed to the encoder.
    """
    rows = rows.copy()
    for field in fields:
        name = field["name"]
        if field["kind"] == "categorical":
            raw_known = rows[name].notna() & (rows[name] != UNKNOWN)
            codes = rows[name].map(field["vocabulary"]).fillna(0).astype("int64")
            available = raw_known & (codes != 0)
            rows[f"{name}_code"] = codes
            status = np.where(raw_known & ~available, "unseen_training_category", "missing_source")
            if name == POI_FIELD and f"{name}_reason" in rows:
                rows[f"{name}_reason"] = np.where(
                    raw_known & ~available, POI_REASON_UNSEEN, rows[f"{name}_reason"])
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
                          names=DEFAULT_FIELDS, poi_path=None, stops_path=None):
    """Write verified v2 sidecars with default fields or opt-in POI enrichment.

    ``--poi`` and ``--stops`` are required only when names includes
    previous_stop_poi_category. The raw FMM trajectory inputs are unchanged;
    POI provenance columns are retained outside the encoder field contract.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new sparse sidecar directory: {output}")
    if not names or len(set(names)) != len(names):
        raise ValueError("Select at least one field without duplicates")
    if POI_FIELD in names and (poi_path is None or stops_path is None):
        raise ValueError("POI conditioning requires both --poi and --stops")
    if POI_FIELD not in names and (poi_path is not None or stops_path is not None):
        raise ValueError("--poi and --stops require --fields previous_stop_poi_category")
    aligned, skipped = replay_alignment(dataset, trip_map, fmm_matches, test_trips)
    _validate_parent_disjointness(aligned)
    weather = pd.read_parquet(weather_path)
    poi_assignments = None
    poi = stops = None
    if POI_FIELD in names:
        poi = pd.read_parquet(poi_path)
        stops = pd.read_parquet(stops_path)
        complete_trip_map = pd.read_csv(trip_map, dtype={"uid": str, "tid": str, "move_id": str})
        poi_assignments = _prepare_poi_assignments(aligned, complete_trip_map, poi, stops)
    prepared = {split: prepare_rows(rows, weather, poi_assignments) for split, rows in aligned.items()}
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
        "validation_split": None,
    }
    sources = {"trip_map": trip_map, "fmm_matches": fmm_matches,
               "test_trips": test_trips, "weather": weather_path}
    if POI_FIELD in names:
        sources.update({"poi": poi_path, "stops": stops_path})
    for name, path in sources.items():
        manifest["source_hashes"][name] = {"path": str(Path(path).resolve()), "sha256": sha256(path)}
    if POI_FIELD in names:
        manifest["selection_policy"] = {
            "field": POI_FIELD,
            "context": "latest same-(uid, tid) stop departure at or before complete-map move departure",
            "max_context_gap_seconds": POI_MAX_GAP_SECONDS,
            "distance_crs": POI_DISTANCE_CRS,
            "distance_metric": POI_DISTANCE_METRIC,
            "distance_units": POI_DISTANCE_UNITS,
            "max_distance": POI_MAX_DISTANCE,
            "ranking": ["distance", "element_type", "numeric_osmid", "category"],
            "candidate_identity": ["index_poi", "element_type", "osmid"],
            "no_backward_fallback": True,
            "provenance_only": ["selected_stop_id", "selected_poi_id",
                                 "selected_poi_distance", "previous_stop_gap_seconds"],
            "validation_split": None,
            "future_policy": "Use a separately versioned 50-ground-metre WGS84 geodesic retrieval policy for raw enrichment; do not treat this projected candidate set as complete.",
        }
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
        if POI_FIELD in names:
            field_reason = f"{POI_FIELD}_reason"
            known_moves = rows.loc[rows[field_reason] == "known",
                                                ["uid", "tid", "move_id"]].drop_duplicates()
            manifest["coverage"][split]["poi"] = {
                "moves": int(prepared[split][["uid", "tid", "move_id"]].drop_duplicates().shape[0]),
                "known_moves": int(len(known_moves)),
                "trajectories": int((rows[f"{POI_FIELD}_available"] if f"{POI_FIELD}_available" in rows else pd.Series(dtype=bool)).sum()),
                "reason_counts": {str(k): int(v) for k, v in rows[field_reason].value_counts().items()},
            }
        for field in fields:
            name = field["name"]
            manifest["coverage"][split]["fields"][name] = {
                str(status): int(count) for status, count in rows[f"{name}_status"].value_counts().items()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def load_sparse_split(dataset, metadata_dir, split="gen_train"):
    """Load a hashed v2 split as aligned IDs, raw rates, sparse values, masks, and rows.

    Only manifest-declared fields become tensor values/masks; raw rates remain
    the replayed converter values.
    """
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
            # Field boundary: sidecar raw values become encoder inputs here.
            # Categorical weather/mode/POI become torch.long codes; temperature
            # is the normalized float; encode_requests creates bool masks.
            requests.append({field["name"]: row[field["name"]] if row[f"{field['name']}_available"] else None
                             for field in manifest["fields"]})
        by_length[length] = encode_requests(requests, manifest["fields"])
    return eids, rates, by_length, rows, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("dataset", "trip-map", "fmm-matches", "test-trips", "weather", "output"):
        parser.add_argument(f"--{argument}", type=Path, required=True)
    parser.add_argument("--poi", type=Path, help="POI-enriched stop candidates")
    parser.add_argument("--stops", type=Path, help="Original stop table")
    parser.add_argument("--fields", nargs="+", choices=DEFAULT_FIELDS + (POI_FIELD,),
                        default=list(DEFAULT_FIELDS))
    args = parser.parse_args()
    manifest = build_sparse_sidecars(args.dataset, args.trip_map, args.fmm_matches,
                                     args.test_trips, args.weather, args.output, args.fields,
                                     args.poi, args.stops)
    print(json.dumps(manifest["coverage"], indent=2))


if __name__ == "__main__":
    main()
