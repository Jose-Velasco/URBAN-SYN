#!/usr/bin/env python3
"""Build project data into the CSV inputs expected by LSTM-TrajGAN.

This script is intentionally an upstream adapter only. It does not replace
``data/csv2npy.py`` and does not run model training or prediction.

Expected source columns (MAT-Dataset enriched stop data):
    uid, tid, lat, lng|lon, datetime, category

Outputs:
    <dataset>_train_latlon.csv
    <dataset>_test_latlon.csv
    <dataset>_dev_train_encoded_final.csv
    <dataset>_dev_test_encoded_final.csv
    <dataset>_lstm_trajgan_metadata.json

Auxiliary mapping files are also written so generated integer IDs/categories
can be mapped back to the source values.
"""

from __future__ import print_function

import argparse
import json
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


CATEGORY_VOCAB_SIZE = 10
REQUIRED_COLUMNS = {"uid", "tid", "lat", "datetime", "category"}
MODEL_COLUMNS = ["label", "tid", "lat", "lon", "day", "hour", "category"]
UNKNOWN_CATEGORY_VALUES = {"", "unknown", "none", "nan", "null"}


@contextmanager
def timed_stage(logger, timings, name):
    """Log and record the elapsed wall-clock time for one build stage."""
    logger.info("Starting: %s", name)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        timings[name] = round(elapsed, 3)
        logger.info("Finished: %s in %.2f s", name, elapsed)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build LSTM-TrajGAN-compatible train/test CSV inputs from "
            "project semantic trajectory data."
        )
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        required=True,
        help="Source .csv or .parquet file containing semantic trajectory rows.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Directory where LSTM-TrajGAN CSV inputs and metadata are written.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="nyc",
        help="Prefix used for generated artifacts (default: nyc).",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Fraction of unique trajectories assigned to training (default: 0.8).",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=101,
        help="Random seed used for the trajectory-level split (default: 101).",
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=None,
        help="Log directory. Defaults to <out_dir>/logs.",
    )
    return parser.parse_args()


def setup_logging(log_dir, dataset_name):
    """Configure logging to both stdout and a timestamped log file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / (
        "build_lstm_trajgan_inputs_{}_{}.log".format(dataset_name, timestamp)
    )

    logger = logging.getLogger("build_lstm_trajgan_inputs")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers = []

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(str(log_path))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger, log_path


def load_source(path):
    """Load a CSV or Parquet source file."""
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "Reading Parquet requires pyarrow. Install a version "
                "compatible with the container's Python version."
            ) from exc

        # The LSTM-TrajGAN container uses an old pandas/pyarrow stack.
        # Reading modern Parquet pandas metadata directly can fail, so
        # load the Arrow table while ignoring that metadata and let the
        # builder normalize the required columns afterward.
        table = pq.read_table(
            str(path),
            use_pandas_metadata=False,
        )

        return table.to_pandas(
            ignore_metadata=True,
        )

    raise ValueError(
        "Unsupported input format '{}'. Use .csv or .parquet.".format(suffix)
    )


def validate_source_columns(df):
    """Validate and normalize source column names required by the adapter."""
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(
            "Input is missing required columns: {}".format(", ".join(missing))
        )

    if "lon" not in df.columns and "lng" not in df.columns:
        raise ValueError("Input must contain either a 'lon' or 'lng' column.")

    if "lon" in df.columns and "lng" in df.columns:
        raise ValueError(
            "Input contains both 'lon' and 'lng'. Keep one longitude column "
            "to avoid ambiguity."
        )

    if "lng" in df.columns:
        df = df.rename(columns={"lng": "lon"})

    return df


def clean_source_rows(df, logger):
    """Coerce core fields, remove unusable rows, and derive day/hour."""
    df = df.copy()

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    before = len(df)
    required_non_null = ["uid", "tid", "lat", "lon", "datetime"]
    df = df.dropna(subset=required_non_null).copy()
    dropped_core = before - len(df)

    if dropped_core:
        logger.warning(
            "Dropped %d rows with missing/invalid uid, tid, coordinates, or datetime.",
            dropped_core,
        )

    # The paper uses seven day classes and 24 hour classes. pandas uses
    # Monday=0 ... Sunday=6, which matches the expected one-hot index range.
    df["day"] = df["datetime"].dt.dayofweek.astype(np.int32)
    df["hour"] = df["datetime"].dt.hour.astype(np.int32)

    category_text = df["category"].astype(str).str.strip()
    unknown_mask = df["category"].isna() | category_text.str.lower().isin(
        UNKNOWN_CATEGORY_VALUES
    )
    unknown_count = int(unknown_mask.sum())

    if unknown_count:
        logger.info(
            "Excluding %d rows without a usable POI category before top-%d selection.",
            unknown_count,
            CATEGORY_VOCAB_SIZE,
        )

    df = df.loc[~unknown_mask].copy()
    df["category"] = df["category"].astype(str).str.strip()

    if df.empty:
        raise ValueError("No usable rows remain after cleaning source data.")

    return df, dropped_core, unknown_count


def select_top_categories(df, logger):
    """Keep the ten most frequent categories expected by the current model."""
    counts = (
        df.groupby("category")
        .size()
        .reset_index(name="count")
        .sort_values(["count", "category"], ascending=[False, True])
        .reset_index(drop=True)
    )

    selected = counts.head(CATEGORY_VOCAB_SIZE).copy()
    selected_categories = selected["category"].tolist()

    if len(selected_categories) < CATEGORY_VOCAB_SIZE:
        logger.warning(
            "Only %d usable categories were found; the current model still "
            "allocates a category vocabulary of %d.",
            len(selected_categories),
            CATEGORY_VOCAB_SIZE,
        )

    before = len(df)
    df = df[df["category"].isin(selected_categories)].copy()
    removed = before - len(df)

    category_to_id = {
        category: idx for idx, category in enumerate(selected_categories)
    }
    df["category"] = df["category"].map(category_to_id).astype(np.int32)

    logger.info(
        "Selected top %d categories; retained %d/%d rows (removed %d).",
        len(selected_categories),
        len(df),
        before,
        removed,
    )
    for row in selected.itertuples(index=False):
        logger.info("  category=%s | rows=%d", row.category, row.count)

    selected["category_id"] = selected["category"].map(category_to_id)
    selected = selected[["category", "category_id", "count"]]

    return df, selected, removed


def build_stable_id_mapping(series, source_name, encoded_name):
    """Create a stable contiguous integer mapping for an identifier column."""
    unique_values = series.drop_duplicates().tolist()
    unique_values = sorted(unique_values, key=lambda value: str(value))

    mapping = pd.DataFrame(
        {
            source_name: unique_values,
            encoded_name: np.arange(len(unique_values), dtype=np.int64),
        }
    )
    value_to_id = dict(zip(mapping[source_name], mapping[encoded_name]))
    encoded = series.map(value_to_id)

    if encoded.isna().any():
        raise ValueError("Failed to encode all values from '{}'".format(source_name))

    return encoded.astype(np.int64), mapping


def encode_identifiers(df):
    """Encode user and trajectory identifiers into integer IDs."""
    df = df.copy()

    df["label"], label_mapping = build_stable_id_mapping(
        df["uid"], "source_uid", "label"
    )
    df["tid"], tid_mapping = build_stable_id_mapping(
        df["tid"], "source_tid", "tid"
    )

    labels_per_tid = df.groupby("tid")["label"].nunique()
    inconsistent = labels_per_tid[labels_per_tid > 1]
    if not inconsistent.empty:
        raise ValueError(
            "Found {} trajectories associated with more than one user label.".format(
                len(inconsistent)
            )
        )

    return df, label_mapping, tid_mapping


def split_by_trajectory(df, train_ratio, random_state):
    """Split rows by unique trajectory ID, never by individual points."""
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be strictly between 0 and 1.")

    unique_tids = pd.Series(df["tid"].drop_duplicates().values)
    if len(unique_tids) < 2:
        raise ValueError("At least two trajectories are required for train/test split.")

    train_tids = unique_tids.sample(
        frac=train_ratio,
        random_state=random_state,
    )

    # Guard very small datasets against an empty side after fractional sampling.
    if train_tids.empty:
        train_tids = unique_tids.sample(n=1, random_state=random_state)
    if len(train_tids) == len(unique_tids):
        train_tids = train_tids.iloc[:-1]

    train_tid_set = set(train_tids.tolist())
    train_df = df[df["tid"].isin(train_tid_set)].copy()
    test_df = df[~df["tid"].isin(train_tid_set)].copy()

    overlap = set(train_df["tid"].unique()) & set(test_df["tid"].unique())
    if overlap:
        raise ValueError(
            "Trajectory leakage detected: {} tids appear in both splits.".format(
                len(overlap)
            )
        )

    return train_df, test_df


def calculate_normalization(df):
    """Calculate the centroid and scale factor used by LSTM-TrajGAN."""
    lat_centroid = float(df["lat"].mean())
    lon_centroid = float(df["lon"].mean())

    lat_deviation = (df["lat"] - lat_centroid).abs().max()
    lon_deviation = (df["lon"] - lon_centroid).abs().max()
    scale_factor = float(max(lat_deviation, lon_deviation))

    if not np.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError("Computed scale_factor must be a positive finite value.")

    return lat_centroid, lon_centroid, scale_factor


def build_model_frames(train_df, test_df, lat_centroid, lon_centroid):
    """Create raw-coordinate and centroid-deviation CSV views."""
    sort_columns = ["tid", "datetime"]
    train_df = train_df.sort_values(sort_columns).reset_index(drop=True)
    test_df = test_df.sort_values(sort_columns).reset_index(drop=True)

    train_raw = train_df[MODEL_COLUMNS].copy()
    test_raw = test_df[MODEL_COLUMNS].copy()

    train_encoded = train_raw.copy()
    test_encoded = test_raw.copy()

    # LSTM-TrajGAN embeds latitude/longitude deviations from the global
    # trajectory-location centroid rather than absolute coordinates.
    train_encoded["lat"] = train_encoded["lat"] - lat_centroid
    train_encoded["lon"] = train_encoded["lon"] - lon_centroid
    test_encoded["lat"] = test_encoded["lat"] - lat_centroid
    test_encoded["lon"] = test_encoded["lon"] - lon_centroid

    return train_raw, test_raw, train_encoded, test_encoded


def trajectory_length_stats(df):
    """Return trajectory length summary statistics."""
    lengths = df.groupby("tid").size()
    if lengths.empty:
        raise ValueError("Cannot calculate trajectory lengths for an empty dataframe.")

    return {
        "trajectories": int(len(lengths)),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "median": float(lengths.median()),
    }


def write_csv(df, path):
    """Write a dataframe to CSV, creating its parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_json(data, path):
    """Write JSON metadata with readable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(data, file, indent=2, sort_keys=True)


def main():
    """Run the LSTM-TrajGAN input build."""
    args = parse_args()
    total_start = time.perf_counter()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.log_dir if args.log_dir is not None else args.out_dir / "logs"
    logger, log_path = setup_logging(log_dir, args.dataset_name)
    timings = {}

    logger.info("=" * 72)
    logger.info("LSTM-TrajGAN input build")
    logger.info("=" * 72)
    logger.info("dataset_name=%s", args.dataset_name)
    logger.info("input_path=%s", args.input_path)
    logger.info("out_dir=%s", args.out_dir)
    logger.info("train_ratio=%.4f", args.train_ratio)
    logger.info("random_state=%d", args.random_state)
    logger.info("category_vocab_size=%d", CATEGORY_VOCAB_SIZE)
    logger.info("log_file=%s", log_path)

    if not args.input_path.exists():
        raise FileNotFoundError("Input file does not exist: {}".format(args.input_path))

    with timed_stage(logger, timings, "load_source"):
        source_df = load_source(args.input_path)
        source_rows = int(len(source_df))
        logger.info(
            "Loaded %d rows and %d columns.",
            source_rows,
            len(source_df.columns),
        )

    with timed_stage(logger, timings, "clean_and_prepare"):
        df = validate_source_columns(source_df)
        source_trajectories = int(df["tid"].nunique(dropna=True))
        source_users = int(df["uid"].nunique(dropna=True))

        df, dropped_core, unknown_category_rows = clean_source_rows(df, logger)
        df, category_mapping, rows_removed_by_category = select_top_categories(
            df, logger
        )
        df, label_mapping, tid_mapping = encode_identifiers(df)

        if df.empty:
            raise ValueError("No rows remain after LSTM-TrajGAN input preparation.")

    with timed_stage(logger, timings, "trajectory_split"):
        train_df, test_df = split_by_trajectory(
            df,
            train_ratio=args.train_ratio,
            random_state=args.random_state,
        )
        logger.info(
            "Trajectory split: train=%d trajectories / %d rows | "
            "test=%d trajectories / %d rows",
            train_df["tid"].nunique(),
            len(train_df),
            test_df["tid"].nunique(),
            len(test_df),
        )

    with timed_stage(logger, timings, "normalization_and_frames"):
        lat_centroid, lon_centroid, scale_factor = calculate_normalization(df)
        train_raw, test_raw, train_encoded, test_encoded = build_model_frames(
            train_df,
            test_df,
            lat_centroid,
            lon_centroid,
        )

        overall_stats = trajectory_length_stats(df)
        train_stats = trajectory_length_stats(train_df)
        test_stats = trajectory_length_stats(test_df)
        max_length = int(overall_stats["max"])

        logger.info(
            "Normalization: lat_centroid=%.8f | lon_centroid=%.8f | "
            "scale_factor=%.8f",
            lat_centroid,
            lon_centroid,
            scale_factor,
        )
        logger.info(
            "Trajectory lengths: overall max=%d | train max=%d | test max=%d",
            max_length,
            train_stats["max"],
            test_stats["max"],
        )

    prefix = args.dataset_name
    paths = {
        "train_latlon": args.out_dir / "{}_train_latlon.csv".format(prefix),
        "test_latlon": args.out_dir / "{}_test_latlon.csv".format(prefix),
        "train_encoded": args.out_dir
        / "{}_dev_train_encoded_final.csv".format(prefix),
        "test_encoded": args.out_dir
        / "{}_dev_test_encoded_final.csv".format(prefix),
        "metadata": args.out_dir
        / "{}_lstm_trajgan_metadata.json".format(prefix),
        "category_mapping": args.out_dir
        / "{}_lstm_trajgan_category_mapping.csv".format(prefix),
        "label_mapping": args.out_dir
        / "{}_lstm_trajgan_label_mapping.csv".format(prefix),
        "tid_mapping": args.out_dir
        / "{}_lstm_trajgan_tid_mapping.csv".format(prefix),
    }

    with timed_stage(logger, timings, "write_artifacts"):
        write_csv(train_raw, paths["train_latlon"])
        write_csv(test_raw, paths["test_latlon"])
        write_csv(train_encoded, paths["train_encoded"])
        write_csv(test_encoded, paths["test_encoded"])
        write_csv(category_mapping, paths["category_mapping"])
        write_csv(label_mapping, paths["label_mapping"])
        write_csv(tid_mapping, paths["tid_mapping"])

    total_elapsed = time.perf_counter() - total_start
    timings["total"] = round(total_elapsed, 3)

    category_records = []
    for row in category_mapping.itertuples(index=False):
        category_records.append(
            {
                "source_category": str(row.category),
                "category": int(row.category_id),
                "rows": int(row.count),
            }
        )

    metadata = {
        "dataset_name": args.dataset_name,
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "source_path": str(args.input_path),
        "parameters": {
            "train_ratio": float(args.train_ratio),
            "random_state": int(args.random_state),
            "category_vocab_size": CATEGORY_VOCAB_SIZE,
            "split_unit": "trajectory_tid",
        },
        "source_summary": {
            "rows": source_rows,
            "trajectories": source_trajectories,
            "users": source_users,
            "dropped_invalid_core_rows": int(dropped_core),
            "excluded_unknown_category_rows": int(unknown_category_rows),
            "removed_outside_top_categories": int(rows_removed_by_category),
        },
        "prepared_summary": {
            "rows": int(len(df)),
            "trajectories": int(df["tid"].nunique()),
            "users": int(df["label"].nunique()),
            "categories": int(df["category"].nunique()),
        },
        "split_summary": {
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_trajectories": int(train_df["tid"].nunique()),
            "test_trajectories": int(test_df["tid"].nunique()),
            "trajectory_overlap": 0,
        },
        "trajectory_lengths": {
            "max_length": max_length,
            "overall": overall_stats,
            "train": train_stats,
            "test": test_stats,
        },
        "normalization": {
            "lat_centroid": lat_centroid,
            "lon_centroid": lon_centroid,
            "scale_factor": scale_factor,
            "encoded_coordinate_definition": "raw_coordinate_minus_global_centroid",
        },
        "categories": category_records,
        "outputs": {name: str(path) for name, path in paths.items()},
        "log_file": str(log_path),
        "timing_seconds": timings,
    }

    # Metadata is written last so it describes only a successfully completed build.
    write_json(metadata, paths["metadata"])

    logger.info("=" * 72)
    logger.info("BUILD SUMMARY")
    logger.info("=" * 72)
    logger.info("Prepared rows: %d", len(df))
    logger.info(
        "Prepared trajectories: %d (train=%d, test=%d)",
        df["tid"].nunique(),
        train_df["tid"].nunique(),
        test_df["tid"].nunique(),
    )
    logger.info("Prepared users: %d", df["label"].nunique())
    logger.info("Categories: %d", df["category"].nunique())
    logger.info("Cached max_length: %d", max_length)
    logger.info("Train rows: %d", len(train_df))
    logger.info("Test rows: %d", len(test_df))
    logger.info("Trajectory overlap: 0")
    logger.info("Total elapsed: %.2f s", total_elapsed)
    logger.info("Artifacts:")
    for name, path in paths.items():
        logger.info("  %-18s %s", name + ":", path)
    logger.info("  %-18s %s", "log:", log_path)
    logger.info("Build completed successfully.")


if __name__ == "__main__":
    main()
