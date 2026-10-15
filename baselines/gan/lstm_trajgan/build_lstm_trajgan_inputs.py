#!/usr/bin/env python3
"""Build project trajectory data into CSV inputs expected by LSTM-TrajGAN.

This script is intentionally an upstream adapter only. It does not replace the
upstream ``data/csv2npy.py`` encoder and does not run model training or
prediction. A future pipeline wrapper should call this script as the dataset
preparation stage, then call the upstream encoder/training/prediction scripts.

Expected source columns for the default GPS-only mode (e.g. traj_cleaned):
    uid, tid, lat, lng|lon, datetime

Optional semantic mode additionally requires:
    category

Category strategies:
    constant  Assign one fixed category ID to every GPS point. This preserves
              the upstream 10-way category input shape while effectively
              ablating POI/category information for preliminary GPS+time runs.
    top10     Preserve the previous semantic-stop behavior: keep the ten most
              frequent usable source categories and encode them as 0..9.

Outputs:
    <dataset>_train_latlon.csv
    <dataset>_test_latlon.csv
    <dataset>_dev_train_encoded_final.csv
    <dataset>_dev_test_encoded_final.csv
    <dataset>_lstm_trajgan_metadata.json

Auxiliary mapping files are also written so generated integer IDs/categories
can be mapped back to source values.
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
from typing import Optional, Tuple

import numpy as np
import pandas as pd


CATEGORY_VOCAB_SIZE = 10
CORE_REQUIRED_COLUMNS = {"uid", "tid", "lat", "datetime"}
MODEL_COLUMNS = ["label", "tid", "lat", "lon", "day", "hour", "category"]
UNKNOWN_CATEGORY_VALUES = {"", "unknown", "none", "nan", "null"}
CATEGORY_MODES = ("constant", "top10")


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
            "project trajectory data."
        )
    )
    parser.add_argument(
        "--input_path",
        type=Path,
        default=None,
        help=(
            "Combined source .csv or .parquet file. The adapter will split "
            "trajectories using --train_ratio. Cannot be combined with "
            "--train_input_path/--test_input_path."
        ),
    )
    parser.add_argument(
        "--train_input_path",
        type=Path,
        default=None,
        help=(
            "Optional pre-split training .csv or .parquet file. Must be used "
            "together with --test_input_path and without --input_path."
        ),
    )
    parser.add_argument(
        "--test_input_path",
        type=Path,
        default=None,
        help=(
            "Optional pre-split test .csv or .parquet file. Must be used "
            "together with --train_input_path and without --input_path."
        ),
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
        help="Random seed used for trajectory sampling/splitting (default: 101).",
    )
    parser.add_argument(
        "--category_mode",
        choices=CATEGORY_MODES,
        default="constant",
        help=(
            "Category handling. 'constant' assigns the same category ID to "
            "every point and is the default preliminary GPS-only baseline. "
            "'top10' requires a source category column and keeps the ten most "
            "frequent usable categories."
        ),
    )
    parser.add_argument(
        "--constant_category",
        type=int,
        default=0,
        help=(
            "Category ID assigned to every point when --category_mode=constant. "
            "Must be in [0, 9] because the upstream model uses a 10-way "
            "category vector (default: 0)."
        ),
    )
    parser.add_argument(
        "--max_trajectories",
        type=int,
        default=None,
        help=(
            "Optional trajectory-level sample limit for smoke/debug runs. "
            "Sampling is deterministic using --random_state. Omit for a full run."
        ),
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=None,
        help="Log directory. Defaults to <out_dir>/logs.",
    )
    return parser.parse_args()


def validate_options(
    input_path,
    train_input_path,
    test_input_path,
    train_ratio,
    constant_category,
    max_trajectories,
):
    # type: (Optional[Path], Optional[Path], Optional[Path], float, int, Optional[int]) -> None
    """Validate CLI values before doing expensive work."""
    has_combined_input = input_path is not None
    has_train_input = train_input_path is not None
    has_test_input = test_input_path is not None

    if has_combined_input and (has_train_input or has_test_input):
        raise ValueError(
            "Use either --input_path or the explicit "
            "--train_input_path/--test_input_path pair, not both."
        )

    if not has_combined_input and not (has_train_input and has_test_input):
        raise ValueError(
            "Provide either --input_path or both --train_input_path "
            "and --test_input_path."
        )

    if has_combined_input and not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be strictly between 0 and 1.")

    if not 0 <= constant_category < CATEGORY_VOCAB_SIZE:
        raise ValueError(
            "constant_category must be in [0, {}], got {}.".format(
                CATEGORY_VOCAB_SIZE - 1,
                constant_category,
            )
        )

    if max_trajectories is not None and max_trajectories < 2:
        raise ValueError("max_trajectories must be at least 2 when provided.")


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

        # The LSTM-TrajGAN container uses a legacy pandas/pyarrow stack.
        # Modern pandas metadata can fail there, so intentionally ignore it.
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


def validate_source_columns(df, category_mode):
    """Validate and normalize source columns required by the adapter."""
    missing = sorted(CORE_REQUIRED_COLUMNS - set(df.columns))
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

    if category_mode == "top10" and "category" not in df.columns:
        raise ValueError(
            "--category_mode=top10 requires a source 'category' column. "
            "Use --category_mode=constant for traj_cleaned.parquet."
        )

    if "lng" in df.columns:
        df = df.rename(columns={"lng": "lon"})

    return df


def clean_core_rows(df, logger):
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
    # Monday=0 ... Sunday=6, matching the expected one-hot index range.
    df["day"] = df["datetime"].dt.dayofweek.astype(np.int32)
    df["hour"] = df["datetime"].dt.hour.astype(np.int32)

    if df.empty:
        raise ValueError("No usable rows remain after cleaning source data.")

    return df, dropped_core


def sample_trajectories(df, max_trajectories, random_state, logger):
    """Optionally keep a deterministic trajectory-level subset for smoke runs."""
    unique_tids = pd.Series(df["tid"].drop_duplicates().values)
    available = int(len(unique_tids))

    if max_trajectories is None or max_trajectories >= available:
        return df, available, available

    sampled_tids = unique_tids.sample(
        n=max_trajectories,
        random_state=random_state,
        replace=False,
    )
    sampled_set = set(sampled_tids.tolist())
    sampled_df = df[df["tid"].isin(sampled_set)].copy()

    logger.info(
        "Smoke/debug trajectory sampling: retained %d/%d trajectories (%d rows).",
        max_trajectories,
        available,
        len(sampled_df),
    )

    return sampled_df, available, int(max_trajectories)


def assign_constant_category(df, category_id, logger):
    """Assign one category token to all points for a GPS+time baseline."""
    df = df.copy()
    df["category"] = np.int32(category_id)

    mapping = pd.DataFrame(
        {
            "category": ["__constant__"],
            "category_id": [np.int32(category_id)],
            "count": [int(len(df))],
        }
    )

    logger.info(
        "Category mode=constant: assigned category=%d to all %d points.",
        category_id,
        len(df),
    )
    logger.info(
        "The upstream model still uses its original %d-dimensional category input.",
        CATEGORY_VOCAB_SIZE,
    )

    return df, mapping, 0, 0


def select_top_categories(df, logger):
    """Keep and encode the ten most frequent usable source categories."""
    df = df.copy()
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
        raise ValueError("No usable category rows remain after cleaning source data.")

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
            "Only %d usable categories were found; the upstream model still "
            "allocates a category vocabulary of %d.",
            len(selected_categories),
            CATEGORY_VOCAB_SIZE,
        )

    before_top10 = len(df)
    df = df[df["category"].isin(selected_categories)].copy()
    removed = before_top10 - len(df)

    category_to_id = {
        category: idx for idx, category in enumerate(selected_categories)
    }
    df["category"] = df["category"].map(category_to_id).astype(np.int32)

    logger.info(
        "Category mode=top10: retained %d/%d categorized rows (removed %d).",
        len(df),
        before_top10,
        removed,
    )
    for row in selected.itertuples(index=False):
        logger.info("  category=%s | rows=%d", row.category, row.count)

    selected["category_id"] = selected["category"].map(category_to_id)
    selected = selected[["category", "category_id", "count"]]

    return df, selected, unknown_count, removed


def apply_category_strategy(df, category_mode, constant_category, logger):
    """Apply the selected category representation strategy."""
    if category_mode == "constant":
        return assign_constant_category(df, constant_category, logger)

    if category_mode == "top10":
        return select_top_categories(df, logger)

    raise ValueError("Unsupported category_mode: {}".format(category_mode))


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
    """Return trajectory-length statistics useful before global padding."""
    lengths = df.groupby("tid").size()
    if lengths.empty:
        raise ValueError("Cannot calculate trajectory lengths for an empty dataframe.")

    values = lengths.values.astype(np.float64)

    return {
        "trajectories": int(len(lengths)),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "median": float(lengths.median()),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
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


def resolve_input_mode(train_input_path):
    # type: (Optional[Path]) -> Tuple[bool, str]
    """Return whether pre-split inputs are used and the metadata mode label."""
    explicit_split = train_input_path is not None
    input_mode = "pre_split" if explicit_split else "single_source_split"
    return explicit_split, input_mode


def log_build_configuration(
    logger,
    dataset_name,
    input_mode,
    input_path,
    train_input_path,
    test_input_path,
    out_dir,
    train_ratio,
    explicit_split,
    random_state,
    category_mode,
    constant_category,
    max_trajectories,
    log_path,
):
    """Log the resolved build configuration once at startup."""
    logger.info("=" * 72)
    logger.info("LSTM-TrajGAN input build")
    logger.info("=" * 72)
    logger.info("dataset_name=%s", dataset_name)
    logger.info("input_mode=%s", input_mode)
    logger.info("input_path=%s", input_path)
    logger.info("train_input_path=%s", train_input_path)
    logger.info("test_input_path=%s", test_input_path)
    logger.info("out_dir=%s", out_dir)
    logger.info("train_ratio=%s", train_ratio if not explicit_split else "n/a")
    logger.info("random_state=%d", random_state)
    logger.info("category_mode=%s", category_mode)
    logger.info("constant_category=%d", constant_category)
    logger.info("category_vocab_size=%d", CATEGORY_VOCAB_SIZE)
    logger.info("max_trajectories=%s", max_trajectories)
    logger.info("log_file=%s", log_path)


def require_existing_path(path, argument_name):
    # type: (Optional[Path], str) -> Path
    """Return an existing path or raise a CLI-oriented error."""
    if path is None:
        raise ValueError("{} was not provided.".format(argument_name))

    if not path.exists():
        raise FileNotFoundError(
            "{} does not exist: {}".format(argument_name, path)
        )

    return path


def load_validated_source(path, category_mode):
    """Load one source dataset and normalize its required columns."""
    df = load_source(path)
    return validate_source_columns(df, category_mode)


def validate_no_trajectory_overlap(train_df, test_df, context):
    """Raise when a trajectory ID occurs in both sides of a split."""
    train_tids = set(train_df["tid"].dropna().tolist())
    test_tids = set(test_df["tid"].dropna().tolist())
    overlap = train_tids & test_tids

    if overlap:
        raise ValueError(
            "Trajectory leakage detected {}: {} tids appear in both "
            "train and test datasets.".format(context, len(overlap))
        )


def load_pre_split_sources(
    train_input_path,
    test_input_path,
    category_mode,
):
    """Load explicit train/test inputs while preserving their split membership."""
    train_path = require_existing_path(
        train_input_path,
        "--train_input_path",
    )
    test_path = require_existing_path(
        test_input_path,
        "--test_input_path",
    )

    train_df = load_validated_source(train_path, category_mode)
    test_df = load_validated_source(test_path, category_mode)

    validate_no_trajectory_overlap(
        train_df,
        test_df,
        "in explicit source files",
    )

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["__input_split"] = "train"
    test_df["__input_split"] = "test"

    return pd.concat(
        [train_df, test_df],
        ignore_index=True,
    )


def load_combined_source(input_path, category_mode):
    """Load the single-source mode used with a trajectory-level random split."""
    source_path = require_existing_path(input_path, "--input_path")
    return load_validated_source(source_path, category_mode)


def load_input_data(
    input_path,
    train_input_path,
    test_input_path,
    explicit_split,
    category_mode,
):
    """Load either one combined source or an explicit train/test pair."""
    if explicit_split:
        return load_pre_split_sources(
            train_input_path,
            test_input_path,
            category_mode,
        )

    return load_combined_source(
        input_path,
        category_mode,
    )


def prepare_source_data(
    source_df,
    max_trajectories,
    random_state,
    category_mode,
    constant_category,
    logger,
):
    """Clean, sample, categorize, and encode a loaded source dataframe."""
    source_trajectories = int(source_df["tid"].nunique(dropna=True))
    source_users = int(source_df["uid"].nunique(dropna=True))

    df, dropped_core = clean_core_rows(source_df, logger)
    cleaned_rows = int(len(df))
    cleaned_trajectories = int(df["tid"].nunique())

    df, sample_available, sample_selected = sample_trajectories(
        df,
        max_trajectories=max_trajectories,
        random_state=random_state,
        logger=logger,
    )

    (
        df,
        category_mapping,
        unknown_category_rows,
        rows_removed_by_category,
    ) = apply_category_strategy(
        df,
        category_mode=category_mode,
        constant_category=constant_category,
        logger=logger,
    )

    df, label_mapping, tid_mapping = encode_identifiers(df)

    if df.empty:
        raise ValueError("No rows remain after LSTM-TrajGAN input preparation.")

    preparation_stats = {
        "source_trajectories": source_trajectories,
        "source_users": source_users,
        "dropped_core": int(dropped_core),
        "cleaned_rows": cleaned_rows,
        "cleaned_trajectories": cleaned_trajectories,
        "sample_available": int(sample_available),
        "sample_selected": int(sample_selected),
        "unknown_category_rows": int(unknown_category_rows),
        "rows_removed_by_category": int(rows_removed_by_category),
    }

    return (
        df,
        category_mapping,
        label_mapping,
        tid_mapping,
        preparation_stats,
    )


def split_explicit_inputs(df):
    """Recover the original train/test membership after shared preprocessing."""
    train_df = df[df["__input_split"] == "train"].copy()
    test_df = df[df["__input_split"] == "test"].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Explicit train/test inputs must both retain at least one "
            "trajectory after preprocessing."
        )

    validate_no_trajectory_overlap(
        train_df,
        test_df,
        "after preprocessing",
    )

    return train_df, test_df


def build_train_test_split(
    df,
    explicit_split,
    train_ratio,
    random_state,
    logger,
):
    """Build or recover the train/test split and log its final size."""
    if explicit_split:
        train_df, test_df = split_explicit_inputs(df)
        logger.info(
            "Using explicit train/test input files; no random split applied."
        )
    else:
        train_df, test_df = split_by_trajectory(
            df,
            train_ratio=train_ratio,
            random_state=random_state,
        )

    logger.info(
        "Trajectory split: train=%d trajectories / %d rows | "
        "test=%d trajectories / %d rows",
        train_df["tid"].nunique(),
        len(train_df),
        test_df["tid"].nunique(),
        len(test_df),
    )

    return train_df, test_df


def prepare_model_outputs(df, train_df, test_df, logger):
    """Build normalized model frames and trajectory-length metadata."""
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
        "Trajectory lengths: min=%d | median=%.1f | mean=%.2f | "
        "p95=%.1f | p99=%.1f | max=%d",
        overall_stats["min"],
        overall_stats["median"],
        overall_stats["mean"],
        overall_stats["p95"],
        overall_stats["p99"],
        overall_stats["max"],
    )
    logger.info(
        "Split max lengths: train=%d | test=%d",
        train_stats["max"],
        test_stats["max"],
    )

    normalization = {
        "lat_centroid": lat_centroid,
        "lon_centroid": lon_centroid,
        "scale_factor": scale_factor,
        "encoded_coordinate_definition": "raw_coordinate_minus_global_centroid",
    }
    trajectory_stats = {
        "max_length": max_length,
        "overall": overall_stats,
        "train": train_stats,
        "test": test_stats,
    }

    return (
        train_raw,
        test_raw,
        train_encoded,
        test_encoded,
        normalization,
        trajectory_stats,
    )


def build_output_paths(out_dir, dataset_name):
    """Return all artifact paths produced by this adapter."""
    return {
        "train_latlon": out_dir / "{}_train_latlon.csv".format(dataset_name),
        "test_latlon": out_dir / "{}_test_latlon.csv".format(dataset_name),
        "train_encoded": out_dir
        / "{}_dev_train_encoded_final.csv".format(dataset_name),
        "test_encoded": out_dir
        / "{}_dev_test_encoded_final.csv".format(dataset_name),
        "metadata": out_dir
        / "{}_lstm_trajgan_metadata.json".format(dataset_name),
        "category_mapping": out_dir
        / "{}_lstm_trajgan_category_mapping.csv".format(dataset_name),
        "label_mapping": out_dir
        / "{}_lstm_trajgan_label_mapping.csv".format(dataset_name),
        "tid_mapping": out_dir
        / "{}_lstm_trajgan_tid_mapping.csv".format(dataset_name),
    }


def write_build_artifacts(
    paths,
    train_raw,
    test_raw,
    train_encoded,
    test_encoded,
    category_mapping,
    label_mapping,
    tid_mapping,
):
    """Write all non-metadata build artifacts."""
    write_csv(train_raw, paths["train_latlon"])
    write_csv(test_raw, paths["test_latlon"])
    write_csv(train_encoded, paths["train_encoded"])
    write_csv(test_encoded, paths["test_encoded"])
    write_csv(category_mapping, paths["category_mapping"])
    write_csv(label_mapping, paths["label_mapping"])
    write_csv(tid_mapping, paths["tid_mapping"])


def category_mapping_records(category_mapping):
    """Convert the category mapping dataframe into JSON-safe records."""
    records = []

    for row in category_mapping.itertuples(index=False):
        records.append(
            {
                "source_category": str(row.category),
                "category": int(row.category_id),
                "rows": int(row.count),
            }
        )

    return records


def build_metadata(
    dataset_name,
    input_mode,
    input_path,
    train_input_path,
    test_input_path,
    explicit_split,
    train_ratio,
    random_state,
    category_mode,
    constant_category,
    max_trajectories,
    source_rows,
    preparation_stats,
    df,
    train_df,
    test_df,
    category_mapping,
    trajectory_stats,
    normalization,
    paths,
    log_path,
    timings,
):
    """Build the JSON metadata document for a successful adapter run."""
    sample_available = preparation_stats["sample_available"]

    return {
        "dataset_name": dataset_name,
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "input_mode": input_mode,
        "source_path": str(input_path) if input_path is not None else None,
        "source_paths": {
            "train": (
                str(train_input_path)
                if train_input_path is not None
                else None
            ),
            "test": (
                str(test_input_path)
                if test_input_path is not None
                else None
            ),
        },
        "parameters": {
            "train_ratio": float(train_ratio) if not explicit_split else None,
            "random_state": int(random_state),
            "category_mode": category_mode,
            "constant_category": int(constant_category),
            "category_vocab_size": CATEGORY_VOCAB_SIZE,
            "max_trajectories": max_trajectories,
            "split_unit": (
                "pre_split_files"
                if explicit_split
                else "trajectory_tid"
            ),
        },
        "source_summary": {
            "rows": int(source_rows),
            "trajectories": preparation_stats["source_trajectories"],
            "users": preparation_stats["source_users"],
            "dropped_invalid_core_rows": preparation_stats["dropped_core"],
            "cleaned_rows": preparation_stats["cleaned_rows"],
            "cleaned_trajectories": preparation_stats["cleaned_trajectories"],
        },
        "sampling_summary": {
            "available_trajectories": sample_available,
            "selected_trajectories": preparation_stats["sample_selected"],
            "limit_applied": (
                max_trajectories is not None
                and max_trajectories < sample_available
            ),
        },
        "category_summary": {
            "mode": category_mode,
            "excluded_unknown_category_rows": (
                preparation_stats["unknown_category_rows"]
            ),
            "removed_outside_top_categories": (
                preparation_stats["rows_removed_by_category"]
            ),
        },
        "prepared_summary": {
            "rows": int(len(df)),
            "trajectories": int(df["tid"].nunique()),
            "users": int(df["label"].nunique()),
            "categories_present": int(df["category"].nunique()),
        },
        "split_summary": {
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_trajectories": int(train_df["tid"].nunique()),
            "test_trajectories": int(test_df["tid"].nunique()),
            "trajectory_overlap": 0,
        },
        "trajectory_lengths": trajectory_stats,
        "normalization": normalization,
        "categories": category_mapping_records(category_mapping),
        "outputs": {name: str(path) for name, path in paths.items()},
        "log_file": str(log_path),
        "timing_seconds": timings,
    }


def log_build_summary(
    logger,
    df,
    train_df,
    test_df,
    category_mode,
    trajectory_stats,
    total_elapsed,
    paths,
    log_path,
):
    """Log the concise final build summary."""
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
    logger.info("Category mode: %s", category_mode)
    logger.info("Categories present: %d", df["category"].nunique())
    logger.info("Cached max_length: %d", trajectory_stats["max_length"])
    logger.info("Train rows: %d", len(train_df))
    logger.info("Test rows: %d", len(test_df))
    logger.info("Trajectory overlap: 0")
    logger.info("Total elapsed: %.2f s", total_elapsed)
    logger.info("Artifacts:")

    for name, path in paths.items():
        logger.info("  %-18s %s", name + ":", path)

    logger.info("  %-18s %s", "log:", log_path)
    logger.info("Build completed successfully.")


def main():
    """Run the LSTM-TrajGAN input build."""
    args = parse_args()

    # Keep argparse at the CLI boundary. From this point onward, use typed,
    # named values so the build helpers do not depend on argparse.Namespace.
    input_path = args.input_path  # type: Optional[Path]
    train_input_path = args.train_input_path  # type: Optional[Path]
    test_input_path = args.test_input_path  # type: Optional[Path]
    out_dir = args.out_dir  # type: Path
    dataset_name = args.dataset_name  # type: str
    train_ratio = args.train_ratio  # type: float
    random_state = args.random_state  # type: int
    category_mode = args.category_mode  # type: str
    constant_category = args.constant_category  # type: int
    max_trajectories = args.max_trajectories  # type: Optional[int]
    log_dir_arg = args.log_dir  # type: Optional[Path]

    validate_options(
        input_path=input_path,
        train_input_path=train_input_path,
        test_input_path=test_input_path,
        train_ratio=train_ratio,
        constant_category=constant_category,
        max_trajectories=max_trajectories,
    )

    explicit_split, input_mode = resolve_input_mode(train_input_path)
    total_start = time.perf_counter()

    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = log_dir_arg if log_dir_arg is not None else out_dir / "logs"
    logger, log_path = setup_logging(log_dir, dataset_name)
    timings = {}

    log_build_configuration(
        logger=logger,
        dataset_name=dataset_name,
        input_mode=input_mode,
        input_path=input_path,
        train_input_path=train_input_path,
        test_input_path=test_input_path,
        out_dir=out_dir,
        train_ratio=train_ratio,
        explicit_split=explicit_split,
        random_state=random_state,
        category_mode=category_mode,
        constant_category=constant_category,
        max_trajectories=max_trajectories,
        log_path=log_path,
    )

    with timed_stage(logger, timings, "load_source"):
        source_df = load_input_data(
            input_path=input_path,
            train_input_path=train_input_path,
            test_input_path=test_input_path,
            explicit_split=explicit_split,
            category_mode=category_mode,
        )
        source_rows = int(len(source_df))
        logger.info(
            "Loaded %d rows and %d columns.",
            source_rows,
            len(source_df.columns),
        )

    with timed_stage(logger, timings, "clean_and_prepare"):
        (
            df,
            category_mapping,
            label_mapping,
            tid_mapping,
            preparation_stats,
        ) = prepare_source_data(
            source_df=source_df,
            max_trajectories=max_trajectories,
            random_state=random_state,
            category_mode=category_mode,
            constant_category=constant_category,
            logger=logger,
        )

    with timed_stage(logger, timings, "trajectory_split"):
        train_df, test_df = build_train_test_split(
            df=df,
            explicit_split=explicit_split,
            train_ratio=train_ratio,
            random_state=random_state,
            logger=logger,
        )

    with timed_stage(logger, timings, "normalization_and_frames"):
        (
            train_raw,
            test_raw,
            train_encoded,
            test_encoded,
            normalization,
            trajectory_stats,
        ) = prepare_model_outputs(
            df=df,
            train_df=train_df,
            test_df=test_df,
            logger=logger,
        )

    paths = build_output_paths(
        out_dir=out_dir,
        dataset_name=dataset_name,
    )

    with timed_stage(logger, timings, "write_artifacts"):
        write_build_artifacts(
            paths=paths,
            train_raw=train_raw,
            test_raw=test_raw,
            train_encoded=train_encoded,
            test_encoded=test_encoded,
            category_mapping=category_mapping,
            label_mapping=label_mapping,
            tid_mapping=tid_mapping,
        )

    total_elapsed = time.perf_counter() - total_start
    timings["total"] = round(total_elapsed, 3)

    metadata = build_metadata(
        dataset_name=dataset_name,
        input_mode=input_mode,
        input_path=input_path,
        train_input_path=train_input_path,
        test_input_path=test_input_path,
        explicit_split=explicit_split,
        train_ratio=train_ratio,
        random_state=random_state,
        category_mode=category_mode,
        constant_category=constant_category,
        max_trajectories=max_trajectories,
        source_rows=source_rows,
        preparation_stats=preparation_stats,
        df=df,
        train_df=train_df,
        test_df=test_df,
        category_mapping=category_mapping,
        trajectory_stats=trajectory_stats,
        normalization=normalization,
        paths=paths,
        log_path=log_path,
        timings=timings,
    )

    # Metadata is written last so it describes only a successfully completed build.
    write_json(metadata, paths["metadata"])

    log_build_summary(
        logger=logger,
        df=df,
        train_df=train_df,
        test_df=test_df,
        category_mode=category_mode,
        trajectory_stats=trajectory_stats,
        total_elapsed=total_elapsed,
        paths=paths,
        log_path=log_path,
    )


if __name__ == "__main__":
    main()
