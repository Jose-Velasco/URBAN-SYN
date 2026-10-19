#!/usr/bin/env python3

"""
Convert model-generated GPS trajectories into FMM point-CSV input.

This script prepares GPS trajectories produced by existing models for map
matching against a previously constructed road network. It operates
independently of prepare_fmm_inputs.py and does not download OSM data,
rebuild the road network, or regenerate FMM network artifacts.

FMM input format
----------------
The output is a semicolon-delimited CSV with the following schema:

    id;x;y;timestamp

    id        Integer trajectory ID assigned for FMM.
    x         Longitude (WGS84).
    y         Latitude (WGS84).
    timestamp Integer timestamp or synthetic sequence index.

FMM requires input rows to be sorted by trajectory ID and timestamp.
All points belonging to the same trajectory must be contiguous, and
their order must reflect the original movement sequence.

Point ordering and timestamps
-----------------------------
1. When --timestamp-col is provided:
   - Numeric values are converted using --timestamp-unit.
   - Datetime-like values are converted to Unix seconds.
   - Points are sorted by trajectory ID and timestamp.
   - --order-col, if provided, breaks timestamp ties.

2. When --timestamp-col is omitted:
   - Points are grouped by trajectory ID.
   - --order-col determines point order when provided.
   - Otherwise, the original input row order is preserved.
   - Synthetic timestamps (0, 1, 2, ...) are assigned independently
     within each trajectory after ordering.

IMPORTANT:
    Synthetic timestamps encode sequence order, not elapsed time.
    They are suitable for geometric map matching (e.g., cpath/opath)
    but must not be interpreted as actual observation times or used
    for temporal evaluation, speed, or duration calculations.

    When neither --timestamp-col nor --order-col is provided, the
    input must already preserve the correct point order within each
    trajectory. The script cannot reconstruct a shuffled sequence.

Trajectory ID mapping
---------------------
Original model trajectory identifiers are mapped to integer IDs for
FMM compatibility. An optional mapping CSV preserves the relationship
between these assigned IDs and the original identifiers.

Examples
--------
LSTM-TrajGAN output (preserving input sequence order):

    uv run prepare_model_fmm_points.py \\
        --input ./lstm_trajgan/outputs/lstm_trajgan_nyc_full/results/nyc_syn_traj_test.csv \\
        --traj-id-col tid \\
        --lat-col lat \\
        --lon-col lon \\
        --gps-output ./data/lstm_trajgan_syn_fmm_ready.csv \\
        --trip-id-map-output ./data/lstm_trajgan_syn_fmm_trip_id_map.csv \\
        --log-file ./outputs/logs/prepare_lstm_trajgan_fmm.log

Model output with an explicit sequence column:

    uv run prepare_model_fmm_points.py \\
        --input generated.csv \\
        --traj-id-col traj_id \\
        --lat-col lat \\
        --lon-col lon \\
        --order-col seq \\
        --gps-output ./data/generated_fmm_ready.csv \\
        --trip-id-map-output ./data/generated_fmm_trip_id_map.csv
"""


from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an existing model GPS output into FMM's point-CSV format "
            "without rebuilding the road network."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV or Parquet containing one row per GPS point.",
    )
    parser.add_argument(
        "--gps-output",
        type=Path,
        required=True,
        help="Output FMM point CSV (semicolon-delimited id;x;y;timestamp).",
    )
    parser.add_argument(
        "--trip-id-map-output",
        type=Path,
        default=None,
        help=(
            "Optional CSV mapping original trajectory IDs to temporary integer "
            "FMM IDs."
        ),
    )

    parser.add_argument(
        "--traj-id-col",
        type=str,
        default="traj_id",
        help="Input trajectory-ID column. Default: traj_id",
    )
    parser.add_argument(
        "--lat-col",
        type=str,
        default="lat",
        help="Input latitude column. Default: lat",
    )
    parser.add_argument(
        "--lon-col",
        type=str,
        default="lon",
        help="Input longitude column. Default: lon",
    )

    parser.add_argument(
        "--timestamp-col",
        type=str,
        default=None,
        help=(
            "Optional input timestamp column. If omitted, synthetic sequential "
            "timestamps are generated within each trajectory."
        ),
    )
    parser.add_argument(
        "--timestamp-unit",
        choices=["auto", "s", "ms", "us", "ns"],
        default="auto",
        help=(
            "Unit for a numeric --timestamp-col. 'auto' treats numeric values "
            "as Unix seconds and datetime-like strings via pandas. Default: auto"
        ),
    )
    # --order-col tells the script which column determines the order of GPS points within each trajectory.
    #   Have actual timestamps? Use --timestamp-col instead.
    #   Have a sequence/index column? Use --order-col
    #   Already have correctly ordered rows? You don't need either; the script preserves their original order within each trajectory.
    parser.add_argument(
        "--order-col",
        type=str,
        default=None,
        help=(
            "Optional point-order column (for example seq). Used before "
            "synthetic timestamps are assigned, or as a stable tie-breaker "
            "when a timestamp column is present."
        ),
    )

    parser.add_argument(
        "--csv-sep",
        type=str,
        default=",",
        help="Input CSV delimiter. Default: ','",
    )
    parser.add_argument(
        "--drop-invalid",
        action="store_true",
        help=(
            "Drop rows with missing/invalid coordinates or trajectory IDs "
            "instead of raising an error."
        ),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional log file.",
    )

    return parser.parse_args()


def setup_logging(log_file: Path | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )


def format_duration(seconds: float) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f}"


def read_input(path: Path, csv_sep: str) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, sep=csv_sep)

    raise ValueError(
        f"Unsupported input format {suffix!r}. Use CSV or Parquet."
    )


def validate_and_clean_input(
    df: pd.DataFrame,
    traj_id_col: str,
    lat_col: str,
    lon_col: str,
    timestamp_col: str | None,
    order_col: str | None,
    drop_invalid: bool,
) -> pd.DataFrame:
    required = {traj_id_col, lat_col, lon_col}

    if timestamp_col is not None:
        required.add(timestamp_col)

    if order_col is not None:
        required.add(order_col)

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input is missing required columns: {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    columns = [traj_id_col, lat_col, lon_col]

    if timestamp_col is not None:
        columns.append(timestamp_col)

    if order_col is not None and order_col not in columns:
        columns.append(order_col)

    work = df[columns].copy()

    # Preserve original row order as a stable final tie-breaker.
    work["_source_row"] = np.arange(len(work), dtype=np.int64)

    work[lat_col] = pd.to_numeric(work[lat_col], errors="coerce")
    work[lon_col] = pd.to_numeric(work[lon_col], errors="coerce")

    invalid = (
        work[traj_id_col].isna()
        | work[lat_col].isna()
        | work[lon_col].isna()
        | ~work[lat_col].between(-90.0, 90.0)
        | ~work[lon_col].between(-180.0, 180.0)
    )

    invalid_count = int(invalid.sum())

    if invalid_count:
        if not drop_invalid:
            bad = work.loc[invalid].head(10)
            raise ValueError(
                f"Found {invalid_count} rows with invalid trajectory IDs or "
                f"coordinates. Re-run with --drop-invalid to remove them.\n"
                f"Examples:\n{bad}"
            )

        logger.warning("Dropping %d invalid GPS rows.", invalid_count)
        work = work.loc[~invalid].copy()

    if work.empty:
        raise ValueError("No valid GPS observations remain after validation.")

    return work


def assign_fmm_ids(
    df: pd.DataFrame,
    traj_id_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assign stable integer FMM IDs based on first occurrence of source trajectory ID.
    """
    df = df.copy()

    fmm_ids, unique_ids = pd.factorize(
        df[traj_id_col],
        sort=False,
    )

    df["id"] = fmm_ids.astype("int64")

    mapping = pd.DataFrame(
        {
            "source_traj_id": unique_ids,
            "id": np.arange(len(unique_ids), dtype=np.int64),
        }
    )

    return df, mapping


def _numeric_timestamp_to_seconds(
    values: pd.Series,
    unit: str,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")

    if unit in {"auto", "s"}:
        seconds = numeric
    elif unit == "ms":
        seconds = numeric / 1_000
    elif unit == "us":
        seconds = numeric / 1_000_000
    elif unit == "ns":
        seconds = numeric / 1_000_000_000
    else:
        raise ValueError(f"Unsupported timestamp unit: {unit}")

    return np.floor(seconds).astype("int64")


def convert_timestamp_column(
    values: pd.Series,
    unit: str,
) -> pd.Series:
    """
    Convert numeric Unix timestamps or datetime-like values to integer seconds.
    """
    if pd.api.types.is_numeric_dtype(values):
        return _numeric_timestamp_to_seconds(values, unit)

    # First try numeric-looking strings when the user specified a numeric unit.
    if unit != "auto":
        return _numeric_timestamp_to_seconds(values, unit)

    numeric = pd.to_numeric(values, errors="coerce")

    if numeric.notna().all():
        return np.floor(numeric).astype("int64")

    parsed = pd.to_datetime(values, errors="raise", utc=True)
    return (parsed.astype("int64") // 10**9).astype("int64")


def build_fmm_point_table(
    df: pd.DataFrame,
    traj_id_col: str,
    lat_col: str,
    lon_col: str,
    timestamp_col: str | None,
    timestamp_unit: str,
    order_col: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """
    Convert generic model GPS observations to FMM's id/x/y/timestamp schema.

    Returns:
        gps_df
        trip_id_map
        synthetic_timestamps_used
    """
    work, trip_id_map = assign_fmm_ids(
        df=df,
        traj_id_col=traj_id_col,
    )

    synthetic_timestamps_used = timestamp_col is None

    if timestamp_col is None:
        sort_columns = ["id"]

        if order_col is not None:
            sort_columns.append(order_col)

        sort_columns.append("_source_row")

        work.sort_values(
            sort_columns,
            kind="stable",
            inplace=True,
        )

        # FMM only needs integer ordering for map matching here.
        work["timestamp"] = (
            work.groupby("id", sort=False)
            .cumcount()
            .astype("int64")
        )

    else:
        work["timestamp"] = convert_timestamp_column(
            work[timestamp_col],
            timestamp_unit,
        )

        sort_columns = ["id", "timestamp"]

        if order_col is not None:
            sort_columns.append(order_col)

        sort_columns.append("_source_row")

        work.sort_values(
            sort_columns,
            kind="stable",
            inplace=True,
        )

    gps_df = pd.DataFrame(
        {
            "id": work["id"].astype("int64"),
            "x": work[lon_col].astype(float),
            "y": work[lat_col].astype(float),
            "timestamp": work["timestamp"].astype("int64"),
        }
    )

    return gps_df, trip_id_map, synthetic_timestamps_used


def validate_fmm_output(gps_df: pd.DataFrame) -> None:
    """Sanity checks required by FMM point input."""
    expected = ["id", "x", "y", "timestamp"]

    if list(gps_df.columns) != expected:
        raise AssertionError(
            f"Unexpected FMM schema: {list(gps_df.columns)}"
        )

    if gps_df.empty:
        raise ValueError("FMM point table is empty.")

    if gps_df[expected].isna().any().any():
        raise ValueError("FMM point table contains missing values.")

    # Validate that each trajectory occupies one contiguous block and that
    # timestamps are non-decreasing inside that block.
    previous_id = None
    closed_ids: set[int] = set()
    previous_timestamp = None

    for row in gps_df.itertuples(index=False):
        current_id = int(row.id)
        current_timestamp = int(row.timestamp)

        if previous_id is None or current_id != previous_id:
            if previous_id is not None:
                closed_ids.add(previous_id)

            if current_id in closed_ids:
                raise ValueError(
                    f"FMM trajectory id {current_id} is not stored contiguously."
                )

            previous_id = current_id
            previous_timestamp = current_timestamp
            continue

        if current_timestamp < previous_timestamp:
            raise ValueError(
                f"FMM trajectory id {current_id} is not sorted by timestamp."
            )

        previous_timestamp = current_timestamp


def write_outputs(
    gps_df: pd.DataFrame,
    trip_id_map: pd.DataFrame,
    gps_output: Path,
    trip_id_map_output: Path | None,
) -> None:
    gps_output.parent.mkdir(parents=True, exist_ok=True)

    gps_df.to_csv(
        gps_output,
        sep=";",
        index=False,
    )

    if trip_id_map_output is not None:
        trip_id_map_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        trip_id_map.to_csv(
            trip_id_map_output,
            index=False,
        )


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file)

    total_start = time.perf_counter()

    logger.info("Reading model GPS observations from %s...", args.input)
    df = read_input(args.input, args.csv_sep)

    logger.info(
        "Loaded %d rows with columns: %s",
        len(df),
        list(df.columns),
    )

    df = validate_and_clean_input(
        df=df,
        traj_id_col=args.traj_id_col,
        lat_col=args.lat_col,
        lon_col=args.lon_col,
        timestamp_col=args.timestamp_col,
        order_col=args.order_col,
        drop_invalid=args.drop_invalid,
    )

    gps_df, trip_id_map, synthetic_timestamps_used = build_fmm_point_table(
        df=df,
        traj_id_col=args.traj_id_col,
        lat_col=args.lat_col,
        lon_col=args.lon_col,
        timestamp_col=args.timestamp_col,
        timestamp_unit=args.timestamp_unit,
        order_col=args.order_col,
    )

    validate_fmm_output(gps_df)

    write_outputs(
        gps_df=gps_df,
        trip_id_map=trip_id_map,
        gps_output=args.gps_output,
        trip_id_map_output=args.trip_id_map_output,
    )

    logger.info("Saved FMM GPS input: %s", args.gps_output)

    if args.trip_id_map_output is not None:
        logger.info(
            "Saved trajectory-ID mapping: %s",
            args.trip_id_map_output,
        )

    logger.info("GPS points: %d", len(gps_df))
    logger.info("Trajectories: %d", len(trip_id_map))

    if synthetic_timestamps_used:
        logger.warning(
            "No --timestamp-col was supplied. Synthetic sequential timestamps "
            "were generated only to preserve point order. Do not interpret "
            "FMM duration/speed outputs as real temporal measurements."
        )

    logger.info(
        "Completed in %s",
        format_duration(time.perf_counter() - total_start),
    )


if __name__ == "__main__":
    main()
