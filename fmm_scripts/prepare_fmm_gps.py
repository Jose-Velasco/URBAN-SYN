"""Create FMM GPS point CSVs without rebuilding the shared road network.

Input datasets may vary; the optional 'trajectories' YAML section maps source
columns to the common uid/tid/datetime/lat/lng schema.

Use the shared identity columns by default, but allow dataset-specific
overrides. Sampled move pieces need (uid, tid, move_id, piece) so each
piece receives a unique FMM trajectory ID.

Example usage:

uv run prepare_fmm_gps.py \
  --config ./config/nyc.yaml \
  --trajectory_parquet ./data/nyc_15s_sampled_trajectories.parquet \
  --gps_output ./data/nyc_15s_fmm_gps_points.csv \
  --trip_id_map_output ./data/nyc_15s_fmm_trip_id_map.csv \
  --identity_columns uid tid piece

^^^ above is example with traj_clean.parquet
  

uv run prepare_fmm_gps.py \
  --config ./config/nyc.yaml \
  --trajectory_parquet ./data/nyc_15s_sampled_trajectories.parquet \
  --gps_output ./data/nyc_15s_fmm_gps_points.csv \
  --trip_id_map_output ./data/nyc_15s_fmm_trip_id_map.csv \
  --identity_columns uid tid move_id piece \
  --log_file ./outputs/logs/prepare_fmm_gps.log

^^^ example using enriched_moves.parquet
"""

import argparse
import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from utils import (
    add_fmm_trajectory_ids,
    format_duration,
    load_config,
    prepare_fmm_points,
    setup_logging,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare FMM GPS inputs using an existing shared network."
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Shared dataset YAML, e.g. ./config/nyc.yaml",
    )
    parser.add_argument(
        "--trajectory_parquet", type=Path, required=True,
        help="GPS observation Parquet. Does not trigger a network download.",
    )
    parser.add_argument(
        "--gps_output", type=Path, required=True,
        help="Semicolon-delimited output CSV (fmm_id;x;y;timestamp).",
    )
    parser.add_argument(
        "--trip_id_map_output", type=Path, required=True,
        help="CSV mapping fmm_id to the source trajectory identity columns.",
    )
    parser.add_argument(
        "--identity_columns", nargs="+", default=None,
        help="Override YAML trajectory identity, e.g. uid tid piece or uid tid move_id piece.",
    )
    parser.add_argument(
        "--log_file", type=Path, default=None,
        help="Optional log file.",
    )
    return parser.parse_args()


def _rename_source_columns(
    df: pd.DataFrame,
    column_map: Mapping[str, str],
) -> pd.DataFrame:
    """Rename configured source fields to the shared GPS/identity schema."""
    if not column_map:
        return df

    # YAML maps shared names to source names, e.g. uid: user_id.
    source_columns = list(column_map.values())
    if any(source_name not in df.columns for source_name in source_columns):
        df = df.reset_index()

    missing = set(source_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")

    if len(set(source_columns)) != len(source_columns):
        raise ValueError("Each source column must map to only one target column.")

    renamed = df.rename(
        columns={source: shared for shared, source in column_map.items()}
    )
    if not renamed.columns.is_unique:
        raise ValueError("Column mapping created duplicate output columns.")
    return renamed


def build_fmm_points_csv(
    parquet_path: Path,
    out_csv: Path,
    trip_id_map_csv: Path | None = None,
    identity_columns: Sequence[str] = ("uid", "tid"),
    column_map: Mapping[str, str] | None = None,
    timestamp_unit: str | None = None,
) -> pd.DataFrame:
    """Write FMM GPS observations and preserve their original trajectory IDs.

    Produces a semicolon-delimited CSV: fmm_id, x, y, timestamp.
    The optional mapping CSV contains fmm_id plus identity columns.
    """
    total_start = time.perf_counter()

    # Read observations
    start_time = time.perf_counter()

    logger.info(
        "Reading GPS observations from %s...",
        parquet_path,
    )

    df = pd.read_parquet(parquet_path)
    df = _rename_source_columns(df, column_map or {})

    logger.info(
        "Parquet read completed in %s (%d rows)",
        format_duration(time.perf_counter() - start_time),
        len(df),
    )

    # Assign trajectory IDs
    start_time = time.perf_counter()

    logger.info(
        "Building integer FMM trajectory IDs..."
    )

    trip_id_map = add_fmm_trajectory_ids(
        df,
        identity_columns=identity_columns,
    )

    logger.info(
        "Trajectory-ID mapping completed in %s (%d trajectories)",
        format_duration(time.perf_counter() - start_time),
        len(trip_id_map),
    )

    # Prepare FMM point table
    gps_df = prepare_fmm_points(df, timestamp_unit=timestamp_unit)

    # We no longer need the larger source DataFrame.
    del df

    # Write outputs
    out_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    start_time = time.perf_counter()

    logger.info(
        "Writing GPS observations to %s...",
        out_csv,
    )

    gps_df.to_csv(
        out_csv,
        sep=";",
        index=False,
    )

    logger.info(
        "GPS CSV write completed in %s",
        format_duration(time.perf_counter() - start_time),
    )

    if trip_id_map_csv is not None:
        trip_id_map_csv.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        start_time = time.perf_counter()

        logger.info(
            "Writing trajectory ID map to %s...",
            trip_id_map_csv,
        )

        trip_id_map.to_csv(
            trip_id_map_csv,
            index=False,
        )

        logger.info(
            "Trajectory ID map write completed in %s",
            format_duration(time.perf_counter() - start_time),
        )

    logger.info(
        "Saved GPS observations: %s",
        out_csv,
    )

    logger.info(
        "GPS points: %d",
        len(gps_df),
    )

    logger.info(
        "Trajectories: %d",
        len(trip_id_map),
    )

    logger.info(
        "FMM point CSV build completed in %s",
        format_duration(time.perf_counter() - total_start),
    )

    return trip_id_map

def main() -> None:
    """Prepare GPS inputs; never download or modify road-network artifacts."""
    args = parse_args()
    setup_logging(args.log_file)

    settings = load_config(args.config).get("trajectories", {}) or {}

    build_fmm_points_csv(
        parquet_path=args.trajectory_parquet,
        out_csv=args.gps_output,
        trip_id_map_csv=args.trip_id_map_output,
        identity_columns=args.identity_columns or settings.get("identity_columns", ("uid", "tid")),
        column_map=settings.get("columns", {}),
        timestamp_unit=settings.get("timestamp_unit"),
    )


if __name__ == "__main__":
    main()
