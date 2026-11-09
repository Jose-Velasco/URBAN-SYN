"""Resample trajectories before passing them to the shared FMM GPS pipeline.

This reproduces 15-second, non-overlapping 20-100-point sampling
without building a road network, assigning FMM IDs, or running FMM.
Run from fmm_scripts/: python -m urban_syn_extension.sample_trajectories

Keeping the last observation is a deliberate preprocessing policy, not necessarily the most geographically accurate choice. 
If two different positions share a timestamp, we cannot know which one is correct from the timestamp alone.

Timestamp handling:
    Duplicate timestamps are resolved before interpolation by keeping the
    last observation at each timestamp.

    This intentionally differs from the original sampling implementation,
    which passes duplicate timestamps directly to np.interp().

    np.interp() expects strictly increasing source timestamps. Duplicate
    timestamps make interpolation ambiguous when different GPS coordinates
    exist at the same time.

    Example:
        Time (s)    Latitude
        0           40.0
        15          40.1
        15          40.2
        30          40.3

    At t=15, two different positions exist. Interpolation cannot uniquely
    determine which observation represents that timestamp.

    Keeping the last observation produces:
        Time (s)    Latitude
        0           40.0
        15          40.2
        30          40.3

    This ensures strictly increasing timestamps and deterministic
    interpolation, although it may change the resulting GPS positions
    compared with processing duplicate timestamps directly.

Example usage:

uv run python -m urban_syn_extension.sample_trajectories \
  --config ./urban_syn_extension/config/nyc.yaml \
  --trajectory_parquet ../data/nyc_output_tabular/output/enriched_moves.parquet \
  --sampled_parquet ./data/nyc_15s_sampled_trajectories.parquet \
  --log_file ./outputs/logs/urban_syn_extension_sample_trajectories.log
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from utils import load_config, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse sampling inputs and output location."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--trajectory_parquet", required=True, type=Path)
    parser.add_argument("--sampled_parquet", required=True, type=Path)
    parser.add_argument("--log_file", type=Path, default=None)
    return parser.parse_args()


def _read_observations(
    parquet_path: Path,
    group_columns: list[str],
    column_map: dict[str, str],
) -> pd.DataFrame:
    """Load required trajectory fields, including identifiers stored in the index.

    Args:
        column_map: Internal field -> source field, when names differ.
    """
    df = pd.read_parquet(parquet_path)
    required = set(group_columns) | {"datetime", "lat", "lng"}
    sources = {column_map.get(col, col) for col in required}

    if not sources.issubset(df.columns):
        df = df.reset_index()

    missing = sources - set(df.columns)
    if missing:
        raise ValueError(f"Missing input trajectory columns: {sorted(missing)}")

    df = df.rename(columns={source: target for target, source in column_map.items()})
    if not df.columns.is_unique:
        raise ValueError("Column mapping produced duplicate field names.")

    if df.empty or df[group_columns].isna().any().any():
        raise ValueError("Trajectory identities must be present and non-null.")

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="raise")
    df["lat"] = pd.to_numeric(df["lat"], errors="raise")
    df["lng"] = pd.to_numeric(df["lng"], errors="raise")
    coords = df[["lat", "lng"]].to_numpy(dtype=float)
    if (
        df["datetime"].isna().any()
        or not np.isfinite(coords).all()
        or not df["lat"].between(-90, 90).all()
        or not df["lng"].between(-180, 180).all()
    ):
        raise ValueError("Input contains missing timestamps or invalid coordinates.")

    return df


def _validate_sampling_settings(settings: dict) -> tuple[list[str], int, int, int]:
    """Validate sampling configuration and return its core settings."""
    group_columns = list(settings.get("group_columns", ["uid", "tid"]))
    interval = settings.get("interval_seconds", 15)
    min_points = settings.get("min_points", 20)
    max_points = settings.get("max_points", 100)

    if (
        not group_columns
        or len(set(group_columns)) != len(group_columns)
        or "piece" in group_columns
    ):
        raise ValueError("group_columns must identify a trajectory without 'piece'.")

    if (
        type(interval) is not int or type(min_points) is not int
        or type(max_points) is not int
        or interval <= 0 or min_points < 2 or max_points < min_points
    ):
        raise ValueError("Require positive integer interval and 2 <= min_points <= max_points.")

    return group_columns, interval, min_points, max_points


def _sample_group(
    group: pd.DataFrame,
    group_columns: list[str],
    interval_seconds: int,
    min_points: int,
    max_points: int,
) -> list[pd.DataFrame]:
    """Interpolate ONE trajectory based on the grouping, then produce non-overlapping pieces.

       trajectory chunks less than min_points are dropped completely, though we might want to keep them to not lose data.

        Returns: pieces list[pd.DataFrame] where the single trajectory's time was interpolated into interval_seconds
                 and then chunked into chunks between lengths of min_points and max_point
    """
    # np.interp() expects strictly increasing timestamps. Duplicate timestamps
    # can represent conflicting GPS positions, making interpolation ambiguous.
    # Keep the last observation for each timestamp to ensure deterministic input.
    move = (
        group.sort_values("datetime", kind="stable")
        .drop_duplicates("datetime", keep="last")
    )
    if len(move) < 2:
        return []

    # since sorted by datetime ascending index 0 is the start time
    start_time = move["datetime"].iloc[0]
    # normalize datetime of the trajectory where the earliest observed time is 0 seconds
    # and source_seconds holds all of times in seconds relative to the "new" 0 second of a single trajectory 
    source_seconds = (move["datetime"] - start_time).dt.total_seconds().to_numpy()
    # since source_seconds is sorted by time index [-1] is the largest seconds from the "new" 0 second
    # thus chunk the max seconds into interval_seconds integer intervals
    # thus each intervals is the new total number samples in the trajectory
    num_samples = int(source_seconds[-1] // interval_seconds) + 1
    # basally cumulative seconds from the first sample to the last each increased by interval_seconds
    sample_seconds = np.arange(num_samples, dtype=np.int64) * interval_seconds

    # interpolate on the "cumulative seconds from the first sample to the last each increased by interval_seconds"
    # each lat/lng gets a interpolated value interpolated over sample_seconds
    latitude = np.interp(sample_seconds, source_seconds, move["lat"].to_numpy())
    longitude = np.interp(sample_seconds, source_seconds, move["lng"].to_numpy())
    # used to convert back form the normalized "new" 0 back to the original datetime times but interpolated times
    # each index is each lat/lng interpolated time
    timestamps = start_time + pd.to_timedelta(sample_seconds, unit="s")

    # Preserve only metadata whose meaning is stable for an entire move aka single trajectory.
    metadata = {}
    for column in ("label", "split"):
        if column in move:
            values = move[column].drop_duplicates()
            if len(values) != 1 or values.isna().any():
                raise ValueError(f"Inconsistent {column!r} within a trajectory.")
            metadata[column] = values.iloc[0]

    pieces = []
    for piece_number, start in enumerate(range(0, num_samples, max_points)):
        end = min(start + max_points, num_samples)
        if end - start < min_points:
            continue

        sampled = pd.DataFrame({
            "datetime": timestamps[start:end],
            "lat": latitude[start:end],
            "lng": longitude[start:end],
        })
        # EX. group_columns = ["uid", "tid"] then add the columns and values back into the trajectory chunk
        for column in group_columns:
            sampled[column] = move[column].iloc[0]

        # basically is the chunk number, the trajectory is chunked and "piece" can be used to sort chunks into their original order
        sampled["piece"] = piece_number
        # added metadata back
        for column, value in metadata.items():
            sampled[column] = value
        pieces.append(sampled)

    return pieces


def sample_trajectories(
    observations: pd.DataFrame,
    settings: dict,
) -> pd.DataFrame:
    """Sample grouped trajectories into fixed-interval GPS pieces.

    Returns:
        Observations with original trajectory keys, a ``piece`` column,
        and sampled timestamps/coordinates; FMM IDs are assigned later.
    """
    groups, interval, min_points, max_points = _validate_sampling_settings(settings)

    if "allowed_labels" in settings and settings["allowed_labels"] is not None:
        if "label" not in observations.columns:
            raise ValueError("allowed_labels requires a 'label' column.")
        observations = observations.loc[
            observations["label"].isin(settings["allowed_labels"])
        ]

    pieces = []
    for _, group in observations.groupby(groups, sort=False):
        # in the case of grouping by groups = ["uid", "tid"] then it can still have a piece columns
        # since a long trajectory get chunked piece is the chunk number
        pieces.extend(_sample_group(group, groups, interval, min_points, max_points))

    if not pieces:
        raise ValueError("No trajectory pieces meet the configured sampling limits.")

    result = pd.concat(pieces, ignore_index=True)
    columns = [*groups, "piece", "datetime", "lat", "lng"]
    columns.extend(column for column in ("label", "split") if column in result)
    return result[columns]


def main() -> None:
    """Write sampled GPS observations for the shared FMM GPS preparation script."""
    args = parse_args()
    setup_logging(args.log_file)

    settings = load_config(args.config)["sampling"]
    group_columns, _, _, _ = _validate_sampling_settings(settings)
    observations = _read_observations(
        args.trajectory_parquet,
        group_columns,
        settings.get("columns", {}),
    )
    sampled = sample_trajectories(observations, settings)

    args.sampled_parquet.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_parquet(args.sampled_parquet, index=False)
    
    logger.info(
        "Saved %d sampled GPS points across %d pieces: %s",
        len(sampled),
        len(sampled[group_columns + ["piece"]].drop_duplicates()),
        args.sampled_parquet,
    )


if __name__ == "__main__":
    main()
