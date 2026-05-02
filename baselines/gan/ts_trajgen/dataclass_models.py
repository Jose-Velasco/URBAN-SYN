from dataclasses import dataclass
from pathlib import Path
import pandas as pd

import pandas as pd
import logging
from enum import StrEnum


@dataclass(frozen=True)
class EdgeTimePoint:
    """
    Represents one road edge with its assigned timestamp.

    Attributes
    ----------
    edge_id : int
        Road segment ID in geo_id space.
    timestamp : pd.Timestamp
        Timestamp assigned to this road segment.
    is_anchor : bool
        True if timestamp comes directly from GPS; False if interpolated.
    """

    edge_id: int
    timestamp: pd.Timestamp
    is_anchor: bool

class DuplicateAction(StrEnum):
    """Actions used when resolving consecutive duplicate edge IDs."""

    APPEND = "append"
    DROP_CURRENT = "drop_current"
    REPLACE_PREVIOUS = "replace_previous"

@dataclass(frozen=True)
class BuildConfig:
    """
    Parsed CLI configuration for building TS-TrajGen input files.
    """

    network_path: Path
    fmm_match_path: Path
    parquet_path: Path
    trip_id_map_csv: Path
    out_dir: Path
    log_dir: Path
    dataset_name: str
    train_ratio: float
    random_state: int
    min_len: int
    fmm_sep: str
    verbose: bool
    min_delta_seconds: float

@dataclass
class InterpolationStats:
    """
    Tracks interpolation and duplicate-resolution behavior during preprocessing.
    """

    total_segments: int = 0
    fallback_segments: int = 0
    non_positive_time_segments: int = 0
    short_time_segments: int = 0
    invalid_length_segments: int = 0

    duplicate_current_dropped: int = 0
    duplicate_previous_replaced: int = 0
    duplicate_anchor_same_time_dropped: int = 0

    def fallback_rate(self) -> float:
        """Return the fraction of segments that used fallback interpolation."""
        if self.total_segments == 0:
            return 0.0
        return self.fallback_segments / self.total_segments

    def log_summary(self, logger: logging.Logger) -> None:
        """Log a compact interpolation and duplicate-resolution summary."""
        logger.info("=== INTERPOLATION SUMMARY ===")
        logger.info(f"Total segments:                    {self.total_segments:,}")
        logger.info(f"Fallback segments:                 {self.fallback_segments:,}")
        logger.info(f"Fallback rate:                     {self.fallback_rate():.3%}")
        logger.info(f"Zero/reversed-time segments:       {self.non_positive_time_segments:,}")
        logger.info(f"Short-time segments:               {self.short_time_segments:,}")
        logger.info(f"Invalid-length segments:           {self.invalid_length_segments:,}")
        logger.info(f"Duplicate current dropped:         {self.duplicate_current_dropped:,}")
        logger.info(f"Duplicate previous replaced:       {self.duplicate_previous_replaced:,}")
        logger.info(f"Duplicate anchor same-time dropped:{self.duplicate_anchor_same_time_dropped:,}")

@dataclass
class BuildCsvStats:
    """
    Tracks trajectory-level preprocessing counts for build_mm_csvs().
    """

    total_rows: int = 0
    kept_rows: int = 0
    skipped_empty_tpath: int = 0
    skipped_missing_times: int = 0
    skipped_short_path: int = 0

    @property
    def skipped_total(self) -> int:
        """Return total skipped trajectory count."""
        return (
            self.skipped_empty_tpath
            + self.skipped_missing_times
            + self.skipped_short_path
        )

    def kept_ratio(self) -> float:
        """Return fraction of FMM rows kept as valid trajectories."""
        if self.total_rows == 0:
            return 0.0
        return self.kept_rows / self.total_rows

    def log_summary(
        self,
        logger: logging.Logger,
        train_size: int,
        test_size: int,
    ) -> None:
        """Log final build summary."""
        logger.info("=== MM CSV BUILD STATS ===")
        logger.info(f"Total input trajectories: {self.total_rows:,}")
        logger.info(f"Kept trajectories:        {self.kept_rows:,}")
        logger.info(f"Kept ratio:               {self.kept_ratio():.3f}")
        logger.info(f"Skipped trajectories:     {self.skipped_total:,}")
        logger.info("Skipped breakdown:")
        logger.info(f"  Empty tpath:            {self.skipped_empty_tpath:,}")
        logger.info(f"  Missing GPS times:      {self.skipped_missing_times:,}")
        logger.info(f"  Too short final path:   {self.skipped_short_path:,}")
        logger.info(f"Train size:               {train_size:,}")
        logger.info(f"Test size:                {test_size:,}")