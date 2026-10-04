# NOTE:
# We do NOT extend timestamps beyond the original GPS interval.
# Real data may contain repeated timestamps (second-level resolution),
# so we preserve anchors and interpolate within bounds using
# sub-second precision instead of enforcing artificial spacing.

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
import logging

from dataclass_models import BuildConfig
from utils import build_geo_and_length_lookups, build_mm_csvs, build_rel, build_trip_time_lookup, ensure_dir, parse_osm_width, setup_logger
from typing import Any
import pandas as pd

import yaml

def _load_yaml_config(config_path: Path | None) -> dict[str, Any]:
    """Load a YAML configuration file, returning an empty config if omitted."""
    if config_path is None:
        return {}

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}

def parse_args() -> BuildConfig:
    """
    Parse command-line arguments and return a typed build configuration.
    """
    parser = argparse.ArgumentParser(description="Build TS-TrajGen inputs")

    parser.add_argument(
        "--network_path",
        type=Path,
        required=True,
        help=(
            "Path to the canonical road-network GeoPackage with at least "
            "[edge_id, u, v, geometry]."
        ),
    )
    parser.add_argument(
        "--fmm_match_path",
        type=Path,
        required=True,
        help="Path to FMM output CSV containing id, cpath, and tpath columns.",
    )
    parser.add_argument(
        "--parquet_path",
        type=Path,
        required=True,
        help="Path to original cleaned parquet with datetime, user, and traj_id columns.",
    )
    parser.add_argument(
        "--trip_id_map_csv",
        type=Path,
        required=True,
        help="Path to the FMM trip-id map CSV created when preparing GPS points for FMM.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=True,
        help="Path to TS-TrajGen YAML configuration.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("./outputs/nyc"),
        help="Output directory for generated files (.geo, .rel, *_mm_train/test.csv).",
    )
    parser.add_argument(
        "--log_dir",
        type=Path,
        default=Path("./outputs/logs"),
        help="Directory where timestamped log files will be saved.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="nyc",
        help="Prefix for output files (e.g., nyc -> nyc.geo, nyc.rel).",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Fraction of trajectories used for training.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=101,
        help="Random seed for reproducible train/test splitting.",
    )
    parser.add_argument(
        "--min_len",
        type=int,
        default=2,
        help="Minimum trajectory length (in edges); sequences with fewer edges are skipped.",
    )
    # NOTE: no longer enforced but will warn in logs because we do not want to distort real GPS point data
    # NOTE: Short-time segments are warnings about resolution limits, not data quality issues.
    # Short GPS intervals (where total duration is insufficient for the DESIRED minimum per-edge spacing) are preserved and interpolated within the original time bounds.
    # The min_delta_seconds parameter is used as a diagnostic threshold rather than a strict constraint, ensuring no synthetic time extension is introduced.
    parser.add_argument(
        "--min_delta_seconds",
        type=float,
        default=0.5,
        help=(
            "Diagnostic minimum time spacing (seconds) used to flag short "
            "GPS intervals; timestamps are never extended beyond GPS bounds."
        ),
    )
    parser.add_argument(
        "--fmm_sep",
        type=str,
        default=";",
        help="Delimiter used in the FMM output CSV.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed processing statistics in addition to logging.",
    )
    parser.add_argument(
        "--interpolate_intermediate_edges",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include intermediate FMM tpath road segments and interpolate their "
            "timestamps. Use --no-interpolate_intermediate_edges to keep only "
            "GPS-anchor road assignments from FMM opath."
        ),
    )
    parser.add_argument(
        "--max_train_trajectories",
        type=int,
        default=None,
        help=(
            "Optional maximum number of training trajectories to retain after "
            "the train/test split. Useful for smoke tests."
        ),
    )

    parser.add_argument(
        "--max_test_trajectories",
        type=int,
        default=None,
        help=(
            "Optional maximum number of test trajectories to retain after "
            "the train/test split. Useful for smoke tests."
        ),
    )

    args = parser.parse_args()

    yaml_config = _load_yaml_config(args.config)
    geo_feature_columns = tuple(yaml_config["data"]["geo"]["feature_columns"])
    
    return BuildConfig(
        network_path=args.network_path,
        fmm_match_path=args.fmm_match_path,
        parquet_path=args.parquet_path,
        trip_id_map_csv=args.trip_id_map_csv,
        geo_feature_columns=geo_feature_columns,
        out_dir=args.out_dir,
        log_dir=args.log_dir,
        dataset_name=args.dataset_name,
        train_ratio=args.train_ratio,
        random_state=args.random_state,
        min_len=args.min_len,
        fmm_sep=args.fmm_sep,
        min_delta_seconds=args.min_delta_seconds,
        verbose=args.verbose,
        interpolate_intermediate_edges=args.interpolate_intermediate_edges,
        max_train_trajectories=args.max_train_trajectories,
        max_test_trajectories=args.max_test_trajectories,
    )

def prepare_output_dirs(config: BuildConfig) -> None:
    """
    Create output and log directories if they do not already exist.
    """
    ensure_dir(config.out_dir)
    ensure_dir(config.log_dir)

def output_paths(config: BuildConfig) -> dict[str, Path]:
    """
    Build all output file paths in one place to avoid repeated path formatting.
    """
    prefix = config.out_dir / config.dataset_name

    return {
        "geo": prefix.with_suffix(".geo"),
        "rel": prefix.with_suffix(".rel"),
        "train": config.out_dir / f"{config.dataset_name}_mm_train.csv",
        "test": config.out_dir / f"{config.dataset_name}_mm_test.csv",
    }

def _validate_normalized_geo_features(geo_df: pd.DataFrame):
    for column in ("length", "maxspeed", "width"):
        if not pd.api.types.is_numeric_dtype(
            geo_df[column]
        ):
            raise TypeError(
                f"{column!r} must be numeric before writing .geo; "
                f"got dtype={geo_df[column].dtype}"
            )

    if geo_df["length"].isna().any():
        raise ValueError(
            "Missing road lengths found in .geo"
        )

    if geo_df["maxspeed"].isna().any():
        raise ValueError(
            "Missing OSMnx-derived maxspeed values found in .geo"
        )

def _normalize_geo_features(geo_df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    # TS-TrajGen historically expects this feature to be named `maxspeed`.
    # The canonical network preserves raw `maxspeed` separately and stores
    # OSMnx's numeric/imputed speed feature as `speed_kph`.
    if "speed_kph" not in geo_df.columns:
        raise ValueError("Missing required canonical feature: 'speed_kph'")

    geo_df = geo_df.rename(
        columns={"speed_kph": "maxspeed"}
    )

    # TS-TrajGen expects numeric width for min-max normalization.
    if "width" in geo_df.columns:
        logger.info(
            "Standardizing OSM width values to meters"
        )

        geo_df["width"] = geo_df["width"].map(
            parse_osm_width
        )

        logger.info(
            "Width values: valid=%s, missing/unparsed=%s",
            f"{geo_df['width'].notna().sum():,}",
            f"{geo_df['width'].isna().sum():,}",
        )
    return geo_df


def build_and_save_geo(
    network_path: Path,
    geo_path: Path,
    logger: logging.Logger,
    feature_columns: Iterable[str]
    ):
    """
    Build .geo data and edge lookup tables, then save the .geo file.
    """
    logger.info("Building .geo file")
    geo_df, edges_df, edge_id_to_geo_id, geo_to_length = build_geo_and_length_lookups(
        network_path,
        feature_columns
    )

    geo_df = _normalize_geo_features(geo_df, logger)
    _validate_normalized_geo_features(geo_df)

    geo_df.to_csv(geo_path, index=False)

    logger.info(f"Saved .geo file: {geo_path}")
    logger.info(f".geo rows: {len(geo_df):,}")

    return geo_df, edges_df, edge_id_to_geo_id, geo_to_length

def build_and_save_rel(
    edges_df,
    rel_path: Path,
    logger: logging.Logger,
    ):
    """
    Build .rel road adjacency data and save the .rel file.
    """
    logger.info("Building .rel file")
    rel_df = build_rel(edges_df)
    rel_df.to_csv(rel_path, index=False)

    logger.info(f"Saved .rel file: {rel_path}")
    logger.info(f".rel rows: {len(rel_df):,}")

    return rel_df

def build_and_save_mm_csvs(
    config: BuildConfig,
    edge_id_to_geo_id: dict[int, int],
    geo_to_length: dict[int, float],
    train_path: Path,
    test_path: Path,
    logger: logging.Logger,
    ):
    """
    Build map-matched train/test CSVs and save them to disk.
    """
    logger.info("Building trip timestamp lookup")
    trip_time_lookup = build_trip_time_lookup(
        parquet_path=config.parquet_path,
        trip_id_map_csv=config.trip_id_map_csv,
    )

    logger.info("Building train/test map-matched CSV files")
    train_df, test_df = build_mm_csvs(
        fmm_path=config.fmm_match_path,
        edge_id_to_geo_id=edge_id_to_geo_id,
        geo_to_length=geo_to_length,
        trip_time_lookup=trip_time_lookup,
        train_ratio=config.train_ratio,
        random_state=config.random_state,
        interpolate_intermediate_edges=config.interpolate_intermediate_edges,
        min_len=config.min_len,
        verbose=config.verbose,
        fmm_sep=config.fmm_sep,
        min_delta_seconds=config.min_delta_seconds,
        logger=logger,
        max_train_trajectories=config.max_train_trajectories,
        max_test_trajectories=config.max_test_trajectories,
    )

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Saved train CSV: {train_path}")
    logger.info(f"Saved test CSV:  {test_path}")
    logger.info(f"Train trajectories: {len(train_df):,}")
    logger.info(f"Test trajectories:  {len(test_df):,}")

    return train_df, test_df

def log_final_summary(
    geo_df,
    rel_df,
    train_df,
    test_df,
    logger: logging.Logger,
    ) -> None:
    """
    Log the final high-level build summary.
    """
    logger.info("=== BUILD COMPLETE ===")
    logger.info(f".geo rows:    {len(geo_df):,}")
    logger.info(f".rel rows:    {len(rel_df):,}")
    logger.info(f"train trajs:  {len(train_df):,}")
    logger.info(f"test trajs:   {len(test_df):,}")

def run_build(config: BuildConfig) -> None:
    """
    Run the full TS-TrajGen input build pipeline.
    """
    prepare_output_dirs(config)

    logger = setup_logger("build_tstrajgen", log_dir=config.log_dir)
    paths = output_paths(config)

    logger.info("Starting TS-TrajGen input build")
    logger.info(f"Dataset name: {config.dataset_name}")
    logger.info(f"Output directory: {config.out_dir}")
    logger.info(f"min_delta_seconds: {config.min_delta_seconds}")
    logger.info(
        "Interpolate intermediate edges: %s",
        config.interpolate_intermediate_edges,
    )
    logger.info(
        "Trajectory limits: train=%s, test=%s",
        config.max_train_trajectories,
        config.max_test_trajectories,
    )

    geo_df, edges_df, edge_id_to_geo_id, geo_to_length = build_and_save_geo(
        network_path=config.network_path,
        geo_path=paths["geo"],
        logger=logger,
        feature_columns=config.geo_feature_columns
    )

    rel_df = build_and_save_rel(
        edges_df=edges_df,
        rel_path=paths["rel"],
        logger=logger,
    )

    train_df, test_df = build_and_save_mm_csvs(
        config=config,
        edge_id_to_geo_id=edge_id_to_geo_id,
        geo_to_length=geo_to_length,
        train_path=paths["train"],
        test_path=paths["test"],
        logger=logger,
    )

    log_final_summary(
        geo_df=geo_df,
        rel_df=rel_df,
        train_df=train_df,
        test_df=test_df,
        logger=logger,
    )

def main() -> None:
    """
    CLI entry point.
    """
    config = parse_args()
    run_build(config)


if __name__ == "__main__":
    main()
