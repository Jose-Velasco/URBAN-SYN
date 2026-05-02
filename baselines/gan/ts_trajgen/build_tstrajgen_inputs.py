# NOTE:
# We do NOT extend timestamps beyond the original GPS interval.
# Real data may contain repeated timestamps (second-level resolution),
# so we preserve anchors and interpolate within bounds using
# sub-second precision instead of enforcing artificial spacing.

from __future__ import annotations

import argparse
from pathlib import Path
import logging

from dataclass_models import BuildConfig
from utils import build_geo_and_length_lookups, build_mm_csvs, build_rel, build_trip_time_lookup, ensure_dir, setup_logger
# from utils import InterpolationStats, build_geo, build_geo_and_length_lookups, build_mm_csvs, build_rel, build_rid_and_time_lists_from_tpath, build_trip_time_lookup, ensure_dir, geometry_to_coordinate_string, parse_cpath_like, parse_tpath, setup_logger

# # HELPERS
# def ensure_dir(path: Path) -> None:
#     """
#     Create directory if it does not exist.

#     Parameters
#     ----------
#     path : Path
#         Directory path to create.
#     """
#     path.mkdir(parents=True, exist_ok=True)


# def geometry_to_coordinate_string(geom) -> str:
#     """
#     Convert a LineString geometry into a JSON string of coordinates.

#     Expected format for TS-TrajGen:
#         [[lon, lat], [lon, lat], ...]

#     Parameters
#     ----------
#     geom : shapely.geometry.LineString

#     Returns
#     -------
#     str
#         JSON string of coordinate pairs.
#     """
#     coords = [[float(x), float(y)] for x, y in geom.coords]
#     return json.dumps(coords, separators=(",", ":"))


# def parse_cpath(value) -> list[int]:
#     """
#     Parse FMM `cpath` field into a list of edge IDs (fid).

#     Handles multiple formats:
#         - "[1,2,3]"
#         - "1,2,3"
#         - "1 2 3"
#         - NaN / empty

#     Parameters
#     ----------
#     value : Any
#         Raw cpath value from FMM output.

#     Returns
#     -------
#     list[int]
#         List of edge IDs.
#     """
#     if pd.isna(value):
#         return []

#     s = str(value).strip()
#     if not s:
#         return []

#     if s.startswith("[") and s.endswith("]"):
#         try:
#             return [int(x) for x in ast.literal_eval(s)]
#         except Exception:
#             pass

#     if "," in s:
#         return [int(x.strip()) for x in s.split(",") if x.strip()]

#     return [int(x.strip()) for x in s.split() if x.strip()]

# def parse_cpath_like(value: Any) -> list[int]:
#     """
#     Parse a comma-separated or Python-list-like edge sequence.

#     Supported formats
#     -----------------
#     - "1,2,3"
#     - "[1, 2, 3]"
#     - ""
#     - NaN

#     Parameters
#     ----------
#     value : Any
#         Raw serialized edge sequence.

#     Returns
#     -------
#     list[int]
#         Parsed edge ID sequence.
#     """
#     if pd.isna(value):
#         return []

#     s = str(value).strip()
#     if not s:
#         return []

#     if s.startswith("[") and s.endswith("]"):
#         try:
#             return [int(x) for x in ast.literal_eval(s)]
#         except Exception:
#             pass

#     return [int(x.strip()) for x in s.split(",") if x.strip()]

# def parse_tpath(value: Any) -> list[list[int]]:
#     """
#     Parse FMM `tpath` into per-GPS-interval edge segments.

#     Example
#     -------
#     "2|2,5,13|13,14|14,23"
#     ->
#     [[2], [2, 5, 13], [13, 14], [14, 23]]

#     Parameters
#     ----------
#     value : Any
#         Raw tpath string from FMM output.

#     Returns
#     -------
#     list[list[int]]
#         One edge list per consecutive GPS-point interval.
#     """
#     if pd.isna(value):
#         return []

#     s = str(value).strip()
#     if not s:
#         return []

#     segments: list[list[int]] = []
#     for chunk in s.split("|"):
#         chunk = chunk.strip()
#         if not chunk:
#             segments.append([])
#             continue
#         segments.append(parse_cpath_like(chunk))

#     return segments




# def build_mm_csvs(
#     fmm_path,
#     fid_to_geo,
#     geo_to_length,
#     trip_time_lookup,
#     train_ratio,
#     random_state,
#     min_len: int = 2,
#     verbose: bool = False,
#     fmm_sep: str = ";",
# ):
#     """
#     Build TS-TrajGen train/test CSVs from FMM output using `tpath` + real timestamps.

#     tpath is added by FMM

#     geo_id is the column in .geo (we created it ourselves from the road network file)

#     Parameters
#     ----------
#     fmm_path : Path | str
#         FMM output CSV path.
#     fid_to_geo : dict[int, int]
#         Mapping from FMM fid to geo_id.
#     geo_to_length : dict[int, float]
#         Mapping from geo_id to edge length.
#     trip_time_lookup : dict[str, list[pd.Timestamp]]
#         Mapping from trajectory id to original GPS timestamps.
#     train_ratio : float
#         Fraction of trajectories used for training.
#     random_state : int
#         Seed for reproducibility.
#     min_len : int, optional
#         Minimum number of edges required in the final trajectory.
#     verbose : bool, optional
#         Print summary stats if True.
#     fmm_sep : str, optional
#         Delimiter used in the FMM CSV.

#     Returns
#     -------
#     tuple[pd.DataFrame, pd.DataFrame]
#         Train and test dataframes.
#     """
#     fmm = pd.read_csv(fmm_path, sep=fmm_sep, engine="python")

#     rows = []

#     total = 0
#     skipped_empty_tpath = 0
#     skipped_short = 0
#     skipped_missing_times = 0

#     for _, row in fmm.iterrows():
#         total += 1
#         traj_id = str(row["id"])

#         if pd.isna(row.get("tpath", "")) or str(row.get("tpath", "")).strip() == "":
#             skipped_empty_tpath += 1
#             continue

#         if traj_id not in trip_time_lookup or len(trip_time_lookup[traj_id]) < 2:
#             skipped_missing_times += 1
#             continue

#         rid_list, time_list = build_rid_and_time_lists_from_tpath(
#             traj_id=traj_id,
#             tpath_value=row["tpath"],
#             trip_time_lookup=trip_time_lookup,
#             fid_to_geo=fid_to_geo,
#             geo_to_length=geo_to_length,
#         )

#         if len(rid_list) < min_len:
#             skipped_short += 1
#             continue

#         rows.append({
#             "traj_id": traj_id,
#             "rid_list": ",".join(str(x) for x in rid_list),
#             "time_list": ",".join(time_list),
#         })

#     df = pd.DataFrame(rows)

#     train = df.sample(frac=train_ratio, random_state=random_state)
#     test = df.drop(train.index)

#     if verbose:
#         kept = len(df)
#         print("\nMM CSV BUILD STATS")
#         print(f"Total input trajectories: {total:,}")
#         print(f"Kept trajectories:        {kept:,}")
#         print(f"Kept ratio:               {kept / total:.3f}" if total else "Kept ratio:               N/A")
#         print()
#         print("Skipped breakdown:")
#         print(f"  Empty tpath:            {skipped_empty_tpath:,}")
#         print(f"  Missing GPS times:      {skipped_missing_times:,}")
#         print(f"  Too short final path:   {skipped_short:,}")
#         print()

#     return train.reset_index(drop=True), test.reset_index(drop=True)


# def build_mm_csvs(
#     fmm_path,
#     fid_to_geo,
#     train_ratio,
#     random_state,
#     min_len: int = 2,
#     verbose: bool = False,
# ):
#     """
#     Build TS-TrajGen train/test CSVs from FMM output.

#     Converts:
#         cpath (fid list) -> rid_list (geo_id list)

        
#     cpath is added by FMM

#     geo_id is the column in .geo (we created it ourselves from the road network file)
    
#     Parameters
#     ----------
#     fmm_path : Path
#         CSV file from FMM output.
#     fid_to_geo : dict
#         Mapping from fid -> geo_id.
#     train_ratio : float
#         Fraction of data for training.
#     random_state : int
#         Seed for reproducibility.
#     min_len : int, optional
#         Minimum number of edges required in a trajectory (default=2).
#         Sequences shorter than this are discarded.
#     verbose : bool, optional
#         If True, print detailed processing statistics.

#     Returns
#     -------
#     train_df : pd.DataFrame
#     test_df : pd.DataFrame
#     """

#     fmm = pd.read_csv(fmm_path, sep=";")

#     rows = []

#     # stats tracking
#     total = 0
#     skipped_empty = 0
#     skipped_short_raw = 0
#     skipped_short_mapped = 0
#     skipped_unmapped = 0

#     for _, row in fmm.iterrows():
#         total += 1

#         path = parse_cpath(row["cpath"])

#         if not path:
#             skipped_empty += 1
#             continue

#         if len(path) < min_len:
#             skipped_short_raw += 1
#             continue

#         # map fid -> geo_id
#         geo_path = []
#         unmapped_flag = False

#         for f in path:
#             if f in fid_to_geo:
#                 geo_path.append(fid_to_geo[f])
#             else:
#                 unmapped_flag = True

#         if unmapped_flag:
#             skipped_unmapped += 1

#         if len(geo_path) < min_len:
#             skipped_short_mapped += 1
#             continue

#         rows.append({
#             "traj_id": str(row["id"]),
#             "rid_list": str(geo_path)
#         })

#     df = pd.DataFrame(rows)

#     # split
#     train = df.sample(frac=train_ratio, random_state=random_state)
#     test = df.drop(train.index)

#     # verbose stats
#     if verbose:
#         kept = len(df)
#         print("\n MM CSV BUILD STATS")
#         print(f"Total input trajectories: {total:,}")
#         print(f"Kept trajectories:        {kept:,}")
#         print(f"Kept ratio:              {kept/total:.3f}")
#         print()
#         print("Skipped breakdown:")
#         print(f"  Empty paths:           {skipped_empty:,}")
#         print(f"  Too short (raw):       {skipped_short_raw:,}")
#         print(f"  Too short (mapped):    {skipped_short_mapped:,}")
#         print(f"  Had unmapped edges:    {skipped_unmapped:,}")
#         print()

#     return train.reset_index(drop=True), test.reset_index(drop=True)


def parse_args() -> BuildConfig:
    """
    Parse command-line arguments and return a typed build configuration.
    """
    parser = argparse.ArgumentParser(description="Build TS-TrajGen inputs")

    parser.add_argument(
        "--network_path",
        type=Path,
        required=True,
        help="Path to road network file (e.g., .shp/.gpkg) with columns [fid, u, v, geometry].",
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
        help="Minimum time spacing (seconds) enforced between consecutive edges during interpolation.",
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

    args = parser.parse_args()

    return BuildConfig(
        network_path=args.network_path,
        fmm_match_path=args.fmm_match_path,
        parquet_path=args.parquet_path,
        trip_id_map_csv=args.trip_id_map_csv,
        out_dir=args.out_dir,
        log_dir=args.log_dir,
        dataset_name=args.dataset_name,
        train_ratio=args.train_ratio,
        random_state=args.random_state,
        min_len=args.min_len,
        fmm_sep=args.fmm_sep,
        min_delta_seconds=args.min_delta_seconds,
        verbose=args.verbose,
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

def build_and_save_geo(
    network_path: Path,
    geo_path: Path,
    logger: logging.Logger,
    ):
    """
    Build .geo data and edge lookup tables, then save the .geo file.
    """
    logger.info("Building .geo file")
    geo_df, edges_df, fid_to_geo, geo_to_length = build_geo_and_length_lookups(
        network_path
    )
    geo_df.to_csv(geo_path, index=False)

    logger.info(f"Saved .geo file: {geo_path}")
    logger.info(f".geo rows: {len(geo_df):,}")

    return geo_df, edges_df, fid_to_geo, geo_to_length

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
    fid_to_geo: dict[int, int],
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
        fid_to_geo=fid_to_geo,
        geo_to_length=geo_to_length,
        trip_time_lookup=trip_time_lookup,
        train_ratio=config.train_ratio,
        random_state=config.random_state,
        min_len=config.min_len,
        verbose=config.verbose,
        fmm_sep=config.fmm_sep,
        min_delta_seconds=config.min_delta_seconds,
        logger=logger,
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

    geo_df, edges_df, fid_to_geo, geo_to_length = build_and_save_geo(
        network_path=config.network_path,
        geo_path=paths["geo"],
        logger=logger,
    )

    rel_df = build_and_save_rel(
        edges_df=edges_df,
        rel_path=paths["rel"],
        logger=logger,
    )

    train_df, test_df = build_and_save_mm_csvs(
        config=config,
        fid_to_geo=fid_to_geo,
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




























# def main():
#     parser = argparse.ArgumentParser(description="Build TS-TrajGen inputs")

#     parser.add_argument(
#         "--network_path",
#         type=str,
#         required=True,
#         help="Path to road network file (e.g., .shp/.gpkg) with columns [fid, u, v, geometry]."
#     )
    
#     parser.add_argument(
#         "--fmm_match_path",
#         type=str,
#         required=True,
#         help="Path to FMM output CSV containing 'id' and 'cpath' columns."
#     )

#     parser.add_argument(
#         "--parquet_path",
#         type=str,
#         required=True,
#         help="Path to original cleaned parquet with datetime, user, and traj_id columns.",
#     )

#     parser.add_argument(
#         "--trip_id_map_csv",
#         type=str,
#         required=True,
#         help="Path to the FMM trip-id map CSV created when preparing GPS points for FMM.",
#     )

#     parser.add_argument(
#         "--out_dir",
#         type=str,
#         default="./outputs/nyc",
#         help="Output directory for generated files (.geo, .rel, *_mm_train/test.csv)."
#     )
    
#     parser.add_argument(
#         "--dataset_name",
#         type=str,
#         default="nyc",
#         help="Prefix for output files (e.g., 'nyc' -> nyc.geo, nyc.rel, etc.)."
#     )
    
#     parser.add_argument(
#         "--train_ratio",
#         type=float,
#         default=0.8,
#         help="Fraction of trajectories used for training (rest used for test split)."
#     )
    
#     parser.add_argument(
#         "--random_state",
#         type=int,
#         default=101,
#         help="Random seed for reproducible train/test splitting."
#     )

#     parser.add_argument(
#         "--min_len",
#         type=int,
#         default=2,
#         help="Minimum trajectory length (in edges); sequences with fewer edges are skipped."
#     )

#     parser.add_argument(
#         "--fmm_sep",
#         type=str,
#         default=";",
#         help="Delimiter used in the FMM output CSV."
#     )

#     parser.add_argument(
#         "--save_log_path",
#         type=str,
#         default="./outputs/logs",
#         help="Print detailed processing statistics."
#     )


#     parser.add_argument(
#         "--verbose",
#         action="store_true",
#         default=True,
#         help="Print detailed processing statistics."
#     )

#     args = parser.parse_args()



#     network_path = Path(args.network_path)
#     fmm_match_path = Path(args.fmm_match_path)
#     parquet_path = Path(args.parquet_path)
#     trip_id_map_csv = Path(args.trip_id_map_csv)
#     out_dir = Path(args.out_dir)
#     save_log_path = Path(args.save_log_path)
#     dataset_name = args.dataset_name

#     ensure_dir(out_dir)
#     ensure_dir(save_log_path)

#     logger = setup_logger("build_tstrajgen", log_dir=save_log_path)


#     print("Building .geo...")
#     geo_df, edges_df, fid_to_geo, geo_to_length = build_geo_and_length_lookups(network_path)
#     geo_df.to_csv(out_dir / f"{dataset_name}.geo", index=False)

#     print("Building .rel...")
#     rel_df = build_rel(edges_df)
#     rel_df.to_csv(out_dir / f"{dataset_name}.rel", index=False)

#     trip_time_lookup = build_trip_time_lookup(
#         parquet_path=parquet_path,
#         trip_id_map_csv=trip_id_map_csv,
#     )

#     print("Building train/test CSV...")
#     train_df, test_df = build_mm_csvs(
#         fmm_path=fmm_match_path,
#         fid_to_geo=fid_to_geo,
#         geo_to_length=geo_to_length,
#         trip_time_lookup=trip_time_lookup,
#         train_ratio=args.train_ratio,
#         random_state=args.random_state,
#         min_len=args.min_len,
#         verbose=args.verbose,
#         fmm_sep=args.fmm_sep,
#     )


#     # train_df, test_df = build_mm_csvs(
#     #     fmm_match_path,
#     #     fid_to_geo,
#     #     args.train_ratio,
#     #     args.random_state,
#     #     min_len=args.min_len,
#     #     verbose=True
#     # )

#     train_df.to_csv(out_dir / f"{dataset_name}_mm_train.csv", index=False)
#     test_df.to_csv(out_dir / f"{dataset_name}_mm_test.csv", index=False)

#     print("\nDone!")
#     print(f".geo rows: {len(geo_df):,}")
#     print(f".rel rows: {len(rel_df):,}")
#     print(f"train trajs: {len(train_df):,}")
#     print(f"test trajs: {len(test_df):,}")


# if __name__ == "__main__":
#     main()