"""
Creates a directed edge layer with columns fid, u, v, and geometry, which matches the common FMM pattern

FMM accepts a CSV point file where each row is one observation with trajectory id,
longitude, latitude, and optional timestamp; the file MUST already be sorted by id AND timestamp

Expected the FMM network to use WGS84 (EPSG:4326)

OSMnx's length edge attribute is measured in meters

FMM says delta uses the spatial unit of the network file geometry

If Shapefile is EPSG:4326, the geometry is longitude/latitude in degrees, so delta is in degrees

where:
    id        = integer trajectory ID
    x         = longitude
    y         = latitude
    timestamp = Unix timestamp in seconds
"""
import osmnx as ox
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Literal
import argparse
import logging
import time
import geopandas as gpd

from utils import build_custom_filter, configure_osmnx, format_duration, load_config, normalize_multivalue_columns, setup_logging

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare shared road-network and trajectory inputs for FMM."
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Shared dataset YAML. Ex. ./config/nyc.yaml",
    )

    parser.add_argument(
        "--canonical_road_network_output",
        type=Path,
        required=True,
        help=(
            "Output path for the canonical shared road network GeoPackage. "
            "This file preserves the full road-network schema and OSM metadata. "
            "Example: ./data/road_network/nyc.gpkg"
        ),
    )

    parser.add_argument(
        "--fmm_road_network_output",
        type=Path,
        required=True,
        help=(
            "Output path for the minimal FMM-compatible road-network Shapefile. "
            "Contains only fid, u, v, and geometry. "
            "Example: ./data/fmm/fmm_nyc.shp"
        ),
    )

    parser.add_argument(
        "--canonical_road_network_layer",
        type=str,
        default="roads",
        help=(
            "Layer name inside the canonical GeoPackage. "
            "Defaults to 'roads'."
        ),
    )

    parser.add_argument(
        "--trajectory_parquet",
        type=Path,
        required=True,
        help="Input cleaned trajectory Parquet. Trajectory dataset used to construct FMM point observations.",
    )

    parser.add_argument(
        "--gps_output",
        type=Path,
        required=True,
        help="Output FMM GPS point CSV. Ex. nyc_gps_points_fmm_ready.csv",
    )

    parser.add_argument(
        "--trip_id_map_output",
        type=Path,
        default=None,
        help="Optional trajectory ID mapping CSV. Maps the temporary integer trajectory IDs used by FMM back to the original dataset trajectory identity. Ex. nyc_gps_points_fmm_trip_id_map",
    )

    parser.add_argument(
        "--log_file",
        type=Path,
        default=None,
        help="Optional file to save run logs.",
    )

    return parser.parse_args()

def download_osmnx_graph(
    place: str,
    network_type: Literal[
        "drive",
        "drive_service",
        "all_public",
        "walk",
        "bike",
        "all",
    ],
    simplify: bool,
    retain_all: bool,
    truncate_by_edge: bool,
    which_result: int | list[int | None] | None,
    custom_filter: str | list[str] | None,
):
    """Download the configured OSMnx road/path network."""
    start_time = time.perf_counter()

    logger.info(
        "Downloading '%s' network for %s...",
        network_type,
        place,
    )

    graph = ox.graph_from_place(
        place,
        network_type=network_type,
        custom_filter=custom_filter,
        simplify=simplify,
        retain_all=retain_all,
        truncate_by_edge=truncate_by_edge,
        which_result=which_result,
    )

    logger.info(
        "Downloaded graph CRS: %s",
        graph.graph.get("crs"),
    )

    logger.info(
        "Road network download completed in %s",
        format_duration(time.perf_counter() - start_time),
    )

    logger.info(
        "Downloaded graph: %d nodes, %d edges",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )

    return graph

def prepare_road_edges(
    graph,
    useful_way_tags: list[str],
) -> gpd.GeoDataFrame:
    """
    Convert an OSMnx graph into the canonical edge table used by FMM.

    FMM requires:
        fid, u, v, geometry

    OSMnx's edge key is additionally preserved because its MultiDiGraph
    may contain multiple edges between the same (u, v) node pair.
    """
    start_time = time.perf_counter()

    logger.info("Converting OSMnx graph to edge GeoDataFrame...")

    # Convert to GeoDataFrames
    _, edges_gdf = ox.graph_to_gdfs(graph)

    # Flatten (u, v, key) index into normal columns
    # OSMnx edge identity is normally stored in the (u, v, key) index.
    edges_gdf = edges_gdf.reset_index()

    # FMM requires a unique integer edge identifier.
    edges_gdf["fid"] = np.arange(
        len(edges_gdf),
        dtype="int64",
    )

    fmm_required_columns = [
        "fid",
        "u",
        "v",
        "geometry",
    ]

    # Not required by FMM, but needed to preserve full OSMnx edge identity.
    graph_identity_columns = [
        "key",
    ]

    reserved_columns = (
        set(fmm_required_columns)
        | set(graph_identity_columns)
    )

    # Protect against accidentally listing a FMM-required
    # column again in the YAML.
    duplicate_reserved = (
        reserved_columns
        & set(useful_way_tags)
    )

    if duplicate_reserved:
        raise ValueError(
            "useful_way_tags should contain only raw OSM tags, "
            "not graph/FMM-generated columns: "
            f"{sorted(duplicate_reserved)}"
        )

    # Keep a stable schema even when a requested tag is absent.
    for col in useful_way_tags:
        if col not in edges_gdf.columns:
            edges_gdf[col] = None

    # These are provided/derived by OSMnx rather than requested raw OSM tags.
    osmnx_columns = [
        col
        for col in ["osmid", "length"]
        if col in edges_gdf.columns
    ]

    columns = (
        fmm_required_columns
        + graph_identity_columns
        + osmnx_columns
        + useful_way_tags
    )

    edges_gdf = edges_gdf[columns].copy()

    logger.info(
        "FMM edge preparation completed in %s",
        format_duration(time.perf_counter() - start_time),
    )

    return edges_gdf

def write_canonical_network(
    edges_gdf: gpd.GeoDataFrame,
    out_gpkg: Path,
    layer: str = "roads",
) -> None:
    """Write the rich shared road network as a GeoPackage."""
    out_gpkg.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    start_time = time.perf_counter()

    logger.info(
        "Preparing canonical road-network attributes for GeoPackage output...",
    )

    canonical_gdf = normalize_multivalue_columns(
        edges_gdf
    )

    logger.info(
        "Writing canonical road network to %s "
        "(layer=%s)...",
        out_gpkg,
        layer,
    )

    canonical_gdf.to_file(
        out_gpkg,
        driver="GPKG",
        layer=layer,
        engine="pyogrio",
        use_arrow=True,
        index=False,

        # GeoPackage normally reserves the name "fid" for its internal
        # feature ID. Use a different internal name so our FMM `fid`
        # remains an ordinary, readable attribute column.
        layer_options={
            "FID": "gpkg_fid",
        },
    )

    logger.info(
        "Canonical road network write completed in %s",
        format_duration(time.perf_counter() - start_time),
    )


def write_fmm_network(
    edges_gdf: gpd.GeoDataFrame,
    out_shp: Path,
) -> None:
    """
    Write the minimal road-network schema required by FMM.

    FMM requires:
        fid, u, v, geometry
    """
    out_shp.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # FMM only needs edge ID, source node, target node, and geometry.
    fmm_edges_gdf = edges_gdf[
        [
            "fid",
            "u",
            "v",
            "geometry",
        ]
    ].copy()

    start_time = time.perf_counter()

    logger.info(
        "Writing minimal FMM road network to %s...",
        out_shp,
    )

    fmm_edges_gdf.to_file(
        out_shp,
        driver="ESRI Shapefile",
        engine="pyogrio",
        use_arrow=True,
        index=False,
    )

    logger.info(
        "FMM road network write completed in %s",
        format_duration(time.perf_counter() - start_time),
    )


def build_fmm_network_from_place(
    place: str,
    # out_shp: Path,
    canonical_output: Path,
    canonical_layer: str,
    fmm_output: Path,
    network_type: Literal[
        "drive",
        "drive_service",
        "all_public",
        "walk",
        "bike",
        "all",
    ],
    simplify: bool,
    retain_all: bool,
    truncate_by_edge: bool,
    which_result: int | list[int | None] | None,
    custom_filter: str | list[str] | None,
    max_query_area_size: float,
    useful_way_tags: list[str],
) -> None:
    """
    Build the shared road network and write both canonical and FMM artifacts.

    Outputs:
        canonical_output:
            Rich GeoPackage containing the complete shared network schema.

        fmm_output:
            Minimal Shapefile containing only the fields required by FMM.
    """
    total_start = time.perf_counter()

    configure_osmnx(
        max_query_area_size=max_query_area_size,
        useful_way_tags=useful_way_tags,
    )

    graph = download_osmnx_graph(
        place=place,
        network_type=network_type,
        simplify=simplify,
        retain_all=retain_all,
        truncate_by_edge=truncate_by_edge,
        which_result=which_result,
        custom_filter=custom_filter,
    )

    edges_gdf = prepare_road_edges(
        graph=graph,
        useful_way_tags=useful_way_tags,
    )

    write_canonical_network(
        edges_gdf=edges_gdf,
        out_gpkg=canonical_output,
        layer=canonical_layer,
    )

    write_fmm_network(
        edges_gdf=edges_gdf,
        out_shp=fmm_output,
    )

    logger.info(
        "Directed roads: %d",
        len(edges_gdf),
    )

    logger.info(
        "Road network build completed in %s",
        format_duration(time.perf_counter() - total_start),
    )

def add_fmm_trajectory_ids(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add integer FMM trajectory IDs and return their mapping.

    The current dataset's `tid` is only unique within each user, so
    `uid` and `tid` are combined into a unique trajectory key.
    """
    # Unique trip key
    # tid is only unique within a user. NOTE: "tid is only unique within a user" this may not be for another NYC dateset 
    df["trip_key"] = (
        df["uid"].astype(str)
        + "_"
        + df["tid"].astype(str)
    )

    # factorize assigns each unique trip key a stable integer ID
    # according to its first occurrence.
    df["id"], unique_trip_keys = pd.factorize(
        df["trip_key"],
        sort=False,
    )

    trip_id_map = pd.DataFrame(
        {
            "trip_key": unique_trip_keys,
            "id": range(len(unique_trip_keys)),
        }
    )

    return trip_id_map


def prepare_fmm_points(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert GPS observations into FMM's point schema."""
    start_time = time.perf_counter()

    logger.info(
        "Converting timestamps and sorting GPS observations..."
    )


    df["timestamp"] = (
        pd.to_datetime(df["datetime"])
        .astype("int64")
        // 10**9
    )

    # FMM expects observations for each trajectory to be sequential.
    # FMM expects points already ordered by trajectory then time.
    df.sort_values(
        ["id", "timestamp"],
        inplace=True,
    )

    gps_df = pd.DataFrame(
        {
            "id": df["id"].astype("int64"),
            "x": df["lng"].astype(float),
            "y": df["lat"].astype(float),
            "timestamp": df["timestamp"].astype("int64"),
        }
    )

    logger.info(
        "FMM point preparation completed in %s",
        format_duration(time.perf_counter() - start_time),
    )

    return gps_df


def build_fmm_points_csv(
    parquet_path: Path,
    out_csv: Path,
    trip_id_map_csv: Path | None = None,
) -> pd.DataFrame:
    """
    Build an FMM-compatible GPS observation CSV.

    CSV point file: a CSV file with a header row and columns separated by ;. 
    Each row stores a single observation containing id(integer), x(longitude), y(latitude), timestamp(optional, integer).

    Build FMM-compatible GPS observations:
        id, x, y, timestamp

    FMM expects observations to be ordered by trajectory ID and timestamp.

    The file must be sorted already by id and timestamp (trajectory will be passed sequentially). The id, x, y and timestamp column names will be specified by the user.

    Since our tid is only unique within a user, build a unique trip key from uid + tid. FMM's docs say id is an integer, so remap each unique trip key to an integer trajectory id.
    """
    total_start = time.perf_counter()

    # Read observations
    start_time = time.perf_counter()

    logger.info(
        "Reading GPS observations from %s...",
        parquet_path,
    )

    df = pd.read_parquet(parquet_path)

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

    trip_id_map = add_fmm_trajectory_ids(df)

    logger.info(
        "Trajectory-ID mapping completed in %s (%d trajectories)",
        format_duration(time.perf_counter() - start_time),
        len(trip_id_map),
    )

    # Prepare FMM point table
    gps_df = prepare_fmm_points(df)

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
    args = parse_args()

    setup_logging(args.log_file)

    dataset_config = load_config(args.config)

    road_config = dataset_config["road_network"]
    place = road_config["place"]

    graph_config = road_config["graph"]
    useful_way_tags = road_config["useful_way_tags"]

    # {"all", "all_public", "bike", "drive", "drive_service", "walk"} What type of street network to retrieve
    network_type = graph_config["network_type"]
    simplify = graph_config["simplify"]
    retain_all = graph_config["retain_all"]
    truncate_by_edge = graph_config["truncate_by_edge"]
    which_result = graph_config["which_result"]
    max_query_area_size = graph_config["max_query_area_size"]

    custom_filter = build_custom_filter(
        graph_config["filter_network_types"]
    )

    build_fmm_network_from_place(
        place=place,
        canonical_output=args.canonical_road_network_output,
        canonical_layer=args.canonical_road_network_layer,
        fmm_output=args.fmm_road_network_output,
        network_type=network_type,
        simplify=simplify,
        retain_all=retain_all,
        truncate_by_edge=truncate_by_edge,
        which_result=which_result,
        custom_filter=custom_filter,
        max_query_area_size=max_query_area_size,
        useful_way_tags=useful_way_tags
    )

    build_fmm_points_csv(
        parquet_path=args.trajectory_parquet,
        out_csv=args.gps_output,
        trip_id_map_csv=args.trip_id_map_output,
    )


if __name__ == "__main__":
    main()