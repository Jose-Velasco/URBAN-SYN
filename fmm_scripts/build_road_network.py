"""Build the shared OSMnx road network used by FMM and downstream baselines.

This script is intentionally separate from GPS input preparation. Reusing an
existing shared network does not require running this script again.

For road_network.coverage.type=trajectory_bbox, the supplied trajectory
Parquet is used only to determine the geographic extent of a *new* network.

example usage:

uv run build_road_network.py \
    --config ./config/nyc.yaml \
    --shared_road_network_output ./data/road_network/nyc.gpkg \
    --shared_road_network_layer roads \
    --fmm_road_network_output ./data/fmm/fmm_nyc.shp \
    --trajectory_parquet ../data/nyc_output_tabular/output/traj_cleaned.parquet \
    --log_file ./outputs/logs/build_road_network.log
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import osmnx as ox

from utils import (
    build_custom_filter,
    build_padded_trajectory_bbox,
    configure_osmnx,
    format_duration,
    load_config,
    log_downloaded_graph,
    setup_logging,
    write_canonical_network as write_shared_road_network,
    write_fmm_network,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse settings for an explicit, one-time road-network build."""
    parser = argparse.ArgumentParser(
        description="Build the shared OSMnx road network and FMM Shapefile."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Shared road-network YAML, e.g. ./config/nyc.yaml",
    )
    parser.add_argument(
        "--shared_road_network_output",
        type=Path,
        required=True,
        help="Shared GeoPackage, e.g. ./data/road_network/nyc.gpkg",
    )
    parser.add_argument(
        "--fmm_road_network_output",
        type=Path,
        required=True,
        help="FMM Shapefile from the same network, e.g. ./data/fmm/fmm_nyc.shp",
    )
    parser.add_argument(
        "--shared_road_network_layer",
        default="roads",
        help="GeoPackage layer name (default: roads).",
    )
    parser.add_argument(
        "--trajectory_parquet",
        type=Path,
        default=None,
        help="Required only when road-network coverage uses trajectory_bbox.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow overwriting shared road-network outputs.",
    )
    parser.add_argument(
        "--log_file",
        type=Path,
        default=None,
        help="Optional log file.",
    )
    return parser.parse_args()


def _ensure_outputs_can_be_written(args: argparse.Namespace) -> None:
    """Protect existing shared road-network artifacts from accidental rebuilds."""
    existing = [
        path
        for path in (
            args.shared_road_network_output,
            args.fmm_road_network_output,
        )
        if path.exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Shared road-network artifacts already exist: "
            f"{', '.join(map(str, existing))}. "
            "Do not rebuild unless intentional; pass --overwrite explicitly."
        )


def prepare_road_edges(
    graph,
    useful_way_tags: list[str],
) -> gpd.GeoDataFrame:
    """Return road edges with unique edge_id, endpoints, and geometry.

    Keeps OSMnx's edge key to distinguish parallel (u, v) edges.
    """
    start_time = time.perf_counter()

    logger.info("Converting OSMnx graph to edge GeoDataFrame...")

    # Convert to GeoDataFrames
    _, edges_gdf = ox.graph_to_gdfs(graph)

    # Flatten (u, v, key) index into normal columns
    # OSMnx edge identity is normally stored in the (u, v, key) index.
    edges_gdf = edges_gdf.reset_index()

    # FMM requires a unique integer edge identifier.
    edges_gdf["edge_id"] = np.arange(
        len(edges_gdf),
        dtype="int64",
    )

    fmm_required_columns = [
        "edge_id",
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
    # Intentionally distinguish raw requested OSM tags from OSMnx-derived fields
    osmnx_columns = [
        col
        for col in ["osmid", "length", "speed_kph"]
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

def build_fmm_network(
    graph,
    shared_network_output: Path,
    shared_network_layer: str,
    fmm_output: Path,
    useful_way_tags: list[str],
) -> None:
    """
    Build shared GeoPackage and FMM Shapefile from an OSMnx graph.

    The graph acquisition strategy is handled separately so this function
    remains independent of whether the graph came from a place, bounding box,
    polygon, or another source.

    Args:
        shared_network_output: Path for the full-feature road GeoPackage.
        shared_network_layer: Layer name in that GeoPackage.
        fmm_output: Path for the minimal FMM road Shapefile.
        useful_way_tags: OSM attributes retained on the shared roads.
    """
    total_start = time.perf_counter()

    logger.info("Deriving OSMnx speed_kph edge feature...")

    graph = ox.routing.add_edge_speeds(
        graph,
        agg=np.mean,
    )

    edges_gdf = prepare_road_edges(
        graph=graph,
        useful_way_tags=useful_way_tags,
    )

    write_shared_road_network(
        edges_gdf=edges_gdf,
        out_gpkg=shared_network_output,
        layer=shared_network_layer,
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

def download_graph_from_place(
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
    """Return an OSMnx road graph downloaded for the given place."""
    start_time = time.perf_counter()

    logger.info(
        "Downloading '%s' network for place '%s'...",
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

    log_downloaded_graph(graph, start_time)

    return graph


def download_graph_from_bbox(
    bbox: tuple[float, float, float, float],
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
    custom_filter: str | list[str] | None,
):
    """Return an OSMnx road graph downloaded for the given bounding box."""
    start_time = time.perf_counter()

    logger.info(
        "Downloading '%s' network from bbox %s...",
        network_type,
        bbox,
    )

    graph = ox.graph_from_bbox(
        bbox,
        network_type=network_type,
        custom_filter=custom_filter,
        simplify=simplify,
        retain_all=retain_all,
        truncate_by_edge=truncate_by_edge,
    )

    log_downloaded_graph(graph, start_time)

    return graph


def main() -> None:
    """Download the configured graph and export both network representations."""
    args = parse_args()
    setup_logging(args.log_file)

    # Fail before any potentially expensive OSMnx download.
    _ensure_outputs_can_be_written(args)

    road_config = load_config(args.config)["road_network"]
    coverage = road_config["coverage"]
    graph_config = road_config["graph"]
    useful_way_tags = road_config["useful_way_tags"]

    configure_osmnx(
        max_query_area_size=graph_config["max_query_area_size"],
        useful_way_tags=useful_way_tags,
    )

    download_options = dict(
        network_type=graph_config["network_type"],
        simplify=graph_config["simplify"],
        retain_all=graph_config["retain_all"],
        truncate_by_edge=graph_config["truncate_by_edge"],
        custom_filter=build_custom_filter(graph_config["filter_network_types"]),
    )

    if coverage["type"] == "place":
        graph = download_graph_from_place(
            place=coverage["place"],
            which_result=coverage.get("which_result"),
            **download_options,
        )
    elif coverage["type"] == "trajectory_bbox":
        if args.trajectory_parquet is None:
            raise ValueError(
                "--trajectory_parquet is required for trajectory_bbox coverage."
            )
        bbox = build_padded_trajectory_bbox(
            parquet_path=args.trajectory_parquet,
            padding_meters=float(coverage["padding_meters"]),
        )
        graph = download_graph_from_bbox(bbox=bbox, **download_options)
    else:
        raise ValueError(f"Unsupported coverage type: {coverage['type']!r}")

    build_fmm_network(
        graph=graph,
        shared_network_output=args.shared_road_network_output,
        shared_network_layer=args.shared_road_network_layer,
        fmm_output=args.fmm_road_network_output,
        useful_way_tags=useful_way_tags,
    )


if __name__ == "__main__":
    main()
