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

# DIFF-RNTRAJ COPY OF prepare_fmm_inputs.py. prepare_fmm_inputs.py itself is unchanged.
# - Road network code (download_osmnx_graph ... build_fmm_network_from_place):
#   same as the original. It reads the shared config/nyc.yaml (one folder up),
#   which asks OSMnx for drivable roads only; main() stops if that changes.
# - GPS point code: the original functions are commented out further down,
#   and the new version is at the bottom of this file.
# - parse_args() and main() are the same as in the original.
# Lives in fmm_scripts/diffrntraj_version/; utils.py is in fmm_scripts, one folder up.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import osmnx as ox
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Literal
import argparse
import logging
import time
import geopandas as gpd
from shapely.geometry import box

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
    place: str | None,
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
    bbox: tuple[float, float, float, float] | None = None,
):
    """Download the configured OSMnx road/path network (named place or bounding box)."""
    start_time = time.perf_counter()

    if bbox is not None:
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
    else:
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

    # CHANGED: reserve 0 for padding/special use in the model and
    # give every FMM edge a stable 1-based Diff-RNTraj road ID. Keeping both
    # IDs in the canonical network makes the later converter a direct lookup:
    #     FMM fid 0 -> Diff-RNTraj rid 1, etc.
    # `rid` is intentionally NOT written to the minimal FMM shapefile.
    edges_gdf["rid"] = edges_gdf["fid"] + 1

    # CHANGED: normalize OSM's highway class into a compact numeric
    # road level now, while the raw OSM metadata is still available. This saves
    # the converter from having to recreate road classes later.
    def highway_to_level(value):
        # normalize_multivalue_columns() runs later, so highway can still be a
        # list/tuple here for ways carrying multiple OSM values.
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None

        level_by_highway = {
            "motorway": 1, "motorway_link": 1,
            "trunk": 2, "trunk_link": 2,
            "primary": 3, "primary_link": 3,
            "secondary": 4, "secondary_link": 4,
            "tertiary": 5, "tertiary_link": 5,
            "residential": 6, "unclassified": 6, "living_street": 6,
            "service": 7,
        }
        return level_by_highway.get(value, 7)

    if "highway" in edges_gdf.columns:
        edges_gdf["level"] = edges_gdf["highway"].apply(highway_to_level)
    else:
        edges_gdf["level"] = 7

    fmm_required_columns = [
        "fid",
        "u",
        "v",
        "geometry",
    ]

    # Not required by FMM, but needed to preserve full OSMnx edge identity.
    graph_identity_columns = [
        "key",
        # CHANGED: converter-ready metadata retained only in the
        # rich canonical network (the FMM shapefile remains minimal).
        "rid",
        "level",
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


def build_padded_trajectory_bbox(
    parquet_path: Path,
    padding_meters: float,
) -> tuple[float, float, float, float]:
    """Bounding box of all GPS points in the parquet, padded by padding_meters.

    Returns (left, bottom, right, top), the order OSMnx expects.
    """
    if padding_meters < 0:
        raise ValueError(f"padding_meters must be >= 0, got {padding_meters}")

    coords = pd.read_parquet(parquet_path, columns=["lng", "lat"]).dropna(subset=["lng", "lat"])
    if coords.empty:
        raise ValueError("No valid GPS coordinates found in trajectory parquet.")

    min_lon, min_lat = float(coords["lng"].min()), float(coords["lat"].min())
    max_lon, max_lat = float(coords["lng"].max()), float(coords["lat"].max())
    logger.info(
        "Processed trajectory bounds: left=%.7f, bottom=%.7f, right=%.7f, top=%.7f",
        min_lon, min_lat, max_lon, max_lat,
    )

    # Pad in meters: buffer the box in a local UTM projection, then go back to lat/lng.
    raw_bbox = gpd.GeoDataFrame(geometry=[box(min_lon, min_lat, max_lon, max_lat)], crs="EPSG:4326")
    projected_crs = raw_bbox.estimate_utm_crs()
    if projected_crs is None:
        raise ValueError("Could not determine projected CRS for trajectory bounds.")
    logger.info("Using projected CRS %s for %.0f m road-network padding", projected_crs, padding_meters)

    buffered = raw_bbox.to_crs(projected_crs).buffer(padding_meters)
    left, bottom, right, top = gpd.GeoSeries(buffered, crs=projected_crs).to_crs("EPSG:4326").total_bounds
    logger.info(
        "Padded road-network bounds: left=%.7f, bottom=%.7f, right=%.7f, top=%.7f",
        left, bottom, right, top,
    )
    return float(left), float(bottom), float(right), float(top)


def build_fmm_network_from_place(
    place: str | None,
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
    bbox: tuple[float, float, float, float] | None = None,
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
        bbox=bbox,
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



def main() -> None:
    args = parse_args()

    setup_logging(args.log_file)

    dataset_config = load_config(args.config)

    road_config = dataset_config["road_network"]
    # place = road_config["place"]
    coverage = road_config["coverage"]
    place, bbox = None, None
    if coverage["type"] == "place":
        place = coverage["place"]
    elif coverage["type"] == "trajectory_bbox":
        bbox = build_padded_trajectory_bbox(
            parquet_path=args.trajectory_parquet,
            padding_meters=float(coverage["padding_meters"]),
        )
    else:
        raise ValueError(f"Unsupported road network coverage type: {coverage['type']!r}")

    graph_config = road_config["graph"]
    useful_way_tags = road_config["useful_way_tags"]

    # {"all", "all_public", "bike", "drive", "drive_service", "walk"} What type of street network to retrieve
    network_type = graph_config["network_type"]
    simplify = graph_config["simplify"]
    retain_all = graph_config["retain_all"]
    truncate_by_edge = graph_config["truncate_by_edge"]
    # which_result = graph_config["which_result"]
    # CHANGED (same as prepare_fmm_inputs.py): which_result is read from the
    # place coverage block, and is None when it is not set (trajectory_bbox).
    which_result = coverage.get("which_result")
    max_query_area_size = graph_config["max_query_area_size"]

    # NEW: Diff-RNTraj uses the shared config/nyc.yaml, so check that it still
    # asks for drivable roads only (a drive + bike + walk network would give
    # Diff-RNTraj roads that cars cannot use).
    if list(graph_config["filter_network_types"]) != ["drive"]:
        raise ValueError(
            f"{args.config}: road_network.graph.filter_network_types is "
            f"{graph_config['filter_network_types']}, but Diff-RNTraj needs [drive]. "
            "Pull the latest main (drive-only config/nyc.yaml)."
        )

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
        useful_way_tags=useful_way_tags,
        bbox=bbox,
    )

    build_fmm_points_csv(
        parquet_path=args.trajectory_parquet,
        out_csv=args.gps_output,
        trip_id_map_csv=args.trip_id_map_output,
    )


# NEW CODE FOR DIFF-RNTRAJ (everything below this line is new)
# What changed compared with the original code
#
# - add_fmm_trajectory_ids(), prepare_fmm_points() and build_fmm_points_csv()
#   are commented out. The new build_fmm_points_csv() below replaces all
#   three. It keeps the same name and arguments, so main() did not change.
#
# - Input file: pass enriched_moves.parquet with --trajectory_parquet
#   (instead of traj_cleaned.parquet). It has one row per GPS point of each
#   "move" (a trip between two stops) and a transport-mode label.
#
# - All transport modes are kept (team decision), even though the road network
#   is drive-only (see config/nyc.yaml). MODE_LABELS below sets which
#   modes are used; it was bus, car and taxi only before.
#
# - Diff-RNTraj describes a trip as one road position every 15 seconds, with
#   20 to 100 points per trip. So each move is resampled to one point every
#   15 s and cut into pieces of at most 100 points. Pieces with fewer than
#   20 points are dropped. Each piece is one FMM trajectory.
#
# - Trajectory ids start at 1 instead of 0, because FMM writes id 0 when a
#   trajectory fails (a point with no road nearby). Starting at 1 means an
#   id of 0 in FMM's output always means "failed".
#
# - The output CSV has the same format as before: id;x;y;timestamp.
#   The id map has the same columns as before (trip_key, id), but trip_key
#   is now "uid_tid_moveid_piece".

# MAT-Builder transport-mode labels in enriched_moves.parquet:
# 0 walk, 1 bike, 2 bus, 3 car, 4 subway, 5 train, 6 taxi
# VEHICLE_LABELS = [2, 3, 6]  # bus, car, taxi
# CHANGED (team decision): use all transport modes, like the other baselines.
# Put [2, 3, 6] back to use only bus, car and taxi moves.
MODE_LABELS = [0, 1, 2, 3, 4, 5, 6]  # walk, bike, bus, car, subway, train, taxi
# Mode names written to the trajectory ID map (label column) instead of the numbers.
MODE_NAMES = {0: "walk", 1: "bike", 2: "bus", 3: "car", 4: "subway", 5: "train", 6: "taxi"}

# Diff-RNTraj trajectory format (same as its Porto dataset)
STEP_S = 15    # seconds between two consecutive points
MIN_LEN = 20   # fewest points in one trajectory (20 points = 4 min 45 s)
MAX_LEN = 100  # most points in one trajectory (100 points = 24 min 45 s)


def build_fmm_points_csv(
    parquet_path: Path,
    out_csv: Path,
    trip_id_map_csv: Path | None = None,
) -> pd.DataFrame:
    """
    NEW (Diff-RNTraj): build the FMM GPS point CSV from the moves in MODE_LABELS.

    Input: enriched_moves.parquet (MAT-Builder moves with a transport-mode label).

    Output CSV (same format as the original): header row, columns separated by ;
        id        = integer trajectory id (one piece of one move), starting at 1
        x         = longitude
        y         = latitude
        timestamp = Unix timestamp in seconds
    Rows of the same id are together and in time order, as FMM requires.

    Returns a converter-ready trajectory ID map containing id plus the original
    move identity, mode label, timestamps, and expected number of points.
    """
    total_start = time.perf_counter()

    # Read observations
    start_time = time.perf_counter()

    logger.info(
        "Reading moves from %s...",
        parquet_path,
    )

    # CHANGED: enriched_moves.parquet stores tid and move_id as the index
    # (not as columns). reset_index() turns them back into normal columns.
    df = pd.read_parquet(parquet_path).reset_index()

    logger.info(
        "Parquet read completed in %s (%d rows)",
        format_duration(time.perf_counter() - start_time),
        len(df),
    )

    # CHANGED: keep the moves whose mode is in MODE_LABELS (all modes by default).
    df = df[df["label"].isin(MODE_LABELS)].copy()

    logger.info(
        "Rows from moves with modes %s: %d",
        [MODE_NAMES[m] for m in MODE_LABELS],
        len(df),
    )

    # CHANGED: Unix time in seconds. Subtracting 1970-01-01 gives the right
    # answer whether pandas stores the datetimes in nanoseconds or in
    # microseconds. The 15 s resampling below depends on these being seconds.
    df["t"] = (
        pd.to_datetime(df["datetime"]) - pd.Timestamp("1970-01-01")
    ).dt.total_seconds()

    # NEW: one move = one (uid, tid, move_id). tid is only unique within a
    # user, like in the original code. np.interp below needs each move's
    # times in increasing order.
    move_keys = ["uid", "tid", "move_id"]
    df = df.sort_values(move_keys + ["t"])

    rows = []  # one (id, x, y, timestamp) per resampled point
    # CHANGED: keep converter/validation metadata for each piece,
    # not just a concatenated trip_key. This lets the converter verify FMM
    # output lengths and trace any failed match back to its source move.
    trip_rows = []
    fmm_id = 0

    for (uid, tid, move_id), move in df.groupby(move_keys, sort=False):
        t = move["t"].to_numpy()

        # NEW: one point every STEP_S seconds, from the first to the last
        # GPS time of the move. Ex. a 7-minute move (420 s) gives
        # 420 // 15 + 1 = 29 points.
        n_points = int((t[-1] - t[0]) // STEP_S) + 1
        new_t = t[0] + STEP_S * np.arange(n_points)

        # NEW: the position at each new time lies on the straight line
        # between the two GPS points around it (linear interpolation).
        lat = np.interp(new_t, t, move["lat"].to_numpy())
        lng = np.interp(new_t, t, move["lng"].to_numpy())

        # NEW: cut the move into pieces of at most MAX_LEN points
        # (points 0-99, 100-199, ...). Skip pieces shorter than MIN_LEN.
        for piece, start in enumerate(range(0, n_points, MAX_LEN)):
            end = min(start + MAX_LEN, n_points)

            if end - start < MIN_LEN:
                continue

            # CHANGED: ids start at 1 (the original started at 0).
            fmm_id += 1

            for k in range(start, end):
                rows.append((fmm_id, lng[k], lat[k], int(new_t[k])))

            # CHANGED: richer trajectory map. The old trip_key can
            # always be reconstructed from uid/tid/move_id/piece, while these
            # separate columns are much easier and safer for a converter.
            trip_rows.append(
                {
                    "id": fmm_id,
                    "uid": uid,
                    "tid": tid,
                    "move_id": move_id,
                    "piece": piece,
                    "label": MODE_NAMES[int(move["label"].iloc[0])],
                    "start_timestamp": int(new_t[start]),
                    "end_timestamp": int(new_t[end - 1]),
                    "num_points": end - start,
                }
            )

    # Same columns as the original: id, x (longitude), y (latitude), timestamp.
    # Rows are already grouped by id and in time order.
    gps_df = pd.DataFrame(rows, columns=["id", "x", "y", "timestamp"])
    # CHANGED: preserve every metadata field above.
    trip_id_map = pd.DataFrame(trip_rows)

    # Write outputs (same as the original)
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

        logger.info(
            "Writing trajectory ID map to %s...",
            trip_id_map_csv,
        )

        trip_id_map.to_csv(
            trip_id_map_csv,
            index=False,
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
        "Trajectories (pieces of 20-100 points): %d",
        len(trip_id_map),
    )

    logger.info(
        "FMM point CSV build completed in %s",
        format_duration(time.perf_counter() - total_start),
    )

    return trip_id_map


# CHANGED: this block moved from above to the very end of the file, so that
# the new build_fmm_points_csv() is defined before main() runs.
if __name__ == "__main__":
    main()
