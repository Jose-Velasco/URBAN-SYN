import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import logging
from osmnx import _overpass
from shapely.geometry import box
import yaml



logger = logging.getLogger(__name__)


_LIST_LIKE_TYPES = (
    list,
    tuple,
    set,
    np.ndarray,
)


def _serialize_multivalue_attribute(value) -> str | None:
    """
    Convert a potentially multi-valued OSM attribute to text.

    OSMnx can store some simplified edge attributes as either a scalar
    or a list. GeoPackage/Arrow columns require a consistent data type,
    so list-like values are serialized as JSON strings while scalar
    values are converted to strings.
    """
    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, (list, tuple, set)):
        return json.dumps(list(value))

    if pd.isna(value):
        return None

    return str(value)


def normalize_multivalue_columns(
    edges_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Normalize columns containing a mixture of scalar and list-like values.

    These mixed values commonly appear after OSMnx graph simplification.
    Columns that contain at least one list-like value are converted to
    text so they can be written consistently with Arrow-backed I/O.
    """
    edges_gdf = edges_gdf.copy()

    for column in edges_gdf.columns:
        if column == edges_gdf.geometry.name:
            continue

        if edges_gdf[column].dtype != "object":
            continue

        contains_list = edges_gdf[column].map(
            lambda value: isinstance(value, _LIST_LIKE_TYPES)
        ).any()

        if not contains_list:
            continue

        logger.info(
            "Serializing multi-valued road attribute '%s' for GeoPackage output...",
            column,
        )

        edges_gdf[column] = edges_gdf[column].map(
            _serialize_multivalue_attribute
        )

    return edges_gdf

def format_duration(seconds: float) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return (
        f"{int(hours):02d}:"
        f"{int(minutes):02d}:"
        f"{seconds:05.2f}"
    )

def load_config(config_path: str | Path) -> dict:
    """Load a YAML configuration."""
    config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def setup_logging(log_file: Path | None = None) -> None:
    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
    ]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.FileHandler(log_file, mode="a")
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )



def build_custom_filter(
    network_types: list[str],
) -> list[str]:
    """Build a union of OSMnx's predefined network filters."""
    # NOTE:
    # OSMnx's PRIVATE preset-filter helper is used here so the unified
    # network follows OSMnx's built-in drive/bike/walk filtering semantics.
    # This project currently uses OSMnx 2.0.7. Recheck this helper when
    # upgrading OSMnx.
    return [
        _overpass._get_network_filter(network_type)
        for network_type in network_types
    ]

def configure_osmnx(
    max_query_area_size: float,
    useful_way_tags: list[str],
) -> None:
    """Configure OSMnx settings used to download the shared road network."""

    # Control the maximum polygon area OSMnx sends in a single
    # Overpass query. Larger areas are subdivided automatically.
    ox.settings.max_query_area_size = max_query_area_size

    # OSMnx only retains configured raw OSM way tags during download.
    # Preserve existing defaults while adding dataset-specific attributes.
    ox.settings.useful_tags_way = list(
        dict.fromkeys(
            ox.settings.useful_tags_way
            + useful_way_tags
        )
    )

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

        # # GeoPackage normally reserves the name "fid" for its internal
        # # feature ID. Use a different internal name so our FMM `fid`
        # # remains an ordinary, readable attribute column.
        # layer_options={
        #     "FID": "gpkg_fid",
        # },
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
        edge_id, u, v, geometry
    """
    out_shp.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # FMM only needs edge ID, source node, target node, and geometry.
    fmm_edges_gdf = edges_gdf[
        [
            "edge_id",
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

def log_downloaded_graph(graph, start_time: float) -> None:
    """Log basic information about a downloaded OSMnx graph."""
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

def build_padded_trajectory_bbox(
    parquet_path: Path,
    padding_meters: float,
) -> tuple[float, float, float, float]:
    """
    Return a metric-padded bounding box around the processed GPS data.

    Returns:
        Tuple in OSMnx order:
        (left, bottom, right, top)
    """
    if padding_meters < 0:
        raise ValueError(
            f"padding_meters must be >= 0, got {padding_meters}"
        )

    coords = pd.read_parquet(
        parquet_path,
        columns=["lng", "lat"],
    ).dropna(
        subset=["lng", "lat"],
    )

    if coords.empty:
        raise ValueError(
            "No valid GPS coordinates found in trajectory parquet."
        )

    min_lon = float(coords["lng"].min())
    min_lat = float(coords["lat"].min())
    max_lon = float(coords["lng"].max())
    max_lat = float(coords["lat"].max())

    logger.info(
        "Processed trajectory bounds: "
        "left=%.7f, bottom=%.7f, right=%.7f, top=%.7f",
        min_lon,
        min_lat,
        max_lon,
        max_lat,
    )

    raw_bbox = gpd.GeoDataFrame(
        geometry=[
            box(
                min_lon,
                min_lat,
                max_lon,
                max_lat,
            )
        ],
        crs="EPSG:4326",
    )

    projected_crs = raw_bbox.estimate_utm_crs()

    if projected_crs is None:
        raise ValueError(
            "Could not determine projected CRS for trajectory bounds."
        )

    logger.info(
        "Using projected CRS %s for %.0f m road-network padding",
        projected_crs,
        padding_meters,
    )

    buffered = (
        raw_bbox
        .to_crs(projected_crs)
        .buffer(padding_meters)
    )

    buffered_wgs84 = gpd.GeoSeries(
        buffered,
        crs=projected_crs,
    ).to_crs("EPSG:4326")

    left, bottom, right, top = buffered_wgs84.total_bounds

    logger.info(
        "Padded road-network bounds: "
        "left=%.7f, bottom=%.7f, right=%.7f, top=%.7f",
        left,
        bottom,
        right,
        top,
    )

    return (
        float(left),
        float(bottom),
        float(right),
        float(top),
    )