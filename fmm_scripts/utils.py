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
from collections.abc import Sequence



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


def _validate_trajectory_identity(
    df: pd.DataFrame,
    identity_columns: Sequence[str],
) -> list[str]:
    """Validate the columns used to uniquely identify trajectories."""
    if isinstance(identity_columns, str):
        raise ValueError("identity_columns must be a sequence of column names.")

    columns = list(identity_columns)

    if not columns or len(set(columns)) != len(columns):
        raise ValueError("identity_columns must contain unique column names.")

    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing trajectory identity columns: {sorted(missing)}")

    if df.empty or df[columns].isna().any().any():
        raise ValueError("Trajectory identities must be nonempty and non-null.")

    return columns


def _to_unix_seconds(
    timestamps: pd.Series,
    timestamp_unit: str | None = None,
) -> pd.Series:
    """Convert timestamps to Unix seconds regardless of datetime resolution."""
    if (
        timestamp_unit is None
        and pd.api.types.is_numeric_dtype(timestamps)
    ):
        raise ValueError(
            "Numeric timestamps require an explicit timestamp_unit."
        )

    datetimes = pd.to_datetime(
        timestamps,
        unit=timestamp_unit,
        utc=True,
        errors="raise",
    ) # pyright: ignore[reportCallIssue]

    if datetimes.isna().any():
        raise ValueError("GPS timestamps cannot contain missing values.")

    # Explicitly convert to seconds before reading integer timestamps.
    # This avoids assuming that datetime64 always uses nanoseconds.
    return datetimes.dt.as_unit("s").astype("int64")


def add_fmm_trajectory_ids(
    df: pd.DataFrame,
    identity_columns: Sequence[str] = ("uid", "tid"),
) -> pd.DataFrame:
    """Assign deterministic FMM trajectory IDs and return their mapping.

    IDs start at 1 because FMM may output ID 0 for fully unmatched
    trajectories. Road edge_id=0 remains a valid, unrelated identifier.

    Parameters
    ----------
    df : DataFrame
        GPS observations. The ``fmm_id`` column is added in place.
    identity_columns : Sequence[str]
        Columns identifying each trajectory, defaulting to (uid, tid).

    Returns
    -------
    DataFrame
        One row per trajectory containing fmm_id and its original
        identity columns.
    """
    columns = _validate_trajectory_identity(df, identity_columns)

    # Sort unique identity tuples so shuffling GPS rows does not
    # change the IDs assigned to the same set of trajectories.
    trip_id_map = (
        df[columns]
        .drop_duplicates()
        .sort_values(columns, kind="stable")
        .reset_index(drop=True)
    )

    trip_id_map.insert(
        0,
        "fmm_id",
        np.arange(1, len(trip_id_map) + 1, dtype=np.int64),
    )

    # Match complete identity tuples, without constructing a
    # potentially ambiguous string such as "uid_tid".
    mapping_index = pd.MultiIndex.from_frame(trip_id_map[columns])
    trajectory_index = pd.MultiIndex.from_frame(df[columns])

    df["fmm_id"] = (
        mapping_index.get_indexer(trajectory_index) + 1
    ).astype("int64")

    return trip_id_map


def prepare_fmm_points(
    df: pd.DataFrame,
    timestamp_unit: str | None = None,
) -> pd.DataFrame:
    """Convert GPS observations into FMM's point CSV schema.

    Parameters
    ----------
    df : DataFrame
        Observations containing fmm_id, datetime, lng, and lat.
    timestamp_unit : str | None
        Unit for numeric timestamps, such as "s" or "ms".
        Not needed for datetime values or timestamp strings.

    Returns
    -------
    DataFrame
        GPS observations with fmm_id, x, y, and timestamp,
        sorted by trajectory ID and time.
    """
    start_time = time.perf_counter()

    logger.info("Converting timestamps and sorting GPS observations...")

    if "fmm_id" not in df.columns:
        raise ValueError(
            "Missing fmm_id. Call add_fmm_trajectory_ids() first."
        )

    fmm_ids = df["fmm_id"].astype("int64")
    if (fmm_ids < 1).any():
        raise ValueError("Input fmm_id values must start at 1.")

    gps_df = pd.DataFrame(
        {
            "fmm_id": fmm_ids,
            "x": df["lng"].astype(float),
            "y": df["lat"].astype(float),
            "timestamp": _to_unix_seconds(
                df["datetime"],
                timestamp_unit=timestamp_unit,
            ),
        }
    )

    # FMM requires points to be ordered by trajectory, then time.
    gps_df = (
        gps_df
        .sort_values(["fmm_id", "timestamp"], kind="stable")
        .reset_index(drop=True)
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