import json
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import logging
from osmnx import _overpass
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