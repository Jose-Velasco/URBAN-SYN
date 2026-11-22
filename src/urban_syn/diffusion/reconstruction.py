"""Reconstruct GPS observations from generated road IDs and position rates.

Road IDs refer to the shared road network's ``edge_id`` values. Rates are
relative movements in [0, 1], matching the training sequence encoder.
"""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
import geopandas as gpd
from shapely.geometry import LineString


_EARTH_RADIUS_METERS = 6_371_008.8


def load_road_geometries(
    road_network_path: str | Path,
    layer: str = "roads"
) -> dict[int, LineString]:
    """Load directed road geometries, keyed by shared road-network ``edge_id``.

    Args:
        road_network_path: Shared GeoPackage used to construct the PyG graph.
        layer: GeoPackage layer containing ``edge_id`` and ``geometry``.

    Returns:
        Road geometries in longitude/latitude coordinate order.
    """
    roads = gpd.read_file(road_network_path, layer=layer, columns=["edge_id", "geometry"])

    if roads.crs is None:
        raise ValueError("Road-network GeoPackage must declare a coordinate system.")
    if roads.crs.to_epsg() != 4326:
        roads = roads.to_crs(epsg=4326)
    if roads["edge_id"].isna().any() or roads["edge_id"].duplicated().any():
        raise ValueError("Road-network edge_id values must be present and unique.")

    result: dict[int, LineString] = {}
    for edge_id, geometry in roads[["edge_id", "geometry"]].itertuples(index=False, name=None):
        if not isinstance(geometry, LineString) or geometry.is_empty:
            raise ValueError(f"Road edge_id={edge_id} has no valid LineString geometry.")
        if len(geometry.coords) < 2:
            raise ValueError(f"Road edge_id={edge_id} has fewer than two coordinates.")
        result[int(edge_id)] = geometry

    return result


def decode_relative_rates(
    edge_ids: Sequence[int],
    rates: Sequence[float]
) -> list[float]:
    """Convert relative position rates to absolute fractions along each road.

    When consecutive observations are on the same road, the new rate describes
    movement through the *remaining* portion of that road, not a fresh offset.
    """
    if len(edge_ids) != len(rates):
        raise ValueError("Road IDs and rates must be aligned one-to-one.")

    positions: list[float] = []
    previous_road: int | None = None
    previous_position = 0.0
    for raw_edge_id, raw_rate in zip(edge_ids, rates):
        edge_id = int(raw_edge_id)
        rate = float(raw_rate)
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"Position rate {rate} is outside [0, 1].")

        if edge_id == previous_road:
            position = previous_position + (1.0 - previous_position) * rate
        else:
            position = rate

        positions.append(position)
        previous_road = edge_id
        previous_position = position
    
    return positions


def _haversine_meters(start: tuple[float, float], end: tuple[float, float]) -> float:
    """Return approximate geographic distance for a lon/lat segment."""
    lon1, lat1 = map(radians, start)
    lon2, lat2 = map(radians, end)
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * _EARTH_RADIUS_METERS * asin(min(1.0, sqrt(a)))


def _coordinate_at_fraction(road: LineString, position: float) -> tuple[float, float]:
    """Interpolate by geographic segment lengths, not degree-based Shapely length."""
    points = list(road.coords)
    if position <= 0:
        return float(points[0][0]), float(points[0][1])
    if position >= 1:
        return float(points[-1][0]), float(points[-1][1])

    lengths = [_haversine_meters(a[:2], b[:2]) for a, b in zip(points, points[1:])] # pyright: ignore[reportArgumentType]
    total = sum(lengths)
    if total <= 0:
        return float(points[0][0]), float(points[0][1])

    distance_remaining = position * total
    for start, end, segment_length in zip(points, points[1:], lengths):
        if distance_remaining <= segment_length and segment_length > 0:
            fraction = distance_remaining / segment_length
            lon = start[0] + fraction * (end[0] - start[0])
            lat = start[1] + fraction * (end[1] - start[1])
            return float(lon), float(lat)
        distance_remaining -= segment_length
    
    return float(points[-1][0]), float(points[-1][1])


def reconstruct_gps_trajectory(
    edge_ids: Sequence[int],
    rates: Sequence[float],
    road_geometries: Mapping[int, LineString]
) -> list[tuple[float, float, float]]:
    """Return ``(longitude, latitude, absolute_position_rate)`` per observation.

    Args:
        edge_ids: Shared road-network identifiers, including valid edge_id=0.
        rates: Model-generated relative position rates in [0, 1].
        road_geometries: Full road geometry lookup from the shared GeoPackage.
    """
    positions = decode_relative_rates(edge_ids, rates)
    
    output = []
    for edge_id, position in zip(edge_ids, positions):
        edge_id = int(edge_id)
        if edge_id not in road_geometries:
            raise ValueError(f"Generated road edge_id={edge_id} is absent from the shared network.")
        
        lon, lat = _coordinate_at_fraction(road_geometries[edge_id], position)
        output.append((lon, lat, position))
    
    return output
