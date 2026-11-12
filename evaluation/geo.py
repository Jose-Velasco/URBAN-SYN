"""Small geometry helpers shared by the converters."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6_371_000.0  # the spec's Earth radius (6371.0 km)


def haversine_m(lat1, lng1, lat2, lng2):
    """Great-circle distance in metres; works on single numbers or numpy arrays."""
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def prepare_polyline(latlng) -> tuple[np.ndarray, np.ndarray]:
    """Turn a road's (lat, lng) vertices into (vertices, cumulative length in metres).

    Doing this once per road lets point_on_polyline find many points quickly.
    """
    coords = np.asarray(latlng, dtype=float).reshape(-1, 2)
    steps = haversine_m(coords[:-1, 0], coords[:-1, 1], coords[1:, 0], coords[1:, 1])
    return coords, np.concatenate([[0.0], np.cumsum(steps)])


def point_on_polyline(prepared: tuple[np.ndarray, np.ndarray], fraction: float) -> tuple[float, float]:
    """Return the (lat, lng) at `fraction` (0 = start, 1 = end) of the way along a prepared road."""
    coords, cumulative = prepared
    if len(coords) == 1 or cumulative[-1] == 0:
        return float(coords[0, 0]), float(coords[0, 1])
    target = min(max(float(fraction), 0.0), 1.0) * cumulative[-1]
    i = int(np.searchsorted(cumulative, target, side="right")) - 1
    i = min(max(i, 0), len(coords) - 2)
    step = cumulative[i + 1] - cumulative[i]
    t = 0.0 if step == 0 else (target - cumulative[i]) / step
    lat = coords[i, 0] + t * (coords[i + 1, 0] - coords[i, 0])
    lng = coords[i, 1] + t * (coords[i + 1, 1] - coords[i, 1])
    return float(lat), float(lng)


def tstrajgen_road_midpoints(geo_path, only=None) -> dict[int, tuple[float, float]]:
    """Read TS-TrajGen's nyc.geo and return {geo_id: (lat, lng)} of each road's midpoint.

    The midpoint is measured along the road's own geometry, so it lies on the road even when the road curves. `only`
    limits the work to the road IDs a model uses. nyc.geo stores coordinates in GeoJSON order [lng, lat].
    """
    geo = pd.read_csv(geo_path, usecols=["geo_id", "coordinates"])
    if only is not None:
        geo = geo[geo["geo_id"].isin(set(only))]
    midpoints = {}
    for geo_id, coordinates in zip(geo["geo_id"], geo["coordinates"]):
        latlng = [(lat, lng) for lng, lat in json.loads(coordinates)]
        midpoints[int(geo_id)] = point_on_polyline(prepare_polyline(latlng), 0.5)
    return midpoints


# Latitude of the middle of the spec's grid, used for the metric projection below.
PROJECTION_LAT = 40.72


def project_m(lat, lng) -> tuple[np.ndarray, np.ndarray]:
    """Project WGS84 points to x, y in metres (equirectangular, true scale at PROJECTION_LAT).

    Over New York City the scale error is under 0.4%, so a 30 m threshold is off by at most about 0.1 m. Used for bulk
    distances to roads or between trajectory features; distances between GPS points use haversine_m.
    """
    lat = np.radians(np.asarray(lat, dtype=float))
    lng = np.radians(np.asarray(lng, dtype=float))
    return EARTH_RADIUS_M * np.cos(np.radians(PROJECTION_LAT)) * lng, EARTH_RADIUS_M * lat
