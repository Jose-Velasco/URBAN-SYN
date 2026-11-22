"""Small geometry helpers shared by the converters."""
from __future__ import annotations

import json
import sqlite3
import struct
from pathlib import Path

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


def read_gpkg_roads(path, columns=("edge_id", "u", "v"), layer: str = "roads",
                    geometry: bool = True) -> tuple[pd.DataFrame, list[np.ndarray] | None]:
    """Read a road layer of a GeoPackage, such as the shared fmm_scripts/data/road_network/nyc.gpkg.

    Uses Python's built-in sqlite3 (a GeoPackage is an SQLite file), so no GDAL or geopandas is needed. Returns the
    requested columns sorted by the first one, and, with geometry=True, each road's [lng, lat] vertices in the same
    order. The geometries must be WGS84 (EPSG:4326) LineStrings, as the shared road network's are.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Road network not found: {path}")
    with sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True) as con:
        row = con.execute("SELECT column_name, srs_id FROM gpkg_geometry_columns WHERE table_name = ?",
                          (layer,)).fetchone()
        if row is None:
            raise ValueError(f"{path}: no road layer named '{layer}'")
        geometry_column, srs_id = row
        if geometry and srs_id != 4326:
            raise ValueError(f"{path}: road geometries are in EPSG:{srs_id}; expected EPSG:4326 (longitude/latitude)")
        selected = list(columns) + ([geometry_column] if geometry else [])
        quoted = ", ".join(f'"{c}"' for c in selected)
        rows = con.execute(f'SELECT {quoted} FROM "{layer}" ORDER BY "{columns[0]}"').fetchall()
    table = pd.DataFrame([r[:len(columns)] for r in rows], columns=list(columns))
    if table[columns[0]].duplicated().any():
        raise ValueError(f"{path}: road IDs in '{columns[0]}' are not unique")
    lines = [_gpkg_linestring(r[-1], path) for r in rows] if geometry else None
    return table, lines


def gpkg_road_count(path, layer: str = "roads") -> int:
    """Number of roads in a GeoPackage road layer."""
    path = Path(path).resolve()
    with sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True) as con:
        return int(con.execute(f'SELECT COUNT(*) FROM "{layer}"').fetchone()[0])


def gpkg_road_polylines(path, only=None, id_column: str = "edge_id") -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Read a GeoPackage road layer and return {road ID: prepared polyline} (see prepare_polyline) of each road.

    `only` limits the result to the road IDs a model uses.
    """
    table, lines = read_gpkg_roads(path, columns=(id_column,))
    keep = None if only is None else set(only)
    return {int(road): prepare_polyline(line[:, ::-1]) for road, line in zip(table[id_column], lines)
            if keep is None or int(road) in keep}


def _gpkg_linestring(blob, path) -> np.ndarray:
    """Decode one GeoPackage geometry (a short header, then standard WKB) into an (n, 2) array of [lng, lat]."""
    if blob is None or bytes(blob[:2]) != b"GP":
        raise ValueError(f"{path}: a road has no GeoPackage geometry")
    flags = blob[3]
    if flags & 0b10000:
        raise ValueError(f"{path}: a road has an empty geometry")
    envelope = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(flags >> 1) & 0b111]
    wkb = memoryview(blob)[8 + envelope:]
    order = "<" if wkb[0] == 1 else ">"
    (kind,) = struct.unpack_from(order + "I", wkb, 1)
    dims = 2 + bool(kind & 0x80000000) + bool(kind & 0x40000000)          # EWKB Z / M flags
    kind &= 0x0FFFFFFF
    dims += {0: 0, 1: 1, 2: 1, 3: 2}[kind // 1000]                         # ISO WKB: 1002 Z, 2002 M, 3002 ZM
    if kind % 1000 != 2:
        raise ValueError(f"{path}: road geometries must be LineStrings (found WKB type {kind})")
    (n,) = struct.unpack_from(order + "I", wkb, 5)
    if n < 2:
        raise ValueError(f"{path}: a road has fewer than two vertices")
    return np.frombuffer(wkb, dtype=order + "f8", count=n * dims, offset=9).reshape(n, dims)[:, :2].copy()


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
