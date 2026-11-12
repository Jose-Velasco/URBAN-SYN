"""Metrics 11-12: off-network rate of the native GPS points, and snapped road connectivity (RSC)."""
from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd

from evaluation.geo import project_m
from evaluation.metrics.trajset import TrajSet

BUCKET_M = 100.0       # side of the square buckets the road segments are filed under
POINT_CHUNK = 50_000   # GPS points per step when measuring distances to roads


class RoadSegments:
    """Every straight piece of every road of a network, filed by bucket for fast nearest-road look-ups.

    Each segment is filed under every bucket its bounding box touches after growing the box by `reach_m`, so all
    segments within reach_m of a point are filed under the point's own bucket. Distances are in metres (project_m).
    """

    def __init__(self, x1, y1, x2, y2, reach_m: float):
        """File segments (given by their end points in metres) under the buckets they reach."""
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.reach_m = float(reach_m)
        bx0, bx1 = _bucket(np.minimum(x1, x2) - reach_m), _bucket(np.maximum(x1, x2) + reach_m)
        by0, by1 = _bucket(np.minimum(y1, y2) - reach_m), _bucket(np.maximum(y1, y2) + reach_m)
        nx, ny = bx1 - bx0 + 1, by1 - by0 + 1
        per_segment = nx * ny
        segment = np.repeat(np.arange(len(x1)), per_segment)
        local = np.arange(per_segment.sum()) - np.repeat(np.cumsum(per_segment) - per_segment, per_segment)
        keys = _key(np.repeat(bx0, per_segment) + local % np.repeat(nx, per_segment),
                    np.repeat(by0, per_segment) + local // np.repeat(nx, per_segment))
        order = np.argsort(keys, kind="stable")
        self.segment = segment[order]
        self.keys, self.start = np.unique(keys[order], return_index=True)
        self.end = np.append(self.start[1:], len(order))

    @classmethod
    def from_tstrajgen_geo(cls, geo_path, reach_m: float) -> "RoadSegments":
        """Read every road geometry of TS-TrajGen's nyc.geo (coordinates are [lng, lat])."""
        geo = pd.read_csv(geo_path, usecols=["coordinates"])
        lines = [np.asarray(json.loads(c), dtype=float) for c in geo["coordinates"]]
        vertices = np.concatenate(lines)
        x, y = project_m(vertices[:, 1], vertices[:, 0])
        sizes = np.array([len(line) for line in lines])
        last = np.cumsum(sizes) - 1                      # index of each road's last vertex
        first = np.ones(len(vertices) - 1, dtype=bool)   # segment from vertex k to k + 1 ...
        first[last[:-1]] = False                         # ... unless k ends a road
        k = np.flatnonzero(first)
        return cls(x[k], y[k], x[k + 1], y[k + 1], reach_m)

    def distance_m(self, lat, lng) -> np.ndarray:
        """Distance from each point to the nearest segment, or inf when no segment is within reach_m."""
        x, y = project_m(lat, lng)
        out = np.full(len(x), np.inf)
        for start in range(0, len(x), POINT_CHUNK):
            part = slice(start, start + POINT_CHUNK)
            out[part] = self._nearest(x[part], y[part])
        return out

    def _nearest(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Nearest-segment distance for one chunk of points (inf when none is filed under their bucket)."""
        keys = _key(_bucket(x), _bucket(y))
        pos = np.minimum(np.searchsorted(self.keys, keys), len(self.keys) - 1)
        found = self.keys[pos] == keys
        counts = np.where(found, self.end[pos] - self.start[pos], 0)
        out = np.full(len(x), np.inf)
        if counts.sum() == 0:
            return out
        point = np.repeat(np.arange(len(x)), counts)
        first = np.repeat(self.start[pos] - (np.cumsum(counts) - counts), counts)
        seg = self.segment[first + np.arange(counts.sum())]
        d = _point_segment_m(x[point], y[point], self.x1[seg], self.y1[seg], self.x2[seg], self.y2[seg])
        has = counts > 0
        out[has] = np.minimum.reduceat(d, (np.cumsum(counts) - counts)[has])
        return out


def off_network_scores(roads: RoadSegments, trajectories: TrajSet, settings: dict, native: str) -> tuple[list[dict], dict]:
    """Metric 11: share of GPS points farther than tau from every road of the canonical network.

    Meant for native generated GPS, before any map matching. For models that write road IDs (native == "road") the
    points are on roads by construction, so the value only checks that their network lines up with the canonical one.
    """
    off = settings["off_network"]
    distance = roads.distance_m(trajectories.lat, trajectories.lng)
    note = "native GPS" if native == "gps" else "points placed on roads by construction"
    rows = []
    for tau in sorted(set(off["sweep_m"]) | {off["tau_m"]}):
        rows.append({"metric": "off_network_rate", "setting": f"tau={tau}m", "main_setting": tau == off["tau_m"],
                     "value": float(np.mean(distance > tau)), "std": np.nan, "n": len(distance), "note": note})
    return rows, {}


def successor_keys(network: str, files: dict) -> np.ndarray:
    """Sorted keys (a * 2**32 + b) of every directed road pair where road b directly follows road a.

    tstrajgen:        the origin_id -> destination_id pairs of nyc.rel.
    diffrntraj_drive: from nyc_drive.gpkg, b follows a when b starts at the node where a ends (u(b) = v(a)).
    """
    if network == "tstrajgen":
        rel = pd.read_csv(files["rel"], usecols=["origin_id", "destination_id"])
        a, b = rel["origin_id"].to_numpy(np.int64), rel["destination_id"].to_numpy(np.int64)
    elif network == "diffrntraj_drive":
        with sqlite3.connect(files["gpkg"]) as con:
            edges = pd.read_sql_query("SELECT fid, u, v FROM roads", con)
        pairs = edges[["fid", "v"]].merge(edges[["fid", "u"]], left_on="v", right_on="u", suffixes=("_a", "_b"))
        a, b = pairs["fid_a"].to_numpy(np.int64), pairs["fid_b"].to_numpy(np.int64)
    else:
        raise ValueError(f"No connectivity rule for road network '{network}'")
    return np.unique(a * 2**32 + b)


def rsc_scores(road_table: pd.DataFrame | None, successors: np.ndarray, n_trajectories: int) -> tuple[list[dict], dict]:
    """Metric 12: transition-weighted share of consecutive roads where the next road directly follows.

    Consecutive repeats of the same road are collapsed first (staying on a road is not a transition). Directed, one step
    only, no 2-hop tolerance. Trajectories without a road sequence (GPS trajectories FMM could not match) are not
    counted; how many have one is in the note.
    """
    if road_table is None or road_table.empty:
        return [], {"rsc_coverage": f"0 of {n_trajectories:,} trajectories have roads"}
    table = road_table.sort_values(["traj_id", "seq"], kind="stable")
    traj, edge = table["traj_id"].to_numpy(), table["edge_id"].to_numpy(np.int64)
    new_traj = np.concatenate([[True], traj[1:] != traj[:-1]])
    keep = new_traj | np.concatenate([[True], edge[1:] != edge[:-1]])
    traj, edge, new_traj = traj[keep], edge[keep], new_traj[keep]
    same = ~new_traj[1:]                                   # transition k -> k + 1 inside one trajectory
    keys = (edge[:-1] * 2**32 + edge[1:])[same]
    pos = np.minimum(np.searchsorted(successors, keys), len(successors) - 1)
    connected = successors[pos] == keys
    with_roads = int(new_traj.sum())
    coverage = f"{with_roads:,} of {n_trajectories:,} trajectories have roads"
    rows = [{"metric": "rsc", "setting": "directed,1-step", "main_setting": True,
             "value": float(connected.mean()) if len(keys) else np.nan, "std": np.nan,
             "n": int(len(keys)), "note": coverage}]
    return rows, {"rsc_coverage": coverage, "rsc_transitions": int(len(keys))}


def _bucket(v) -> np.ndarray:
    """Bucket number along one axis."""
    return np.floor(np.asarray(v) / BUCKET_M).astype(np.int64)


def _key(bx, by) -> np.ndarray:
    """One number per bucket (bucket numbers are well within +-2**30 for any real coordinates)."""
    return (np.asarray(by, dtype=np.int64) << 31) + np.asarray(bx, dtype=np.int64)


def _point_segment_m(px, py, x1, y1, x2, y2) -> np.ndarray:
    """Distance from points to segments (all arrays of the same length), in metres."""
    dx, dy = x2 - x1, y2 - y1
    length2 = dx * dx + dy * dy
    t = np.where(length2 > 0, ((px - x1) * dx + (py - y1) * dy) / np.where(length2 > 0, length2, 1), 0.0)
    t = np.clip(t, 0.0, 1.0)
    return np.hypot(px - (x1 + t * dx), py - (y1 + t * dy))
