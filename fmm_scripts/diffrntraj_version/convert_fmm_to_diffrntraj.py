#!/usr/bin/env python3
"""
Convert the NYC FMM outputs into the SAME dataset folder layout used by the
processed Diff-RNTraj Porto data.

Target structure
----------------

NYC/
├── extra_file/
│   ├── new2raw_rid.json
│   ├── raw_rn_dict.json
│   ├── raw2new_rid.json
│   └── rn_dict.json
├── gen_all/
│   ├── eid_seqs.bin
│   ├── length_distri.npy
│   └── rate_seqs.bin
├── gen_train/          (same files; trips not in TS-TrajGen's test set, used for training)
├── gen_test/           (same files; TS-TrajGen's test trips, held out for evaluation)
├── graph/              (filled later by build_graph_nyc.py and the embed step;
│                        graph_A.csv is not created, training uses graph_edge.edgelist)
└── road_network/
    ├── edges.cpg
    ├── edges.dbf
    ├── edges.prj
    ├── edges.shp
    ├── edges.shx
    ├── nodes.cpg
    ├── nodes.dbf
    ├── nodes.prj
    ├── nodes.shp
    └── nodes.shx

Inputs expected from the FMM pipeline
-------------------------------------
./data/nyc_drive_fmm_match.csv
./data/road_network/nyc_drive.gpkg

The trajectory-ID map is not needed by Diff-RNTraj itself, so it is not copied
into the final NYC folder.

Important
---------
1. FMM uses raw 0-based `fid`.
2. Model road IDs 1..N are given ONLY to roads that appear in at least one
   matched trajectory (sorted by raw fid). ID 0 is padding. This matches the
   processed Porto data (14,455 roads in raw_rn_dict.json, 13,695 model IDs).
   raw_rn_dict.json keeps the full network keyed by fid; rn_dict.json,
   new2raw_rid.json and raw2new_rid.json contain only the N observed roads.
   (Using new_rid = fid + 1 for every NYC road made N = 139,611 and graph_A
   ~487 GB, while only ~21.6k roads are ever used.)
3. The moving rate follows Diff-RNTraj's encoding (see models/model_utils.py
   toseq() and the Porto rate_seqs.bin):
       p    = clamp(offset / length, 0, 1)        position on the road
       first point on a road:      rate = p
       next points on SAME road:   rate = (p - p_prev) / (1 - p_prev)
   i.e. the share of the REMAINING road covered since the previous point,
   clamped to [0, 1]. Diff-RNTraj decodes it as
       p = p_prev + (1 - p_prev) * rate.
4. Repeated FMM road IDs are preserved point-by-point.
5. `eid_seqs.bin` and `rate_seqs.bin` are dictionaries grouped by exact
   trajectory length, because Diff-RNTraj batches trajectories by length.
6. `road_embed.txt` must contain a real pre-trained road embedding. This script
   NEVER invents one. If you already have one, pass --road-embed-source.
7. Diff-RNTraj's original graph_A.csv is a dense N x N UTGraph adjacency
   matrix. It is not created (that code is commented out, and
   --dense-graph-max-nodes is ignored): training reads the sparse
   graph_edge.edgelist from build_graph_nyc.py instead.

Commands
-------
From fmm_scripts/ (the FMM container mounts fmm_scripts/ at /workspace;
this script is in fmm_scripts/diffrntraj_version/):

    docker compose exec -w /workspace/diffrntraj_version fmm python3 ./convert_fmm_to_diffrntraj.py

If road_embed.txt already exists somewhere:

    docker compose exec -w /workspace/diffrntraj_version fmm python3 ./convert_fmm_to_diffrntraj.py \
        --road-embed-source /path/to/road_embed.txt
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point


FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
INT_RE = re.compile(r"-?\d+")


# Command-line options
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert NYC FMM output to the processed Diff-RNTraj folder layout."
    )

    parser.add_argument(
        "--fmm-output",
        default="./data/nyc_drive_fmm_match.csv",
        help="FMM CSV with id, opath, offset and length.",
    )

    parser.add_argument(
        "--road-network",
        default="./data/road_network/nyc_drive.gpkg",
        help="Canonical NYC road-network GeoPackage.",
    )

    parser.add_argument(
        "--road-layer",
        default="roads",
        help="Layer name in the GeoPackage.",
    )

    parser.add_argument(
        "--output-dir",
        default="../../baselines/diffusion/diff_rntraj/repo/data/NYC",
        help="Root of the final Diff-RNTraj-style NYC dataset.",
    )

    # Change for Diff-RNTraj (NYC): same train/test split as TS-TrajGen
    parser.add_argument(
        "--trip-map",
        default="./data/nyc_gps_points_fmm_trip_id_map_drive.csv",
        help="FMM id -> uid, tid map written by the prep step.",
    )

    parser.add_argument(
        "--test-trips",
        default="../nyc_test.csv",
        help="CSV with uid, tid of the test trips (TS-TrajGen's test set). "
             "Their trajectories go to gen_test/, the rest to gen_train/.",
    )

    parser.add_argument(
        "--dense-graph-max-nodes",
        type=int,
        default=5000,
        help=(
            "Maximum number of road segments for which graph_A.csv will be "
            "materialized as an original-style dense matrix."
        ),
    )

    parser.add_argument(
        "--road-embed-source",
        default=None,
        help=(
            "Optional existing Node2Vec road_embed.txt to copy into graph/. "
            "If omitted, no fake embedding is created."
        ),
    )

    return parser.parse_args()


# Small parsing helpers
def detect_delimiter(path: Path) -> str:
    """
    Detect the top-level FMM CSV delimiter.

    FMM's sequence values can themselves contain commas, so looking only at
    the header is safer than using a generic delimiter sniffer.
    """
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        header = f.readline().rstrip("\r\n")

    for delimiter in (";", "\t", "|"):
        if delimiter in header:
            return delimiter

    return ","


def parse_int_sequence(value: object) -> List[int]:
    """Convert FMM's serialized road-ID sequence into Python integers."""
    if value is None:
        return []

    if isinstance(value, float) and math.isnan(value):
        return []

    # Example: "53,53,72,91" -> [53, 53, 72, 91]
    # Repeated roads are intentionally preserved.
    return [int(x) for x in INT_RE.findall(str(value))]


def parse_float_sequence(value: object) -> List[float]:
    """Convert FMM's serialized offset/length sequence into floats."""
    if value is None:
        return []

    if isinstance(value, float) and math.isnan(value):
        return []

    return [float(x) for x in FLOAT_RE.findall(str(value))]


def normalize_level(value: object) -> str:
    """
    Convert an OSM highway/level value to a JSON/shapefile-friendly string.

    OSMnx can store highway tags as strings or lists.
    """
    if value is None:
        return "unknown"

    try:
        if pd.isna(value):
            return "unknown"
    except (TypeError, ValueError):
        pass

    if isinstance(value, (list, tuple, set, np.ndarray)):
        return "|".join(str(x) for x in value)

    return str(value)


def highway_level(value: object) -> str:
    """
    Road level as the OSM highway type, e.g. "residential" or "motorway_link",
    the same kind of value as the original Porto rn_dict.json.

    Ways with several highway values are stored as a JSON list such as
    '["residential", "tertiary"]'; the first value is used, the same rule
    prepare_fmm_inputs_diffrntraj.py uses for its numeric level.
    """
    text = normalize_level(value)
    if text.startswith("["):
        try:
            values = json.loads(text)
        except ValueError:
            return text
        if isinstance(values, list) and values:
            return str(values[0])
    return text


def line_coords_latlng(geometry) -> List[List[float]]:
    """
    Convert a road geometry into Diff-RNTraj coordinate order:
        [[latitude, longitude], ...]

    Shapely/OSM geometries use (longitude, latitude), so the values are swapped.
    """
    if geometry is None or geometry.is_empty:
        return []

    if geometry.geom_type == "LineString":
        return [[float(y), float(x)] for x, y, *rest in geometry.coords]

    if geometry.geom_type == "MultiLineString":
        result: List[List[float]] = []

        for part in geometry.geoms:
            part_coords = line_coords_latlng(part)

            # Avoid duplicating the shared point where two pieces meet.
            if result and part_coords and result[-1] == part_coords[0]:
                result.extend(part_coords[1:])
            else:
                result.extend(part_coords)

        return result

    raise ValueError(f"Unsupported road geometry type: {geometry.geom_type}")


def write_json(data: object, path: Path) -> None:
    """Write compact JSON to disk."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))


# Load canonical road network and establish road-ID mapping
def load_roads(
    road_path: Path,
    layer: str,
) -> gpd.GeoDataFrame:
    print(f"[1/6] Reading road network: {road_path}")

    roads = gpd.read_file(road_path, layer=layer).copy()

    if "geometry" not in roads.columns:
        raise ValueError("Road network has no geometry column.")

    # GeoPackage sometimes treats `fid` as its internal feature ID and does not
    # return it as a normal attribute. Our canonical file also contains `rid`,
    # so fid can safely be recovered as rid - 1.
    if "fid" in roads.columns:
        roads["fid"] = pd.to_numeric(roads["fid"], errors="raise").astype(int)
    elif "rid" in roads.columns:
        roads["rid"] = pd.to_numeric(roads["rid"], errors="raise").astype(int)
        roads["fid"] = roads["rid"] - 1
        print("      GPKG did not expose fid; reconstructed fid = rid - 1.")
    else:
        raise ValueError(
            "Canonical road network exposes neither fid nor rid."
        )

    if roads["fid"].duplicated().any():
        raise ValueError("Duplicate raw fid values found.")

    # Note: the GeoPackage's `rid` column (fid + 1) is NOT the model road ID.
    # Model IDs are assigned later, only to roads seen in trajectories.
    roads = roads.sort_values("fid").reset_index(drop=True)

    print(f"      Roads: {len(roads):,}")
    return roads


# Model road IDs: only roads observed in matched trajectories
def build_observed_id_map(
    raw_eid_by_length: Dict[int, List[List[int]]],
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    raw fid -> model ID 1..N for every road that appears in a trajectory,
    in ascending fid order. 0 stays reserved for padding.
    """
    observed = sorted({
        fid
        for trajectories in raw_eid_by_length.values()
        for sequence in trajectories
        for fid in sequence
    })

    raw2new = {fid: new for new, fid in enumerate(observed, start=1)}
    new2raw = {new: fid for fid, new in raw2new.items()}

    print(f"      Model road IDs (observed roads only): {len(raw2new):,}")
    return raw2new, new2raw


def remap_sequences(
    raw_eid_by_length: Dict[int, List[List[int]]],
    raw2new: Dict[int, int],
) -> Dict[int, List[List[int]]]:
    return {
        length: [[raw2new[fid] for fid in sequence] for sequence in trajectories]
        for length, trajectories in raw_eid_by_length.items()
    }


# Build extra_file/*.json
def build_road_dictionaries(
    roads: gpd.GeoDataFrame,
    raw2new: Dict[int, int],
    new2raw: Dict[int, int],
    output_dir: Path,
) -> None:
    print("[4/6] Writing extra_file JSON files")

    extra_dir = output_dir / "extra_file"
    extra_dir.mkdir(parents=True, exist_ok=True)

    if "length" not in roads.columns:
        raise ValueError(
            "Canonical road network must contain the OSMnx length column."
        )

    # Level = OSM highway type (string, like Porto). The numeric "level" column
    # from the prep step is only used if the network has no highway column.
    if "highway" in roads.columns:
        level_column = "highway"
    elif "level" in roads.columns:
        level_column = "level"
    else:
        level_column = None

    raw_rn_dict: Dict[str, dict] = {}
    rn_dict: Dict[str, dict] = {}

    for row in roads.itertuples(index=False):
        fid = int(row.fid)

        coords = line_coords_latlng(row.geometry)

        if len(coords) < 2:
            raise ValueError(
                f"Road fid={fid} has fewer than two geometry coordinates."
            )

        length = float(row.length)

        if not np.isfinite(length):
            length = 0.0

        if level_column == "highway":
            level = highway_level(row.highway)
        elif level_column:
            level = normalize_level(getattr(row, level_column))
        else:
            level = "unknown"

        road_data = {
            "coords": coords,
            "length": length,
            "level": level,
        }

        # raw_rn_dict is indexed using FMM's raw fid (full network).
        raw_rn_dict[str(fid)] = road_data

    # rn_dict is indexed by model ID 1..N and holds only observed roads.
    rn_dict = {
        str(new): raw_rn_dict[str(fid)]
        for new, fid in sorted(new2raw.items())
    }

    write_json(
        {str(new): int(fid) for new, fid in sorted(new2raw.items())},
        extra_dir / "new2raw_rid.json",
    )
    write_json(raw_rn_dict, extra_dir / "raw_rn_dict.json")
    write_json(
        {str(fid): int(new) for fid, new in sorted(raw2new.items())},
        extra_dir / "raw2new_rid.json",
    )
    write_json(rn_dict, extra_dir / "rn_dict.json")

    print(f"      raw_rn_dict roads: {len(raw_rn_dict):,}")
    print(f"      rn_dict roads:     {len(rn_dict):,}")



# Read FMM output and build eid/rate sequences
def encode_rates(
    raw_eids: List[int],
    offsets: List[float],
    lengths: List[float],
) -> List[float]:
    """
    Diff-RNTraj moving-rate encoding (inverse of toseq() in model_utils.py):
      first point on a road:    rate = p
      next point on same road:  rate = (p - p_prev) / (1 - p_prev)
    where p = offset / length is the position on the road. The running
    position is tracked with the DECODED value so decoding reproduces the
    clamped positions exactly.
    """
    rates: List[float] = []
    last_eid = None
    last_p = 0.0

    for eid, offset, road_length in zip(raw_eids, offsets, lengths):
        if (
            not np.isfinite(offset)
            or not np.isfinite(road_length)
            or road_length <= 0
        ):
            p = 0.0
        else:
            p = min(1.0, max(0.0, offset / road_length))

        if eid == last_eid:
            if last_p >= 1.0:
                rate = 0.0
            else:
                rate = (p - last_p) / (1.0 - last_p)
                # GPS noise can move the match slightly backwards; Diff-RNTraj
                # cannot represent that, so stay at the previous position.
                rate = min(1.0, max(0.0, rate))
            last_p = last_p + (1.0 - last_p) * rate
        else:
            rate = p
            last_p = p

        last_eid = eid
        rates.append(float(rate))

    return rates


# def convert_fmm_sequences(
#     fmm_path: Path,
#     valid_fids: set,
# ) -> Tuple[Dict[int, List[List[int]]], Dict[int, List[List[float]]]]:
# Change for Diff-RNTraj (NYC): also return which trajectories are test trips
def convert_fmm_sequences(
    fmm_path: Path,
    valid_fids: set,
    test_ids: set = frozenset(),
) -> Tuple[Dict[int, List[List[int]]], Dict[int, List[List[float]]], Dict[int, List[bool]]]:
    """Returns sequences in RAW fids; they are re-indexed afterwards."""
    print(f"[2/6] Reading FMM matches: {fmm_path}")

    delimiter = detect_delimiter(fmm_path)
    matches = pd.read_csv(fmm_path, sep=delimiter)

    required = {"id", "opath", "offset", "length"}
    missing = required.difference(matches.columns)

    if missing:
        raise ValueError(
            f"FMM output is missing {sorted(missing)}. "
            f"Found: {list(matches.columns)}"
        )

    # Diff-RNTraj's training loop groups trajectories by exact sequence length.
    eid_by_length: Dict[int, List[List[int]]] = defaultdict(list)
    rate_by_length: Dict[int, List[List[float]]] = defaultdict(list)
    # Change for Diff-RNTraj (NYC): True if the trajectory belongs to a test trip
    test_by_length: Dict[int, List[bool]] = defaultdict(list)

    skipped = Counter()
    accepted_points = 0

    for _, row in matches.iterrows():
        raw_eids = parse_int_sequence(row["opath"])
        offsets = parse_float_sequence(row["offset"])
        lengths = parse_float_sequence(row["length"])

        if not raw_eids:
            skipped["empty_opath"] += 1
            continue

        # FMM should return one offset and one road length for every matched point.
        if not (len(raw_eids) == len(offsets) == len(lengths)):
            skipped["length_mismatch"] += 1
            continue

        # Do not silently use a road ID absent from the canonical network.
        if any(fid not in valid_fids for fid in raw_eids):
            skipped["unknown_fid"] += 1
            continue

        rates = encode_rates(raw_eids, offsets, lengths)

        seq_len = len(raw_eids)

        # Preserve each matched observation, including repeated road IDs.
        eid_by_length[seq_len].append(raw_eids)
        rate_by_length[seq_len].append(rates)
        # Change for Diff-RNTraj (NYC): mark test trips
        test_by_length[seq_len].append(int(row["id"]) in test_ids)

        accepted_points += seq_len

    # Plain dicts make the pickle stable/readable and preserve ascending lengths.
    eid_by_length = {
        length: eid_by_length[length]
        for length in sorted(eid_by_length)
    }

    rate_by_length = {
        length: rate_by_length[length]
        for length in sorted(rate_by_length)
    }

    accepted_trajs = sum(len(v) for v in eid_by_length.values())

    print(f"      FMM rows: {len(matches):,}")
    print(f"      Accepted trajectories: {accepted_trajs:,}")
    print(f"      Accepted matched points: {accepted_points:,}")

    if skipped:
        print(f"      Skipped: {dict(skipped)}")

    # return eid_by_length, rate_by_length
    # Change for Diff-RNTraj (NYC): also return the test flags
    return eid_by_length, rate_by_length, test_by_length


# Change for Diff-RNTraj (NYC): load the FMM ids of the test trips
def load_test_ids(trip_map_path: Path, test_trips_path: Path) -> set:
    if not test_trips_path.exists():
        print(f"      No test-trip file at {test_trips_path}: no train/test split (gen_train/ = all).")
        return set()
    trip_map = pd.read_csv(trip_map_path, usecols=["id", "uid", "tid"])
    test_trips = pd.read_csv(test_trips_path, usecols=["uid", "tid"]).drop_duplicates()
    test = trip_map.merge(test_trips, on=["uid", "tid"])
    print(f"      Test trips: {len(test_trips):,} in {test_trips_path} -> {len(test):,} FMM trajectories")
    return set(test["id"].astype(int))


# Change for Diff-RNTraj (NYC): split the length-grouped sequences into train and test
def split_by_flag(eid_by_length, rate_by_length, test_by_length, want_test: bool):
    eids, rates = {}, {}
    for length in eid_by_length:
        keep = [i for i, t in enumerate(test_by_length[length]) if t == want_test]
        if keep:
            eids[length] = [eid_by_length[length][i] for i in keep]
            rates[length] = [rate_by_length[length][i] for i in keep]
    return eids, rates


# Change for Diff-RNTraj (NYC): folder name is a parameter (gen_all, gen_train, gen_test)
def save_gen_all(
    eid_by_length: Dict[int, List[List[int]]],
    rate_by_length: Dict[int, List[List[float]]],
    output_dir: Path,
    folder: str = "gen_all",
) -> None:
    # print("[3/6] Writing gen_all files")
    n = sum(len(v) for v in eid_by_length.values())
    print(f"[3/6] Writing {folder} files ({n:,} trajectories)")

    # gen_dir = output_dir / "gen_all"
    gen_dir = output_dir / folder
    gen_dir.mkdir(parents=True, exist_ok=True)

    with (gen_dir / "eid_seqs.bin").open("wb") as f:
        pickle.dump(eid_by_length, f, protocol=pickle.HIGHEST_PROTOCOL)

    with (gen_dir / "rate_seqs.bin").open("wb") as f:
        pickle.dump(rate_by_length, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Index = trajectory length; value = number of trajectories of that length.
    max_length = max(eid_by_length.keys(), default=0)
    length_distribution = np.zeros(max_length + 1, dtype=np.int64)

    for seq_len, trajectories in eid_by_length.items():
        length_distribution[seq_len] = len(trajectories)

    np.save(gen_dir / "length_distri.npy", length_distribution)


# Build road_network/edges.* and road_network/nodes.*
def write_road_network_shapefiles(
    roads: gpd.GeoDataFrame,
    output_dir: Path,
) -> None:
    print("[5/6] Writing road_network edges/nodes shapefiles")

    rn_dir = output_dir / "road_network"
    rn_dir.mkdir(parents=True, exist_ok=True)

    # Diff-RNTraj's loader sets edge eid = edge attribute `fid`.
    # Therefore edges.shp keeps the RAW FMM fid, not rid=fid+1.
    edge_data = pd.DataFrame(
        {
            "fid": roads["fid"].astype(int),
            "highway": (
                roads["highway"].map(normalize_level)
                if "highway" in roads.columns
                else (
                    roads["level"].map(normalize_level)
                    if "level" in roads.columns
                    else pd.Series(["unknown"] * len(roads))
                )
            ),
        }
    )

    edges = gpd.GeoDataFrame(
        edge_data,
        geometry=roads.geometry,
        crs=roads.crs,
    )

    edges.to_file(rn_dir / "edges.shp")

    # The Porto processed folder also contains nodes.shp.
    # We create one point for every unique OSM node referenced by u/v.
    node_records: Dict[str, Point] = {}

    has_uv = "u" in roads.columns and "v" in roads.columns

    for row in roads.itertuples(index=False):
        coords = line_coords_latlng(row.geometry)

        # Convert [lat, lng] back to Point(lng, lat) for the shapefile.
        start_point = Point(coords[0][1], coords[0][0])
        end_point = Point(coords[-1][1], coords[-1][0])

        if has_uv:
            u = str(getattr(row, "u"))
            v = str(getattr(row, "v"))
        else:
            # Fallback: use coordinate strings if u/v were not retained.
            u = f"{start_point.x:.12f},{start_point.y:.12f}"
            v = f"{end_point.x:.12f},{end_point.y:.12f}"

        node_records.setdefault(u, start_point)
        node_records.setdefault(v, end_point)

    nodes = gpd.GeoDataFrame(
        {
            "nid": list(node_records.keys()),
            "geometry": list(node_records.values()),
        },
        geometry="geometry",
        crs=roads.crs,
    )

    nodes.to_file(rn_dir / "nodes.shp")

    print(f"      edges.shp rows: {len(edges):,}")
    print(f"      nodes.shp rows: {len(nodes):,}")


# Build the UTGraph adjacency matrix used by Diff-RNTraj
def build_graph_files(
    eid_by_length: Dict[int, List[List[int]]],
    road_count: int,
    output_dir: Path,
    dense_graph_max_nodes: int,
    road_embed_source: str | None,
) -> None:
    print("[6/6] Preparing graph directory")

    graph_dir = output_dir / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)

    # graph_A.csv creation commented out: training uses the sparse
    # graph_edge.edgelist written by build_graph_nyc.py (graph step) instead.
    # # UTGraph edge weight = number of observed consecutive transitions.
    # transition_counts: Counter[Tuple[int, int]] = Counter()
    #
    # for trajectories in eid_by_length.values():
    #     for sequence in trajectories:
    #         for src_rid, dst_rid in zip(sequence, sequence[1:]):
    #             transition_counts[(src_rid, dst_rid)] += 1
    #
    # graph_csv = graph_dir / "graph_A.csv"
    # graph_zip = graph_dir / "graph_A.zip"
    #
    # if road_count <= dense_graph_max_nodes:
    #     print(
    #         f"      Writing original-style dense graph_A.csv "
    #         f"({road_count:,} x {road_count:,})"
    #     )
    #
    #     # Write one row at a time so we never allocate the entire N x N matrix.
    #     by_source: Dict[int, Dict[int, int]] = defaultdict(dict)
    #
    #     for (src_rid, dst_rid), weight in transition_counts.items():
    #         by_source[src_rid][dst_rid] = int(weight)
    #
    #     with graph_csv.open("w", encoding="utf-8", newline="\n") as f:
    #         for src_rid in range(1, road_count + 1):
    #             row = ["0"] * road_count
    #
    #             for dst_rid, weight in by_source.get(src_rid, {}).items():
    #                 # Matrix index is model_rid - 1 because row/column 0 of the
    #                 # model is added later as padding by Diff-RNTraj.
    #                 row[dst_rid - 1] = str(weight)
    #
    #             f.write(",".join(row))
    #             f.write("\n")
    #
    #     # The Porto folder contains a zipped copy of graph_A.csv.
    #     with zipfile.ZipFile(
    #         graph_zip,
    #         mode="w",
    #         compression=zipfile.ZIP_DEFLATED,
    #         compresslevel=9,
    #     ) as zf:
    #         zf.write(graph_csv, arcname="graph_A.csv")
    #
    #     print(f"      Wrote: {graph_csv}")
    #     print(f"      Wrote: {graph_zip}")
    #
    # else:
    #     cells = road_count * road_count
    #     print(
    #         "      graph_A.csv NOT written automatically."
    #     )
    #     print(
    #         f"      {road_count:,} roads would require a dense "
    #         f"{road_count:,} x {road_count:,} matrix "
    #         f"({cells:,} cells)."
    #     )
    #     print(
    #         "      Do not create a fake sparse graph_A.csv under the same name; "
    #         "the original Diff-RNTraj loader expects a dense matrix."
    #     )
    print("      graph_A.csv not created (training uses graph_edge.edgelist from build_graph_nyc.py).")

    # road_embed.txt is a learned Node2Vec artifact. Never fabricate it.
    if road_embed_source:
        source = Path(road_embed_source)

        if not source.exists():
            raise FileNotFoundError(
                f"--road-embed-source does not exist: {source}"
            )

        shutil.copy2(source, graph_dir / "road_embed.txt")
        print("      Copied road_embed.txt")
    else:
        print(
            "      road_embed.txt not created: it must be learned with Node2Vec "
            "from the UTGraph."
        )


# Main
def main() -> None:
    args = parse_args()

    fmm_path = Path(args.fmm_output)
    road_path = Path(args.road_network)
    output_dir = Path(args.output_dir)

    if not fmm_path.exists():
        raise FileNotFoundError(f"FMM result not found: {fmm_path}")

    if not road_path.exists():
        raise FileNotFoundError(f"Road network not found: {road_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    roads = load_roads(
        road_path,
        args.road_layer,
    )

    # raw_eid_by_length, rate_by_length = convert_fmm_sequences(
    #     fmm_path,
    #     set(roads["fid"].astype(int)),
    # )
    # Change for Diff-RNTraj (NYC): mark TS-TrajGen's test trips while converting
    test_ids = load_test_ids(Path(args.trip_map), Path(args.test_trips))
    raw_eid_by_length, rate_by_length, test_by_length = convert_fmm_sequences(
        fmm_path,
        set(roads["fid"].astype(int)),
        test_ids,
    )

    if not raw_eid_by_length:
        raise RuntimeError(
            "No usable FMM trajectories remained after conversion."
        )

    raw2new, new2raw = build_observed_id_map(raw_eid_by_length)
    eid_by_length = remap_sequences(raw_eid_by_length, raw2new)

    save_gen_all(
        eid_by_length,
        rate_by_length,
        output_dir,
    )

    # Change for Diff-RNTraj (NYC): gen_train/ (used for training and the graph) and gen_test/ (held out)
    for folder, want_test in (("gen_train", False), ("gen_test", True)):
        save_gen_all(
            *split_by_flag(eid_by_length, rate_by_length, test_by_length, want_test),
            output_dir,
            folder,
        )

    build_road_dictionaries(
        roads,
        raw2new,
        new2raw,
        output_dir,
    )

    write_road_network_shapefiles(
        roads,
        output_dir,
    )

    build_graph_files(
        eid_by_length,
        road_count=len(new2raw),
        output_dir=output_dir,
        dense_graph_max_nodes=args.dense_graph_max_nodes,
        road_embed_source=args.road_embed_source,
    )

    print("\nDone.")
    print(f"NYC dataset root: {output_dir}")
    print(f"Set 'id_size': {len(new2raw)} + 1 in multi_main.py / generate_data.py")
    print("\nGenerated target structure:")
    print("  extra_file/")
    print("  gen_all/")
    print("  gen_train/  gen_test/")
    print("  graph/")
    print("  road_network/")
    print(
        "\nNext: the graph step (build_graph_nyc.py) writes the graph files, "
        "and the embed step writes road_embed.txt."
    )


if __name__ == "__main__":
    main()
