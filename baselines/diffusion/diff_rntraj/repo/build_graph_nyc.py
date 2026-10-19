#!/usr/bin/env python3
"""
NYC adaptation of Diff-RNTraj's original build_graph.py.

What stays the same as Diff-RNTraj
----------------------------------
* Every road in new2raw_rid.json is added as a UTGraph node.
* Directed graph edges come from consecutive roads observed in trajectories.
* Edge weight = number of times the road transition is observed.
* graph_A.csv (the dense weighted adjacency matrix) is NOT written for NYC;
  training reads graph_edge.edgelist instead (see the baseline README, "NYC adaptation notes").
* graph_X.csv stores the road-node features.
* graph_edge.edgelist and graph_node_id2idx.txt use 0-based graph indices.

NYC change
----------
The original repository parses the authors' proprietary/intermediate
map-matched trajectory files with ParseMMTraj().  Our NYC pipeline already
converted FMM output into gen_train/eid_seqs.bin (training trips only), so this
version reads those road-ID sequences directly.

Expected dataset layout
-----------------------
data/NYC/
├── extra_file/
│   ├── new2raw_rid.json
│   └── raw_rn_dict.json
├── gen_train/
│   └── eid_seqs.bin
└── graph/

graph_A.csv creation is commented out in main(): the dense N x N matrix is
not needed, because training reads the sparse graph_edge.edgelist.
--write-dense-adjacency and --zip-adjacency are accepted but ignored.

Example
-------
    python build_graph_nyc.py --data-dir ./data/NYC
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm


# Same constants used by the original Diff-RNTraj build_graph.py.
LAT_PER_METER = 8.993203677616966e-06
LNG_PER_METER = 1.1700193970443768e-05


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the NYC Diff-RNTraj UTGraph from eid_seqs.bin."
    )
    p.add_argument(
        "--data-dir",
        default="./data/NYC",
        help="Diff-RNTraj-style NYC dataset root.",
    )
    p.add_argument(
        "--grid-size",
        type=int,
        default=50,
        help="Grid size in meters used for graph_X.csv road endpoint features.",
    )
    p.add_argument(
        "--write-dense-adjacency",
        action="store_true",
        help="Write original-style dense graph_A.csv. Full NYC is ~487 GB.",
    )
    p.add_argument(
        "--dense-chunk-rows",
        type=int,
        default=32,
        help="Rows buffered at a time while streaming graph_A.csv.",
    )
    p.add_argument(
        "--zip-adjacency",
        action="store_true",
        help="After graph_A.csv is written, create graph_A.zip.",
    )
    p.add_argument(
        "--force-dense",
        action="store_true",
        help="Ignore the free-space preflight check for graph_A.csv.",
    )
    return p.parse_args()


def gps2grid(
    lat: float,
    lng: float,
    min_lat: float,
    min_lng: float,
    max_lat: float,
    max_lng: float,
    grid_size: int,
) -> Tuple[int, int]:
    """
    Same coordinate-to-grid idea used by the original Diff-RNTraj script.
    """
    lat_unit = LAT_PER_METER * grid_size
    lng_unit = LNG_PER_METER * grid_size

    max_xid = int((max_lat - min_lat) / lat_unit) + 1
    max_yid = int((max_lng - min_lng) / lng_unit) + 1

    locgrid_x = int((lat - min_lat) / lat_unit) + 1
    locgrid_y = int((lng - min_lng) / lng_unit) + 1

    locgrid_x = min(max(locgrid_x, 0), max_xid)
    locgrid_y = min(max(locgrid_y, 0), max_yid)

    return locgrid_x, locgrid_y


def normalize_sequence(seq: Sequence) -> List[int]:
    """
    Accept either:
      [1, 2, 3]
    or the original dataset-style nested representation:
      [[1], [2], [3]]
    """
    result: List[int] = []

    for item in seq:
        if isinstance(item, (list, tuple, np.ndarray)):
            if len(item) == 0:
                continue
            item = item[0]

        result.append(int(item))

    return result


def iter_eid_sequences(eid_obj) -> Iterator[List[int]]:
    """
    Support both formats:
      * our NYC converter: {length: [sequence, sequence, ...]}
      * original-style list: [sequence, sequence, ...]
    """
    if isinstance(eid_obj, dict):
        def numeric_key(x):
            try:
                return int(x)
            except Exception:
                return x

        for key in sorted(eid_obj.keys(), key=numeric_key):
            for seq in eid_obj[key]:
                yield normalize_sequence(seq)

    elif isinstance(eid_obj, (list, tuple)):
        for seq in eid_obj:
            yield normalize_sequence(seq)

    else:
        raise TypeError(
            f"Unsupported eid_seqs.bin object type: {type(eid_obj).__name__}"
        )


def derive_mbr(raw_rn_dict: Dict[str, dict]) -> Tuple[float, float, float, float]:
    """
    Derive the minimum bounding rectangle from road geometry instead of
    hard-coding the Porto/Chengdu bounds from the original script.
    """
    min_lat = math.inf
    min_lng = math.inf
    max_lat = -math.inf
    max_lng = -math.inf

    for road in raw_rn_dict.values():
        for lat, lng in road["coords"]:
            min_lat = min(min_lat, float(lat))
            min_lng = min(min_lng, float(lng))
            max_lat = max(max_lat, float(lat))
            max_lng = max(max_lng, float(lng))

    if not np.isfinite([min_lat, min_lng, max_lat, max_lng]).all():
        raise ValueError("Could not derive a valid road-network MBR.")

    return min_lat, min_lng, max_lat, max_lng


def road_features(
    raw_rn_dict: Dict[str, dict],
    raw_eid,
    mbr: Tuple[float, float, float, float],
    grid_size: int,
):
    """
    Equivalent to the original get_length_level_loc() helper.
    """
    road = raw_rn_dict[str(raw_eid)]

    length = road["length"]
    level = road["level"]
    coords = road["coords"]

    start_lat, start_lng = float(coords[0][0]), float(coords[0][1])
    end_lat, end_lng = float(coords[-1][0]), float(coords[-1][1])

    min_lat, min_lng, max_lat, max_lng = mbr

    start_lat, start_lng = gps2grid(
        start_lat, start_lng,
        min_lat, min_lng, max_lat, max_lng,
        grid_size,
    )
    end_lat, end_lng = gps2grid(
        end_lat, end_lng,
        min_lat, min_lng, max_lat, max_lng,
        grid_size,
    )

    return start_lat, start_lng, end_lat, end_lng, length, level


def build_utgraph(
    new2raw_rid: Dict[str, object],
    raw_rn_dict: Dict[str, dict],
    eid_seqs_path: Path,
    grid_size: int,
) -> nx.DiGraph:
    """
    Build the same conceptual UTGraph as Diff-RNTraj:

      nodes = ALL roads in new2raw_rid.json
      edges = consecutive road transitions observed in trajectories
      weight = transition frequency
    """
    print("Deriving NYC MBR from raw_rn_dict.json ...")
    mbr = derive_mbr(raw_rn_dict)
    print(
        f"MBR: min_lat={mbr[0]:.6f}, min_lng={mbr[1]:.6f}, "
        f"max_lat={mbr[2]:.6f}, max_lng={mbr[3]:.6f}"
    )

    # Sort numerically so node insertion order is road IDs 1..N.
    road_ids = sorted((int(k) for k in new2raw_rid.keys()))

    # Diff-RNTraj's training code inserts a zero/padding row in front of the
    # graph/embedding data, so our model-road IDs must be contiguous 1..N.
    expected = list(range(1, len(road_ids) + 1))
    if road_ids != expected:
        raise ValueError(
            "new2raw_rid.json keys must be contiguous model IDs 1..N. "
            "Regenerate the NYC conversion files before building the graph."
        )

    G = nx.DiGraph()

    print(f"Adding all {len(road_ids):,} road segments as UTGraph nodes ...")
    for rid in tqdm(road_ids):
        raw_eid = new2raw_rid[str(rid)]
        start_lat, start_lng, end_lat, end_lng, length, level = road_features(
            raw_rn_dict,
            raw_eid,
            mbr,
            grid_size,
        )

        # Keep node names as strings, matching the original build_graph.py.
        G.add_node(
            str(rid),
            start_lat=start_lat,
            start_lng=start_lng,
            end_lat=end_lat,
            end_lng=end_lng,
            length=length,
            level=level,
            freq_cnt=1,
        )

    print(f"Loading map-matched road sequences from {eid_seqs_path} ...")
    with eid_seqs_path.open("rb") as f:
        eid_obj = pickle.load(f)

    sequence_count = 0
    transition_count = 0

    print("Adding trajectory-observed road transitions ...")
    for seq in tqdm(iter_eid_sequences(eid_obj)):
        sequence_count += 1

        if len(seq) < 2:
            continue

        for u_int, v_int in zip(seq, seq[1:]):
            u = str(u_int)
            v = str(v_int)

            if u not in G or v not in G:
                raise KeyError(
                    f"Trajectory contains road ID {u}->{v} that is absent from "
                    "new2raw_rid.json. Check the road-ID mapping."
                )

            # This is the same weighting rule as the original Diff-RNTraj code.
            if G.has_edge(u, v):
                G.edges[u, v]["weight"] += 1
            else:
                G.add_edge(u, v, weight=1)

            transition_count += 1

    print(f"Trajectory sequences read: {sequence_count:,}")
    print(f"Observed road transitions: {transition_count:,}")
    print(f"UTGraph nodes: {G.number_of_nodes():,}")
    print(f"UTGraph unique directed edges: {G.number_of_edges():,}")

    isolates = sum(1 for node in G if G.degree(node) == 0)
    print(f"UTGraph isolated road nodes: {isolates:,}")

    return G


def save_graph_to_pickle(G: nx.DiGraph, graph_dir: Path) -> None:
    """
    Same artifact as the helper in the original source.
    """
    with (graph_dir / "graph.pkl").open("wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_graph_x(G: nx.DiGraph, graph_dir: Path) -> None:
    """
    Save the original graph_X.csv node-feature file.
    """
    path = graph_dir / "graph_X.csv"

    with path.open("w", encoding="utf-8") as f:
        print(
            "node_name/road_id,start_lat,start_lng,end_lat,end_lng,"
            "length,level,freq_cnt",
            file=f,
        )

        for node_name, attrs in G.nodes(data=True):
            print(
                f"{node_name},"
                f"{attrs['start_lat']},"
                f"{attrs['start_lng']},"
                f"{attrs['end_lat']},"
                f"{attrs['end_lng']},"
                f"{attrs['length']},"
                f"{attrs['level']},"
                f"{attrs['freq_cnt']}",
                file=f,
            )


def save_graph_edgelist(G: nx.DiGraph, graph_dir: Path) -> Dict[str, int]:
    """
    Same indexing idea as Diff-RNTraj's original save_graph_edgelist():

      model road ID 1 -> graph index 0
      model road ID 2 -> graph index 1
      ...

    Node2Vec will be trained on these 0-based graph indices because
    multi_main.py later loads embedding index i into SE[i + 1].
    """
    nodelist = list(G.nodes())
    node_id2idx = {node: idx for idx, node in enumerate(nodelist)}

    node_map_path = graph_dir / "graph_node_id2idx.txt"
    with node_map_path.open("w", encoding="utf-8") as f:
        for node, idx in node_id2idx.items():
            print(f"{node}, {idx}", file=f)

    edge_path = graph_dir / "graph_edge.edgelist"
    with edge_path.open("w", encoding="utf-8") as f:
        for src, dst, attrs in G.edges(data=True):
            print(
                f"{node_id2idx[src]} {node_id2idx[dst]} {attrs['weight']}",
                file=f,
            )

    return node_id2idx


# Change for Diff-RNTraj (NYC): sparse replacement for the dense graph_A.csv, used by
# multi_main.py / generate_data.py / models/diff_util.py. The UTGraph is kept as
# sorted int64 edge keys: key = src_id * EDGE_KEY_BASE + dst_id, with model road
# IDs (graph index + 1; ID 0 = padding), i.e. exactly the entries that are 1 (not
# 1e-10) in the original dense spatial_A_trans.
EDGE_KEY_BASE = 1 << 32


def load_graph_edge_keys(path):
    """Sorted, unique int64 edge keys from graph_edge.edgelist ("src_idx dst_idx weight")."""
    e = np.loadtxt(path, ndmin=2)
    e = e[e[:, 2] != 0]  # same rule as build_graph.load_graph_adj_mtx: non-zero weight = edge
    src = e[:, 0].astype(np.int64) + 1
    dst = e[:, 1].astype(np.int64) + 1
    keys = np.unique(src * EDGE_KEY_BASE + dst)
    print("UTGraph edges:", len(keys))
    return keys
# End change


def estimated_graph_a_bytes(n: int) -> int:
    """
    Diff-RNTraj uses NumPy's default %.18e text formatting.

    Your Porto file confirmed this works out to exactly ~25 bytes per
    matrix cell including separators/newlines.
    """
    return n * n * 25


def write_dense_graph_a_streaming(
    G: nx.DiGraph,
    graph_dir: Path,
    chunk_rows: int,
    force_dense: bool,
) -> Path:
    """
    Write graph_A.csv with the same weighted adjacency values as the original
    nx.adjacency_matrix(...).todense() + np.savetxt(..., delimiter=',') path,
    but stream rows in chunks so the full dense matrix never lives in RAM.
    """
    graph_path = graph_dir / "graph_A.csv"

    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    estimated = estimated_graph_a_bytes(n)
    free = shutil.disk_usage(graph_dir).free

    print(
        f"Estimated graph_A.csv size: {estimated / 1e9:.2f} GB "
        f"({estimated / (1024**3):.2f} GiB)"
    )
    print(
        f"Filesystem free space: {free / 1e9:.2f} GB "
        f"({free / (1024**3):.2f} GiB)"
    )

    # Leave a modest safety margin for filesystem metadata and other outputs.
    if not force_dense and free < int(estimated * 1.05):
        raise RuntimeError(
            "Not enough free space for graph_A.csv with a 5% safety margin. "
            "Use HPC scratch with more space, or --force-dense only if the "
            "reported free-space check is misleading."
        )

    # Pre-index outgoing edges so each chunk can be populated quickly.
    outgoing: Dict[int, List[Tuple[int, float]]] = {}
    for src, dst, attrs in G.edges(data=True):
        src_idx = node_to_idx[src]
        dst_idx = node_to_idx[dst]
        outgoing.setdefault(src_idx, []).append(
            (dst_idx, float(attrs.get("weight", 1.0)))
        )

    print(
        f"Writing dense graph_A.csv in chunks of {chunk_rows} rows. "
        "This can take many hours for full NYC."
    )

    with graph_path.open("w", encoding="utf-8", newline="\n") as f:
        for start in tqdm(range(0, n, chunk_rows)):
            stop = min(n, start + chunk_rows)
            block = np.zeros((stop - start, n), dtype=np.float64)

            for src_idx in range(start, stop):
                for dst_idx, weight in outgoing.get(src_idx, []):
                    block[src_idx - start, dst_idx] = weight

            # Default fmt is %.18e, which matches original np.savetxt usage.
            np.savetxt(f, block, delimiter=",")

    return graph_path


def zip_graph_a(graph_path: Path, graph_dir: Path) -> Path:
    """
    Create the graph_A.zip file shown in the processed Porto/Chengdu folders.
    """
    zip_path = graph_dir / "graph_A.zip"

    print(f"Compressing {graph_path.name} -> {zip_path.name} ...")
    with zipfile.ZipFile(
        zip_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as zf:
        zf.write(graph_path, arcname="graph_A.csv")

    return zip_path


def print_graph_statistics(G: nx.DiGraph) -> None:
    degrees = np.asarray([degree for _, degree in G.degree()], dtype=float)

    print("\nUTGraph statistics")
    print("------------------")
    print(f"Num of nodes: {G.number_of_nodes():,}")
    print(f"Num of edges: {G.number_of_edges():,}")

    if len(degrees):
        print(f"Node degree (mean): {degrees.mean():.2f}")
        for percentile in range(0, 101, 20):
            print(
                f"Node degree ({percentile} percentile): "
                f"{np.percentile(degrees, percentile):.2f}"
            )

    edge_weights = np.asarray(
        [attrs["weight"] for _, _, attrs in G.edges(data=True)],
        dtype=float,
    )

    if len(edge_weights):
        print(f"Edge frequency (mean): {edge_weights.mean():.2f}")
        for percentile in range(0, 101, 20):
            print(
                f"Edge frequency ({percentile} percentile): "
                f"{np.percentile(edge_weights, percentile):.2f}"
            )


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    extra_dir = data_dir / "extra_file"
    # gen_dir = data_dir / "gen_all"
    # Change for Diff-RNTraj (NYC): graph edges come from training trajectories only (no test trips)
    gen_dir = data_dir / "gen_train"
    graph_dir = data_dir / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)

    new2raw_path = extra_dir / "new2raw_rid.json"
    raw_rn_path = extra_dir / "raw_rn_dict.json"
    eid_path = gen_dir / "eid_seqs.bin"

    for path in (new2raw_path, raw_rn_path, eid_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    with new2raw_path.open("r", encoding="utf-8") as f:
        new2raw_rid = json.load(f)

    with raw_rn_path.open("r", encoding="utf-8") as f:
        raw_rn_dict = json.load(f)

    G = build_utgraph(
        new2raw_rid=new2raw_rid,
        raw_rn_dict=raw_rn_dict,
        eid_seqs_path=eid_path,
        grid_size=args.grid_size,
    )

    print_graph_statistics(G)

    print("\nSaving graph.pkl ...")
    save_graph_to_pickle(G, graph_dir)

    print("Saving graph_X.csv ...")
    save_graph_x(G, graph_dir)

    print("Saving graph_node_id2idx.txt and graph_edge.edgelist ...")
    save_graph_edgelist(G, graph_dir)

    # Change for Diff-RNTraj (NYC): graph_A.csv creation commented out. Training uses the
    # sparse graph_edge.edgelist (see the baseline README), so the dense matrix is not needed.
    # if args.write_dense_adjacency:
    #     graph_a = write_dense_graph_a_streaming(
    #         G,
    #         graph_dir,
    #         chunk_rows=args.dense_chunk_rows,
    #         force_dense=args.force_dense,
    #     )
    #
    #     if args.zip_adjacency:
    #         zip_graph_a(graph_a, graph_dir)
    # else:
    #     n = G.number_of_nodes()
    #     est = estimated_graph_a_bytes(n)
    #     print(
    #         "\ngraph_A.csv was NOT written because --write-dense-adjacency "
    #         "was not supplied."
    #     )
    #     print(
    #         f"For {n:,} roads, the expected dense CSV is about "
    #         f"{est / 1e9:.2f} GB."
    #     )
    if args.write_dense_adjacency or args.zip_adjacency:
        print("\n--write-dense-adjacency / --zip-adjacency ignored: graph_A.csv creation is commented out.")
    print("\ngraph_A.csv not written: training uses graph_edge.edgelist.")

    print("\nGraph build complete.")
    print(f"Graph directory: {graph_dir}")
    print(
        "Next: run train_road_embed.py to create road_embed.txt."
    )


if __name__ == "__main__":
    main()
