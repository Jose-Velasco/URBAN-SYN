from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LibCity .geo/.rel road network files into KaHIP graph format."
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Xian",
        help="Dataset folder name under data_root.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="../data",
        help="Root directory containing dataset folders.",
    )
    parser.add_argument(
        "--geo_filename",
        type=str,
        default="xian.geo",
        help="Geo filename inside the dataset folder.",
    )
    parser.add_argument(
        "--rel_filename",
        type=str,
        default="xian.rel",
        help="Rel filename inside the dataset folder.",
    )
    parser.add_argument(
        "--graph_filename",
        type=str,
        default="xian.graph",
        help="Output KaHIP graph filename.",
    )
    parser.add_argument(
        "--rid2new_filename",
        type=str,
        default="rid2new.json",
        help="Output original-road-id to KaHIP-node-id mapping filename.",
    )
    parser.add_argument(
        "--new2rid_filename",
        type=str,
        default="new2rid.json",
        help="Output KaHIP-node-id to original-road-id mapping filename.",
    )

    return parser.parse_args()

def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    """
    Resolve all input/output paths from CLI arguments.
    """
    data_dir = Path(args.data_root) / args.dataset_name

    return {
        "data_dir": data_dir,
        "geo_path": data_dir / args.geo_filename,
        "rel_path": data_dir / args.rel_filename,
        "graph_path": data_dir / args.graph_filename,
        "rid2new_path": data_dir / args.rid2new_filename,
        "new2rid_path": data_dir / args.new2rid_filename,
    }

def build_active_road_mappings(road_info: pd.DataFrame, road_rel: pd.DataFrame):
    """
    Build 1-indexed KaHIP node ids for roads that appear in at least one relation.

    KaHIP expects node ids starting at 1, while our road ids start at 0.
    """
    connected_roads = set(road_rel["origin_id"]).union(set(road_rel["destination_id"]))

    rid2new: dict[int, int] = {}
    new2rid: dict[int, int] = {}

    new_id = 1
    for rid in road_info["geo_id"]:
        rid = int(rid)
        if rid not in connected_roads:
            continue

        rid2new[rid] = new_id
        new2rid[new_id] = rid
        new_id += 1

    return rid2new, new2rid


def build_undirected_adjacency(
    road_rel: pd.DataFrame,
    rid2new: dict[int, int],
) -> dict[int, set[int]]:
    """
    Build sparse undirected adjacency for KaHIP graph format.

    The original script used a dense N x N matrix, which is too large for NYC.
    """
    adjacency: dict[int, set[int]] = defaultdict(set)

    for _, row in tqdm(
        road_rel.iterrows(),
        total=road_rel.shape[0],
        desc="build sparse undirected adjacency",
    ):
        origin = int(row["origin_id"])
        destination = int(row["destination_id"])

        if origin not in rid2new or destination not in rid2new:
            continue

        u = rid2new[origin]
        v = rid2new[destination]

        # KaHIP graph is undirected; skip self-loops.
        if u == v:
            continue

        adjacency[u].add(v)
        adjacency[v].add(u)

    return adjacency


def write_kahip_graph(
    graph_path: Path,
    total_node_num: int,
    adjacency: dict[int, set[int]],
) -> None:
    """
    Write KaHIP/METIS-style graph file.

    First line: <num_nodes> <num_edges>
    Following lines: neighbors for node 1, node 2, ..., node N.
    """
    total_edge_num = sum(len(neighbors) for neighbors in adjacency.values()) // 2

    with open(graph_path, "w", encoding="utf-8") as f:
        f.write(f"{total_node_num} {total_edge_num}\n")

        for node_id in range(1, total_node_num + 1):
            neighbors = sorted(adjacency.get(node_id, set()))
            f.write(" ".join(str(x) for x in neighbors) + "\n")

    print(f"nodes: {total_node_num:,}")
    print(f"edges: {total_edge_num:,}")
    print(f"saved: {graph_path}")


def save_mapping(path: Path, mapping: dict[int, int]) -> None:
    """
    Save integer mapping as JSON.

    JSON keys become strings, so convert back to int when loading later.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f)


def main() -> None:
    """
    Convert road network .rel into sparse KaHIP graph format.
    """
    args = parse_args()
    paths = resolve_paths(args)

    data_dir = paths["data_dir"]
    geo_path = paths["geo_path"]
    rel_path = paths["rel_path"]
    graph_path = paths["graph_path"]
    rid2new_path = paths["rid2new_path"]
    new2rid_path = paths["new2rid_path"]

    data_dir.mkdir(parents=True, exist_ok=True)

    road_info = pd.read_csv(geo_path)
    road_rel = pd.read_csv(rel_path)

    rid2new, new2rid = build_active_road_mappings(road_info, road_rel)
    adjacency = build_undirected_adjacency(road_rel, rid2new)

    total_node_num = len(new2rid)
    write_kahip_graph(graph_path, total_node_num, adjacency)

    save_mapping(rid2new_path, rid2new)
    save_mapping(new2rid_path, new2rid)


if __name__ == "__main__":
    main()



# # 将路网输出为 metis 格式的图描述
# import numpy as np
# import pandas as pd
# from tqdm import tqdm

# dataset_name = 'Xian'

# with open('../data/{}/xian.graph'.format(dataset_name), 'w') as f:

#     if dataset_name == 'Xian':
#         road_info = pd.read_csv('../data/Xian/xian.geo')
#         road_rel = pd.read_csv('../data/Xian/xian.rel')

#     total_road_num = road_info.shape[0]

#     # 需要删除孤立的路段，因此需要做一个重编码（双映射）
#     # 找到孤立点
#     # outlier_set = roads that NEVER appear in any edge
#     # isolated nodes
#     outlier_set = set(road_info['geo_id'])
#     for index, row in tqdm(road_rel.iterrows(), total=road_rel.shape[0], desc='find outlier'):
#         f_id = row['origin_id']
#         t_id = row['destination_id']
#         if f_id in outlier_set:
#             outlier_set.remove(f_id)
#         if t_id in outlier_set:
#             outlier_set.remove(t_id)
#     print(outlier_set)
#     # original road_id -> new compact ID (1…N)
#     rid2new = {}
#     # reverse mapping
#     new2rid = {}
#     # KaHIP uses 1-based indexing
#     new_id = 1
#     for rid in range(total_road_num):
#         if rid not in outlier_set:
#             # 这个点不需要被删除，重新编码
#             rid2new[rid] = new_id
#             new2rid[new_id] = rid
#             new_id += 1

#     # 因为图分割算法只能处理无向图，所以这里边需要做额外的处理
#     total_road_num = len(new2rid)
#     assert total_road_num + 1 == new_id
#     road_undirected_adj_mx = np.zeros((total_road_num, total_road_num)).astype(int)
#     road_undirected_rel = {}
#     total_edge_num = 0
#     # 注意 road id 从 1 开始
#     # KaHIP uses 1-based indexing
#     for index, row in tqdm(road_rel.iterrows(), total=road_rel.shape[0]):
#         from_road = rid2new[row['origin_id']]
#         to_road = rid2new[row['destination_id']]
#         if from_road == to_road:
#             # 自环就跳过了
#             continue
#         min_road = min(from_road, to_road)
#         max_road = max(from_road, to_road)
#         if min_road not in road_undirected_rel:
#             road_undirected_rel[min_road] = {max_road}
#             road_undirected_adj_mx[min_road - 1][max_road - 1] = 1
#             road_undirected_adj_mx[max_road - 1][min_road - 1] = 1
#             total_edge_num += 1
#         elif max_road not in road_undirected_rel[min_road]:
#             road_undirected_rel[min_road].add(max_road)
#             road_undirected_adj_mx[min_road - 1][max_road - 1] = 1
#             road_undirected_adj_mx[max_road - 1][min_road - 1] = 1
#             total_edge_num += 1

#     f.write('{} {}\n'.format(total_road_num, total_edge_num))
#     print(total_road_num, total_edge_num)
#     output_cnt = 0
#     for road_id in range(1, total_road_num + 1):
#         road_adjacent = (np.where(road_undirected_adj_mx[road_id - 1] == 1)[0] + 1).tolist()
#         output_cnt += len(road_adjacent)
#         adjacent_str = ' '.join([str(x) for x in road_adjacent])
#         f.write(adjacent_str + '\n')
#     print(output_cnt)
