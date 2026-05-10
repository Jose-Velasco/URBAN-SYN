# 因为区域数目不多，所以这里可以都计算一个
# sum of road lengths across trajectories -> average -> region distance
import json
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
from geopy import distance
import argparse

# going off of https://github.com/WenMellors/TS-TrajGen/issues/10
# in that issue list 1. The training set data is used ...
#                       ^^^ google translation
# just in case can toggle to include both train/test but might become data leak depending how this
# file is used downstream
# TODO: test this updated version
def ensure_processed_traj_file(
    data_dir: Path,
    processed_filename: str,
    train_filename: str,
    test_filename: str | None = None,
    include_test: bool = False,
) -> Path:
    """
    Ensure the processed trajectory file exists.

    By default, create it from train only to avoid test leakage. If include_test=True,
    combine train and test for broader OD coverage.
    """
    processed_path = data_dir / processed_filename

    if processed_path.exists():
        print(f"[INFO] Using existing processed traj file: {processed_path}")
        return processed_path

    train_path = data_dir / train_filename
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train file: {train_path}")

    dfs = [pd.read_csv(train_path)]

    if include_test:
        if test_filename is None:
            raise ValueError("test_filename is required when include_test=True")

        test_path = data_dir / test_filename
        if not test_path.exists():
            raise FileNotFoundError(f"Missing test file: {test_path}")

        dfs.append(pd.read_csv(test_path))

    traj_df = pd.concat(dfs, ignore_index=True)

    if "rid_list" not in traj_df.columns:
        raise ValueError("Expected column 'rid_list' not found in trajectory file")

    traj_df.to_csv(processed_path, index=False)

    print(f"[INFO] Saved processed traj file: {processed_path}")
    print(f"[INFO] Source: {'train+test' if include_test else 'train only'}")
    print(f"[INFO] Shape: {traj_df.shape}")

    return processed_path

# def ensure_processed_traj_file(
#     data_dir: Path,
#     processed_filename: str,
#     train_filename: str,
#     test_filename: str,
#     ) -> Path:
#     """
#     Ensure the processed trajectory file exists. If not, create it by combining
#     train and test map-matched trajectory CSVs.

#     Parameters
#     ----------
#     data_dir : Path
#         Dataset directory
#     processed_filename : str
#         Target processed trajectory filename
#     train_filename : str
#         Train MM trajectory file
#     test_filename : str
#         Test MM trajectory file

#     Returns
#     -------
#     Path
#         Path to processed trajectory CSV
#     """
#     processed_path = data_dir / processed_filename

#     if processed_path.exists():
#         print(f"[INFO] Using existing processed traj file: {processed_path}")
#         return processed_path

#     print(f"[INFO] Creating processed traj file: {processed_path}")

#     train_path = data_dir / train_filename
#     test_path = data_dir / test_filename

#     if not train_path.exists():
#         raise FileNotFoundError(f"Missing train file: {train_path}")
#     if not test_path.exists():
#         raise FileNotFoundError(f"Missing test file: {test_path}")

#     # Load
#     train_df = pd.read_csv(train_path)
#     test_df = pd.read_csv(test_path)

#     # Combine
#     traj_df = pd.concat([train_df, test_df], ignore_index=True)

#     # Optional: keep only required columns (safe)
#     if "rid_list" not in traj_df.columns:
#         raise ValueError("Expected column 'rid_list' not found in MM files")

#     # Save
#     traj_df.to_csv(processed_path, index=False)

#     print(f"[INFO] Saved processed traj file: {processed_path}")
#     print(f"[INFO] Shape: {traj_df.shape}")

#     return processed_path

parser = argparse.ArgumentParser(
    description=(
        "Compute region-to-region distance matrix using trajectory-based "
        "average travel distance (TS-TrajGen)."
    )
)

# ---- dataset ----
parser.add_argument(
    "--dataset_name",
    type=str,
    default="Xian",
    help="Dataset folder name (e.g., Xian, nyc).",
)

parser.add_argument(
    "--data_root",
    type=Path,
    default=Path("../data"),
    help="Root directory containing dataset folders.",
)

# ---- core files ----
parser.add_argument(
    "--geo_filename",
    type=str,
    default="xian.geo",
    help="Road network .geo file (used to build road_length if missing).",
)

parser.add_argument(
    "--road_length_filename",
    type=str,
    default="road_length.json",
    help="Cached road length dictionary file.",
)

parser.add_argument(
    "--rid2region_filename",
    type=str,
    default="rid2region.json",
    help="Mapping from road id → region id.",
)

parser.add_argument(
    "--region_gps_filename",
    type=str,
    default="region_gps.json",
    help="Region centroid GPS coordinates.",
)

# ---- trajectory inputs ----
parser.add_argument(
    "--processed_traj_filename",
    type=str,
    default="xianshi_partA_traj_mm_processed.csv",
    help="Processed trajectory CSV used for OD distance estimation. Output processed trajectory file used for region distance computation.",
)

parser.add_argument(
    "--train_mm_filename",
    type=str,
    default="xianshi_partA_mm_train.csv",
    help="Map-matched training trajectories (fallback build source).",
)

parser.add_argument(
    "--test_mm_filename",
    type=str,
    default="xianshi_partA_mm_test.csv",
    help="Map-matched test trajectories (fallback build source).  (optional).",
)

parser.add_argument(
    "--include_test",
    action="store_true",
    help="Include test trajectories when building processed trajectory file (default: train only).",
)

# ---- output ----
parser.add_argument(
    "--region_dist_filename",
    type=str,
    default="region_count_dist.npy",
    help="Output region distance matrix (NumPy .npy).",
)

args = parser.parse_args()

data_dir: Path = args.data_root / args.dataset_name

geo_path: Path = data_dir / args.geo_filename
road_length_path: Path = data_dir / args.road_length_filename
rid2region_path: Path = data_dir / args.rid2region_filename
region_gps_path: Path = data_dir / args.region_gps_filename
processed_traj_path: Path = data_dir / args.processed_traj_filename
train_mm_path: Path = data_dir / args.train_mm_filename
test_mm_path: Path = data_dir / args.test_mm_filename
region_dist_path: Path = data_dir / args.region_dist_filename

processed_traj_filename: str = args.processed_traj_filename
train_mm_filename: str = args.train_mm_filename
test_mm_filename: str = args.test_mm_filename

include_test: bool = args.include_test

# 内存可能会炸吗？

# 主要对于 OD 间有多条轨迹的需要取一个平均
distance_dict = {}  # 记录 f_region, t_region, distance
# dataset_name = 'Xian'
dataset_name: str = args.dataset_name
# 读取路网长度字典
# road_len_file = '../data/Xian/road_length.json'
if not os.path.exists(road_length_path):
    # road_info = pd.read_csv('../data/Xian/xian.geo')
    road_info = pd.read_csv(geo_path)
    road_length = {}
    for index, row in tqdm(road_info.iterrows(), desc='cal road length'):
        rid = row['geo_id']
        length = row['length']
        road_length[str(rid)] = length
    # 保存
    with open(road_length_path, 'w') as f:
        json.dump(road_length, f)
else:
    with open(road_length_path, 'r') as f:
        road_length = json.load(f)

# 读取路段区域映射表
# with open('../data/Xian/rid2region.json', 'r') as f:
with open(rid2region_path, 'r') as f:
    rid2region = json.load(f)


# 开始遍历轨迹数据
# traj = pd.read_csv('../data/Xian/xianshi_partA_traj_mm_processed.csv')

traj_path = ensure_processed_traj_file(
    data_dir=data_dir,
    processed_filename=processed_traj_filename,
    train_filename=train_mm_filename,
    test_filename=test_mm_filename,
    include_test=include_test
)

traj = pd.read_csv(traj_path)
for index, row in tqdm(traj.iterrows(), desc='count traj', total=traj.shape[0]):
    rid_list = [int(i) for i in row['rid_list'].split(',')]
    if len(rid_list) < 2:
        continue
    count_length = 0
    step_length = []
    for i in range(len(rid_list)):
        # 因为 road_length 的单位是米，所以这里做个缩放感觉会好一点
        # 目前搞成了千米
        count_length += road_length[str(rid_list[i])]
        step_length.append(count_length)
    for i in range(len(rid_list)):
        f_rid = rid_list[i]
        f_region = rid2region[str(f_rid)]
        for j in range(i + 1, len(rid_list)):
            t_rid = rid_list[j]
            t_region = rid2region[str(t_rid)]
            travel_length = step_length[j] - step_length[i]
            if t_region != f_region:
                if f_region not in distance_dict:
                    distance_dict[f_region] = {}
                    distance_dict[f_region][t_region] = (travel_length, 1)
                elif t_region not in distance_dict[f_region]:
                    distance_dict[f_region][t_region] = (travel_length, 1)
                else:
                    pair = distance_dict[f_region][t_region]
                    distance_dict[f_region][t_region] = (pair[0] + travel_length, pair[1] + 1)

# 还需要计算 f_region 与 t_region 之间的直线距离
# with open('../data/Xian/region_gps.json', 'r') as f:
with open(region_gps_path, 'r') as f:
    region_gps = json.load(f)

# 根据统计得到的 distance_dict 以及经纬度来生成距离矩阵
region_num = len(region_gps)
region_dist = np.zeros((region_num, region_num), dtype=float)
for f_region in tqdm(range(region_num), desc='generate region dist'):
    f_gps = region_gps[str(f_region)]
    for t_region in range(region_num):
        if f_region != t_region:
            if f_region in distance_dict and t_region in distance_dict[f_region]:
                pair = distance_dict[f_region][t_region]
                avg_length = pair[0] / pair[1]
                region_dist[f_region][t_region] = avg_length
            else:
                region_dist[f_region][t_region] = -1

# np.save('../data/Xian/region_count_dist', region_dist)
np.save(region_dist_path, region_dist)
