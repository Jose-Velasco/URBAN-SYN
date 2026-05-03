# 将路段轨迹映射为区域轨迹
import json
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import os


def str2bool(s) -> bool:
    if isinstance(s, bool):
        return s
    if s.lower() in ('yes', 'true'):
        return True
    elif s.lower() in ('no', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError('bool value expected.')


parser = argparse.ArgumentParser(
    description=(
        "Map road-level trajectories (mm_train/test) to region-level trajectories "
        "using rid2region mappings and region adjacency."
    )
)

parser.add_argument('--local', type=str2bool,
                    default=True, help='whether save the trained model')


# ---- dataset ----
parser.add_argument(
    "--dataset_name",
    type=str,
    default="Xian",
    help="Dataset folder name (e.g., Xian, nyc).",
)

parser.add_argument(
    "--data_root",
    type=str,
    default="../data",
    help="Root directory containing dataset folders.",
)

# ---- inputs ----
parser.add_argument(
    "--rid2region_filename",
    type=str,
    default="rid2region.json",
    help="Mapping from road id → region id.",
)

parser.add_argument(
    "--region_adjacent_filename",
    type=str,
    default="region_adjacent_list.json",
    help="Region adjacency list (region → downstream regions).",
)

parser.add_argument(
    "--train_mm_filename",
    type=str,
    default="xianshi_partA_mm_train.csv",
    help="Map-matched training trajectories (road-level).",
)

parser.add_argument(
    "--test_mm_filename",
    type=str,
    default="xianshi_partA_mm_test.csv",
    help="Map-matched test trajectories (road-level).",
)

# ---- outputs ----
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help=(
        "Directory to write region-level trajectory CSVs. "
        "Defaults to <data_root>/<dataset_name>/"
    ),
)

parser.add_argument(
    "--train_region_filename",
    type=str,
    default="xianshi_mm_region_train.csv",
    help="Output region-level training trajectories.",
)

parser.add_argument(
    "--eval_region_filename",
    type=str,
    default="xianshi_mm_region_eval.csv",
    help="Output region-level validation trajectories.",
)

parser.add_argument(
    "--test_region_filename",
    type=str,
    default="xianshi_mm_region_test.csv",
    help="Output region-level test trajectories.",
)

# ---- split ----
parser.add_argument(
    "--train_rate",
    type=float,
    default=0.9,
    help="Train/validation split ratio for region-level data.",
)


args = parser.parse_args()

local: bool = args.local
dataset_name: str = args.dataset_name

if local:
    data_root: Path = Path(args.data_root)
    # data_root = '../data/'
    data_dir: Path = data_root / dataset_name
else:
    data_root: Path = Path('/mnt/data/jwj/TS_TrajGen_data_archive/')
    data_dir: Path = data_root

output_dir: Path = Path(args.output_dir) if args.output_dir else data_dir

output_dir.mkdir(parents=True, exist_ok=True)

# inputs
rid2region_path: Path = data_dir / args.rid2region_filename
region_adjacent_path: Path = data_dir / args.region_adjacent_filename
train_mm_path: Path = data_dir / args.train_mm_filename
test_mm_path: Path = data_dir / args.test_mm_filename

# outputs
train_region_path: Path = output_dir / args.train_region_filename
eval_region_path: Path = output_dir / args.eval_region_filename
test_region_path: Path = output_dir / args.test_region_filename

if dataset_name == 'BJ_Taxi':
    # 读取路段与区域之间的映射关系
    with open('/mnt/data/jwj/TS_TrajGen_data_archive/kaffpa_tarjan_rid2region.json', 'r') as f:
        rid2region = json.load(f)
    # 读取区域邻接表
    with open('/mnt/data/jwj/TS_TrajGen_data_archive/kaffpa_tarjan_region_adjacent_list.json', 'r') as f:
        region_adjacent_list = json.load(f)
    train_mm_traj = pd.read_csv('/mnt/data/jwj/BJ_Taxi/chaoyang_traj_mm_train.csv')
    test_mm_traj = pd.read_csv('/mnt/data/jwj/BJ_Taxi/chaoyang_traj_mm_test.csv')
    # 开始 Map
    headers = 'traj_id,region_list,time_list\n'
    train_file = open('/mnt/data/jwj/TS_TrajGen_data_archive/bj_taxi_mm_region_train.csv', 'w')
    eval_file = open('/mnt/data/jwj/TS_TrajGen_data_archive/bj_taxi_mm_region_eval.csv', 'w')
    test_file = open('/mnt/data/jwj/TS_TrajGen_data_archive/bj_taxi_mm_region_test.csv', 'w')
elif dataset_name == 'Porto_Taxi':
    # 读取路段与区域之间的映射关系
    with open('/mnt/data/jwj/TS_TrajGen_data_archive/porto_rid2region.json', 'r') as f:
        rid2region = json.load(f)
    # 读取区域邻接表
    with open('/mnt/data/jwj/TS_TrajGen_data_archive/porto_region_adjacent_list.json', 'r') as f:
        region_adjacent_list = json.load(f)
    train_mm_traj = pd.read_csv('/mnt/data/jwj/Porto_Taxi/porto_mm_train.csv')
    test_mm_traj = pd.read_csv('/mnt/data/jwj/Porto_Taxi/porto_mm_test.csv')
    # 开始 Map
    headers = 'traj_id,region_list,time_list\n'
    train_file = open('/mnt/data/jwj/TS_TrajGen_data_archive/porto_taxi_mm_region_train.csv', 'w')
    eval_file = open('/mnt/data/jwj/TS_TrajGen_data_archive/porto_taxi_mm_region_eval.csv', 'w')
    test_file = open('/mnt/data/jwj/TS_TrajGen_data_archive/porto_taxi_mm_region_test.csv', 'w')
else:
    # Xian
    # 读取路段与区域之间的映射关系
    # with open(os.path.join(data_root, dataset_name, 'rid2region.json'), 'r') as f:
    with open(rid2region_path, 'r') as f:
        rid2region = json.load(f)
    # 读取区域邻接表
    # with open(os.path.join(data_root, dataset_name, 'region_adjacent_list.json'), 'r') as f:
    with open(region_adjacent_path, 'r') as f:
        region_adjacent_list = json.load(f)
    # train_mm_traj = pd.read_csv('/mnt/data/jwj/Xian/xianshi_partA_mm_train.csv')
    train_mm_traj = pd.read_csv(train_mm_path)
    # test_mm_traj = pd.read_csv('/mnt/data/jwj/Xian/xianshi_partA_mm_test.csv')
    test_mm_traj = pd.read_csv(test_mm_path)
    # 开始 Map
    headers = 'traj_id,region_list,time_list\n'
    # train_file = open('/mnt/data/jwj/TS_TrajGen_data_archive/Xian/xianshi_mm_region_train.csv', 'w')
    train_file = open(train_region_path, 'w')
    # eval_file = open('/mnt/data/jwj/TS_TrajGen_data_archive/Xian/xianshi_mm_region_eval.csv', 'w')
    eval_file = open(eval_region_path, 'w')
    # test_file = open('/mnt/data/jwj/TS_TrajGen_data_archive/Xian/xianshi_mm_region_test.csv', 'w')
    test_file = open(test_region_path, 'w')
train_file.write(headers)
eval_file.write(headers)
test_file.write(headers)


def write_row(write_file, write_row, region_list, time_list):
    """
    写入结果
    Args:
        write_file:
        write_row:
        region_list:
        time_list:

    Returns:

    """
    traj_id = write_row['traj_id']
    map_region_str = ','.join([str(x) for x in region_list])
    map_time_str = ','.join(time_list)
    write_file.write('{},\"{}\",\"{}\"\n'.format(traj_id, map_region_str, map_time_str))


train_rate = 0.9
total_data_num = train_mm_traj.shape[0]
train_num = int(total_data_num * train_rate)

for index, row in tqdm(train_mm_traj.iterrows(), total=train_mm_traj.shape[0], desc='map traj'):
    # map
    rid_list = row['rid_list'].split(',')
    time_list = row['time_list'].split(',')
    start_region = rid2region[rid_list[0]]
    start_time = time_list[0]
    map_region_list = [start_region]
    map_time_list = [start_time]
    for j, rid in enumerate(rid_list[1:]):
        map_region = rid2region[rid]
        if map_region != map_region_list[-1]:
            map_region_list.append(map_region)
            map_time_list.append(time_list[j+1])
    if index <= train_num:
        write_row(train_file, row, map_region_list, map_time_list)
    else:
        write_row(eval_file, row, map_region_list, map_time_list)

for index, row in tqdm(test_mm_traj.iterrows(), total=test_mm_traj.shape[0], desc='map traj'):
    # map
    rid_list = row['rid_list'].split(',')
    time_list = row['time_list'].split(',')
    start_region = rid2region[rid_list[0]]
    start_time = time_list[0]
    map_region_list = [start_region]
    map_time_list = [start_time]
    for j, rid in enumerate(rid_list[1:]):
        map_region = rid2region[rid]
        if map_region != map_region_list[-1]:
            map_region_list.append(map_region)
            map_time_list.append(time_list[j+1])
    write_row(test_file, row, map_region_list, map_time_list)


train_file.close()
eval_file.close()
test_file.close()
