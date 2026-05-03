# 生成区域生成器 G 函数部分预训练数据
import pandas as pd
from tqdm import tqdm
import json
from datetime import datetime
from geopy import distance
import numpy as np
import os
import argparse
from pathlib import Path

def str2bool(value):
    """
    Parse common string boolean values for argparse.

    Parameters
    ----------
    value : str | bool
        Raw CLI value.

    Returns
    -------
    bool
        Parsed boolean value.

    Raises
    ------
    argparse.ArgumentTypeError
        If the value cannot be interpreted as boolean.
    """
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "1"):
        return True
    if value.lower() in ("no", "false", "0"):
        return False
    raise argparse.ArgumentTypeError("bool value expected.")

parser = argparse.ArgumentParser(
    description=(
        "Encode region-level trajectories into TS-TrajGen pretraining format "
        "(region-level G/H models)."
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
    type=str,
    default="../data",
    help="Root directory containing dataset folders.",
)

parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory for outputs (defaults to <data_root>/<dataset_name>).",
)

# ---- encoding params ----
parser.add_argument(
    "--max_step",
    type=int,
    default=4,
    help="Maximum step size for encoding transitions used when random_encode=True.",
)

parser.add_argument(
    "--random_encode",
    type=str2bool,
    default=False,
    help=(
        "Use random step encoding to reduce data size (mainly for long trajectories). "
        "Region trajectories are short, so typically disabled."
    ),
)

# ---- inputs ----
parser.add_argument(
    "--rid2region_filename",
    type=str,
    default="rid2region.json",
    help="Mapping from road id -> region id.",
)

parser.add_argument(
    "--region2rid_filename",
    type=str,
    default="region2rid.json",
    help="Mapping from region id -> list of road ids.",
)

parser.add_argument(
    "--rid_gps_filename",
    type=str,
    default="rid_gps.json",
    help="Road-level GPS coordinates.",
)

parser.add_argument(
    "--region_adjacent_filename",
    type=str,
    default="region_adjacent_list.json",
    help="Region adjacency list.",
)

parser.add_argument(
    "--train_region_filename",
    type=str,
    default="xianshi_mm_region_train.csv",
    help="Region-level training trajectories.",
)

parser.add_argument(
    "--eval_region_filename",
    type=str,
    default="xianshi_mm_region_eval.csv",
    help="Region-level validation trajectories.",
)

parser.add_argument(
    "--test_region_filename",
    type=str,
    default="xianshi_mm_region_test.csv",
    help="Region-level test trajectories.",
)

# ---- outputs ----
parser.add_argument(
    "--region_gps_output_filename",
    type=str,
    default="region_gps.json",
    help="Output region-level GPS coordinates.",
)

parser.add_argument(
    "--train_output_filename",
    type=str,
    default="xianshi_region_pretrain_input_train.csv",
    help="Output pretrain train file.",
)

parser.add_argument(
    "--eval_output_filename",
    type=str,
    default="xianshi_region_pretrain_input_eval.csv",
    help="Output pretrain eval file.",
)

parser.add_argument(
    "--test_output_filename",
    type=str,
    default="xianshi_region_pretrain_input_test.csv",
    help="Output pretrain test file.",
)

args = parser.parse_args()

data_dir: Path = Path(args.data_root) / args.dataset_name
output_dir: Path = Path(args.output_dir) if args.output_dir else data_dir

output_dir.mkdir(parents=True, exist_ok=True)

max_step: int = args.max_step
# max_step = 4
# 随机步数 encode，主要是减少数据量，避免过拟合，因为区域轨迹都比较短，所以就不跳步了
random_encode: bool = args.random_encode
# random_encode = True

rid2region_path: Path = data_dir / args.rid2region_filename
region2rid_path: Path = data_dir / args.region2rid_filename
rid_gps_path: Path = data_dir / args.rid_gps_filename
region_adjacent_path: Path = data_dir / args.region_adjacent_filename
train_region_path: Path = data_dir / args.train_region_filename
eval_region_path: Path = data_dir / args.eval_region_filename
test_region_path: Path = data_dir / args.test_region_filename

# outputs
region_gps_path: Path = output_dir / args.region_gps_output_filename
train_output_path: Path = output_dir / args.train_output_filename
eval_output_path: Path = output_dir / args.eval_output_filename
test_output_path: Path = output_dir / args.test_output_filename

dataset_name = args.dataset_name

if dataset_name == 'BJ_Taxi':
    # 读取路段与区域之间的映射关系
    with open('/mnt/data/jwj/TS_TrajGen_data_archive/kaffpa_tarjan_rid2region.json', 'r') as f:
        rid2region = json.load(f)
    # 读取区域邻接表
    with open('/mnt/data/jwj/TS_TrajGen_data_archive/kaffpa_tarjan_region_adjacent_list.json', 'r') as f:
        region_adjacent_list = json.load(f)
    with open('/mnt/data/jwj/TS_TrajGen_data_archive/kaffpa_tarjan_region_gps.json', 'r') as f:
        region_gps = json.load(f)
    train_data = pd.read_csv('/mnt/data/jwj/TS_TrajGen_data_archive/bj_taxi_mm_region_train.csv')
    eval_data = pd.read_csv('/mnt/data/jwj/TS_TrajGen_data_archive/bj_taxi_mm_region_eval.csv')
    test_data = pd.read_csv('/mnt/data/jwj/TS_TrajGen_data_archive/bj_taxi_mm_region_test.csv')
elif dataset_name == 'Porto_Taxi':
    # 读取路段与区域之间的映射关系
    with open('/mnt/data/jwj/TS_TrajGen_data_archive/porto_rid2region.json', 'r') as f:
        rid2region = json.load(f)
    # 读取区域邻接表
    with open('/mnt/data/jwj/TS_TrajGen_data_archive/porto_region_adjacent_list.json', 'r') as f:
        region_adjacent_list = json.load(f)
    with open('/mnt/data/jwj/TS_TrajGen_data_archive/porto_region_gps.json', 'r') as f:
        region_gps = json.load(f)
    train_data = pd.read_csv('/mnt/data/jwj/TS_TrajGen_data_archive/porto_taxi_mm_region_train.csv')
    eval_data = pd.read_csv('/mnt/data/jwj/TS_TrajGen_data_archive/porto_taxi_mm_region_eval.csv')
    test_data = pd.read_csv('/mnt/data/jwj/TS_TrajGen_data_archive/porto_taxi_mm_region_test.csv')
else:
    # 读取路段与区域之间的映射关系
    # with open('/mnt/data/jwj/TS_TrajGen_data_archive/Xian/rid2region.json', 'r') as f:
    with open(rid2region_path, 'r') as f:
        rid2region = json.load(f)
    # 读取区域邻接表
    # with open('/mnt/data/jwj/TS_TrajGen_data_archive/Xian/rid_gps.json', 'r') as f:
    with open(rid_gps_path, 'r') as f:
        rid_gps = json.load(f)
    # with open('/mnt/data/jwj/TS_TrajGen_data_archive/Xian/region2rid.json', 'r') as f:
    with open(region2rid_path, 'r') as f:
        region2rid = json.load(f)
    region_gps = {}
    for region in region2rid:
        rid_set = region2rid[region]
        lat_list = []
        lon_list = []
        for rid in rid_set:
            rid_center_gps = rid_gps[str(rid)]
            lon_list.append(rid_center_gps[0])
            lat_list.append(rid_center_gps[1])
        # TODO: 这里是几何中心，不一定科学
        region_center = (np.average(lon_list), np.average(lat_list))
        region_gps[region] = region_center
    # with open('/mnt/data/jwj/TS_TrajGen_data_archive/Xian/region_gps.json', 'w') as f:
    with open(region_gps_path, 'w') as f:
        json.dump(region_gps, f)
    # with open('/mnt/data/jwj/TS_TrajGen_data_archive/Xian/region_adjacent_list.json', 'r') as f:
    with open(region_adjacent_path, 'r') as f:
        region_adjacent_list = json.load(f)
    # train_data = pd.read_csv('/mnt/data/jwj/TS_TrajGen_data_archive/Xian/xianshi_mm_region_train.csv')
    train_data = pd.read_csv(train_region_path)
    # eval_data = pd.read_csv('/mnt/data/jwj/TS_TrajGen_data_archive/Xian/xianshi_mm_region_eval.csv')
    eval_data = pd.read_csv(eval_region_path)
    # test_data = pd.read_csv('/mnt/data/jwj/TS_TrajGen_data_archive/Xian/xianshi_mm_region_test.csv')
    test_data = pd.read_csv(test_region_path)


def encode_time(timestamp: str) -> int:
    """
    Encode a timestamp into the model's minute-level time slot.

    Supports ISO timestamps with or without milliseconds/timezone suffix.
    Weekdays use [0, 1439], weekends use [1440, 2879].

    Parameters
    ----------
    timestamp : str

    Returns
    -------
    int
        Encoded time slot.
    """
    time = pd.Timestamp(timestamp).to_pydatetime()

    if time.weekday() in (5, 6):
        return time.hour * 60 + time.minute + 1440

    return time.hour * 60 + time.minute


# def encode_time(timestamp):
    # """
    # 编码时间
    # """
    # # 按
    # time = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%SZ')
    # if time.weekday() == 5 or time.weekday() == 6:
    #     return time.hour * 60 + time.minute + 1440
    # else:
    #     return time.hour * 60 + time.minute


def encode_trace(trace, fp):
    """
    编码轨迹

    Args:
        trace: 一条轨迹记录
        fp: 写入编码结果的文件
    """
    region_list = [int(i) for i in trace['region_list'].split(',')]
    time_list = [encode_time(i) for i in trace['time_list'].split(',')]
    des = region_list[-1]
    des_gps = region_gps[str(des)]
    # 训练数据还是感觉有点多
    # 这里为了避免过拟合，还是随机步数 encode 吧
    # 可以做个对比实验看哪个效果好一点
    if not random_encode:
        for i in range(1, len(region_list)):
            cur_loc = region_list[:i]
            cur_time = time_list[:i]
            cur_region = cur_loc[-1]
            if str(cur_region) not in region_adjacent_list or str(region_list[i]) not in region_adjacent_list[str(cur_region)]:
                # 这不应该发生，如果发生了则舍弃掉后面的路径
                # 说明发生了断路
                return
            candidate_set = list(region_adjacent_list[str(cur_region)].keys())
            if len(candidate_set) > 1:
                # 对于有多个候选点的才有学习的价值
                target = str(region_list[i])
                target_index = 0
                candidate_dis = []
                for index, c in enumerate(candidate_set):
                    if c == target:
                        target_index = index
                    c_gps = region_gps[c]
                    dis = distance.distance((des_gps[1], des_gps[0]), (c_gps[1], c_gps[0])).kilometers  # 单位为千米
                    candidate_dis.append(dis)
                # 开始写入编码结果
                cur_loc_str = ",".join([str(i) for i in cur_loc])
                cur_time_str = ",".join([str(i) for i in cur_time])
                candidate_set_str = ",".join([str(i) for i in candidate_set])
                candidate_dis_str = ",".join([str(i) for i in candidate_dis])
                fp.write("\"{}\",\"{}\",{},\"{}\",\"{}\",{}\n".format(cur_loc_str, cur_time_str, des, candidate_set_str,
                                                                      candidate_dis_str, target_index))
    else:
        i = 1
        while i < len(region_list):
            cur_loc = region_list[:i]
            cur_time = time_list[:i]
            cur_region = cur_loc[-1]
            if str(cur_region) not in region_adjacent_list or str(region_list[i]) not in region_adjacent_list[
                str(cur_region)]:
                # 这不应该发生，如果发生了则舍弃掉后面的路径
                # 说明发生了断路
                return
            candidate_set = list(region_adjacent_list[str(cur_region)].keys())
            if len(candidate_set) > 1:
                # 对于有多个候选点的才有学习的价值
                target = str(region_list[i])
                target_index = 0
                candidate_dis = []
                for index, c in enumerate(candidate_set):
                    if c == target:
                        target_index = index
                    c_gps = region_gps[c]
                    dis = distance.distance((des_gps[1], des_gps[0]), (c_gps[1], c_gps[0])).kilometers * 10  # 单位为百米
                    candidate_dis.append(dis)
                # 开始写入编码结果
                cur_loc_str = ",".join([str(i) for i in cur_loc])
                cur_time_str = ",".join([str(i) for i in cur_time])
                candidate_set_str = ",".join([str(i) for i in candidate_set])
                candidate_dis_str = ",".join([str(i) for i in candidate_dis])
                fp.write("\"{}\",\"{}\",{},\"{}\",\"{}\",{}\n".format(cur_loc_str, cur_time_str, des, candidate_set_str,
                                                                      candidate_dis_str, target_index))
            # i 不再是 ++ 而是随机加一定步数
            step = np.random.randint(1, max_step)
            i += step


if __name__ == '__main__':
    if dataset_name == 'BJ_Taxi':
        train_output = open('/mnt/data/jwj/TS_TrajGen_data_archive/{}.csv'.format('bj_taxi_region_pretrain_input_train'), 'w')
        eval_output = open('/mnt/data/jwj/TS_TrajGen_data_archive/{}.csv'.format('bj_taxi_region_pretrain_input_eval'), 'w')
        test_output = open('/mnt/data/jwj/TS_TrajGen_data_archive/{}.csv'.format('bj_taxi_region_pretrain_input_test'), 'w')
    elif dataset_name == 'Porto_Taxi':
        train_output = open('/mnt/data/jwj/TS_TrajGen_data_archive/{}.csv'.format('porto_taxi_region_pretrain_input_train'), 'w')
        eval_output = open('/mnt/data/jwj/TS_TrajGen_data_archive/{}.csv'.format('porto_taxi_region_pretrain_input_eval'), 'w')
        test_output = open('/mnt/data/jwj/TS_TrajGen_data_archive/{}.csv'.format('porto_taxi_region_pretrain_input_test'), 'w')
    else:
        assert dataset_name == 'Xian'
        # train_output = open(
        #     '/mnt/data/jwj/TS_TrajGen_data_archive/Xian/{}.csv'.format('xianshi_region_pretrain_input_train'), 'w')
        train_output = open(train_output_path, 'w')
        # eval_output = open(
        #     '/mnt/data/jwj/TS_TrajGen_data_archive/Xian/{}.csv'.format('xianshi_region_pretrain_input_eval'), 'w')
        eval_output = open(eval_output_path, 'w')
        # test_output = open(
        #     '/mnt/data/jwj/TS_TrajGen_data_archive/Xian/{}.csv'.format('xianshi_region_pretrain_input_test'), 'w')
        test_output = open(test_output_path, 'w')
    train_output.write('trace_loc,trace_time,des,candidate_set,candidate_dis,target\n')
    eval_output.write('trace_loc,trace_time,des,candidate_set,candidate_dis,target\n')
    test_output.write('trace_loc,trace_time,des,candidate_set,candidate_dis,target\n')
    for index, row in tqdm(train_data.iterrows(), total=train_data.shape[0], desc='encode train traj'):
        encode_trace(row, train_output)
    for index, row in tqdm(eval_data.iterrows(), total=eval_data.shape[0], desc='encode eval traj'):
        encode_trace(row, eval_output)
    for index, row in tqdm(test_data.iterrows(), total=test_data.shape[0], desc='encode test traj'):
        encode_trace(row, test_output)
    train_output.close()
    eval_output.close()
    test_output.close()
