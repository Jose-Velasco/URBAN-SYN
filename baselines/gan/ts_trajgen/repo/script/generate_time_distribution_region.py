# 统计每个时间段每条道路的通行平均时间
# 一小时一个时间段
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import json
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(
    description=(
        "Compute average travel time per region per hour from region-level "
        "trajectories (TS-TrajGen region_time_distribution)."
    )
)

# ---- dataset ----
parser.add_argument(
    "--dataset_name",
    type=str,
    default="Xian",
    help="Dataset folder name (use Xian if using symlink).",
)

parser.add_argument(
    "--data_root",
    type=Path,
    default=Path("../data"),
    help="Root directory containing dataset folders.",
)

# ---- inputs ----
parser.add_argument(
    "--region2rid_filename",
    type=str,
    default="region2rid.json",
    help="Region-to-road mapping used to determine number of regions.",
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

parser.add_argument(
    "--include_eval_test",
    action="store_true",
    help="Include eval/test trajectories when computing time distribution (default: train only).",
)

# ---- output ----
parser.add_argument(
    "--output_filename",
    type=str,
    default="region_time_distribution.npy",
    help="Output NumPy file for region time distribution.",
)

args = parser.parse_args()

data_dir: Path = args.data_root / args.dataset_name

region2rid_path: Path = data_dir / args.region2rid_filename
train_path: Path = data_dir / args.train_region_filename
eval_path: Path = data_dir / args.eval_region_filename
test_path: Path = data_dir / args.test_region_filename
output_path: Path = data_dir / args.output_filename

include_eval_test: bool = args.include_eval_test

# with open('../data/Xian/region2rid.json', 'r') as f:
with open(region2rid_path, 'r') as f:
    region2rid = json.load(f)

region_num = len(region2rid)
time_distribution = np.ones((24, region_num))
time_distribution_cnt = {}


# def parse_time(time_in):
#     """
#     将 json 中 time_format 格式的 time 转化为 local datatime
#     """
#     date = datetime.strptime(time_in, '%Y-%m-%dT%H:%M:%SZ')  # 这是 UTC 时间
#     return date

def parse_time(time_in: str) -> pd.Timestamp:
    """
    Parse ISO timestamps with or without milliseconds/timezone suffix.
    """
    return pd.Timestamp(time_in)

# data_file = ['../data/Xian/xianshi_mm_region_train.csv',
#              '../data/Xian/xianshi_mm_region_eval.csv',
#              '../data/Xian/xianshi_mm_region_test.csv']

# select data
data_files = [train_path]

if include_eval_test:
    data_files.extend([eval_path, test_path])

for file in data_files:
    data = pd.read_csv(file)
    for index, row in tqdm(data.iterrows(), total=data.shape[0]):
        rid_list = [int(x) for x in row['region_list'].split(',')]
        time_list = row['time_list'].split(',')
        now_time = parse_time(time_list[0])
        for i in range(len(rid_list) - 1):
            next_time = parse_time(time_list[i+1])
            # .seconds can hide negative/day-crossing behavior.
            # cost_time = (next_time - now_time).seconds
            # This is safer and still preserves the intended average travel-time logic.
            cost_time = (next_time - now_time).total_seconds()
            assert cost_time >= 0
            if cost_time > 0:
                now_rid = rid_list[i]
                now_hour = now_time.hour
                if now_hour not in time_distribution_cnt:
                    time_distribution_cnt[now_hour] = {now_rid: [1, cost_time]}
                elif now_rid not in time_distribution_cnt[now_hour]:
                    time_distribution_cnt[now_hour][now_rid] = [1, cost_time]
                else:
                    time_distribution_cnt[now_hour][now_rid][0] += 1
                    time_distribution_cnt[now_hour][now_rid][1] += cost_time
            now_time = next_time

cnt_times = 0
for hour in time_distribution_cnt:
    for rid in time_distribution_cnt[hour]:
        avg_cost_time = time_distribution_cnt[hour][rid][1] // time_distribution_cnt[hour][rid][0]
        if avg_cost_time > 0:
            time_distribution[hour][rid] = avg_cost_time
            cnt_times += 1

print('cnt {} / {}'.format(cnt_times, 24*region_num))
# np.save('../data/Xian/region_time_distribution', time_distribution)
np.save(output_path, time_distribution)
