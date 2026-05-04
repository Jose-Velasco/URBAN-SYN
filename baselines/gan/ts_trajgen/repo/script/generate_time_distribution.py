# 统计每个时间段每条道路的通行平均时间
# 一小时一个时间段
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(
    description=(
        "Compute average travel time per road per hour from map-matched "
        "trajectories (TS-TrajGen road_time_distribution)."
    )
)

# ---- dataset ----
parser.add_argument(
    "--dataset_name",
    type=str,
    default="Xian",
    help="Dataset folder name (e.g., Xian, nyc via symlink).",
)

parser.add_argument(
    "--data_root",
    type=Path,
    default=Path("../data"),
    help="Root directory containing dataset folders.",
)

# ---- inputs ----
parser.add_argument(
    "--traj_filename",
    type=str,
    default="xianshi_partA_traj_mm_processed.csv",
    help="Trajectory CSV containing rid_list and time_list.",
)

parser.add_argument(
    "--geo_filename",
    type=str,
    default="xian.geo",
    help="Geo file used to determine number of roads (road_num).",
)

# ---- output ----
parser.add_argument(
    "--output_filename",
    type=str,
    default="road_time_distribution.npy",
    help="Output NumPy file for road time distribution.",
)

args = parser.parse_args()

data_dir: Path = args.data_root / args.dataset_name

traj_path: Path = data_dir / args.traj_filename
geo_path: Path = data_dir / args.geo_filename
output_path: Path = data_dir / args.output_filename

# data = pd.read_csv('../data/Xian/xianshi_partA_traj_mm_processed.csv')
data = pd.read_csv(traj_path)

# road_num = 17378
# dynamically compute road_num
road_num = pd.read_csv(geo_path).shape[0]

time_distribution = np.ones((24, road_num))
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

for index, row in tqdm(data.iterrows(), total=data.shape[0]):
    rid_list = [int(x) for x in row['rid_list'].split(',')]
    time_list = row['time_list'].split(',')
    now_time = parse_time(time_list[0])
    for i in range(len(rid_list) - 1):
        next_time = parse_time(time_list[i+1])
        cost_time = (next_time - now_time).seconds
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

print('cnt {} / {}'.format(cnt_times, 24*road_num))
# np.save('../data/Xian/road_time_distribution', time_distribution)
np.save(output_path, time_distribution)
