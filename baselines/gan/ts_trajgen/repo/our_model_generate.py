# The generation script uses the pretrained Function G and Function H/GAT checkpoints rather than GAN-trained full-generator checkpoints.
# Therefore, the current generated outputs represent the imitation-learning pretrained TS-TrajGen pipeline unless the script is modified to load adversarial generator state_dicts.
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.data_util import encode_time
from utils.parser import str2bool
from search import DoubleLayerSearcher
import json
from generator.generator_v4 import GeneratorV4
import torch
import scipy.sparse as sp
from utils.map_manager import MapManager
import argparse
from pathlib import Path
from copy import deepcopy
from utils.refactor_utils import load_config

parser = argparse.ArgumentParser(
    description=(
        "Generate trajectories using pretrained TS-TrajGen "
        "road-level and region-level generators."
    )
)

parser.add_argument('--device', type=str, default='cuda:0')

# ---- dataset ----
parser.add_argument(
    "--dataset_name",
    type=str,
    required=True,
    help="Dataset folder name (Xian, nyc, etc).",
)

parser.add_argument(
    "--data_root",
    type=Path,
    default=Path("./data"),
    help="Root directory containing dataset folders.",
)

# ---- input trajectories ----
parser.add_argument(
    "--true_traj_file",
    type=str,
    default="xianshi_partA_mm_test.csv",
    help="Ground-truth road-level test trajectories used as OD input.",
)

# ---- output ----
parser.add_argument(
    "--generated_trace_output_file",
    type=str,
    default="TS_TrajGen_generate.csv",
    help="Output generated trajectory CSV.",
)

# ---- pretrained road-level models ----
parser.add_argument(
    "--pretrain_gen_file",
    type=Path,
    default=Path("./save/Xian/function_g_fc.pt"),
    help="Pretrained road-level Function G checkpoint.",
)

parser.add_argument(
    "--pretrain_gat_file",
    type=Path,
    default=Path("./save/Xian/gat_fc.pt"),
    help="Pretrained road-level Function H (GAT) checkpoint.",
)

# ---- pretrained region-level models ----
parser.add_argument(
    "--pretrain_region_gen_file",
    type=Path,
    default=Path("./save/Xian/region_function_g_fc.pt"),
    help="Pretrained region-level Function G checkpoint.",
)

parser.add_argument(
    "--pretrain_region_gat_file",
    type=Path,
    default=Path("./save/Xian/region_gat_fc.pt"),
    help="Pretrained region-level Function H (GAT) checkpoint.",
)

# ---- map manager ----
parser.add_argument(
    "--geo_path",
    type=Path,
    required=True,
    help="Path to road network .geo file used by MapManager.",
)

parser.add_argument(
    "--map_manager_cache_dir",
    type=Path,
    default=Path("./data/Xian"),
    help="Directory used by MapManager to cache computed city bounds.",
)

# ---- feature / graph files ----
parser.add_argument(
    "--node_feature_file",
    type=str,
    default="node_feature.pt",
    help="Road-level node feature tensor.",
)

parser.add_argument(
    "--adjacent_np_file",
    type=str,
    default="adjacent_mx.npz",
    help="Road-level sparse adjacency matrix.",
)

parser.add_argument(
    "--region_adjacent_np_file",
    type=str,
    default="region_adj_mx.npz",
    help="Region-level sparse adjacency matrix.",
)

parser.add_argument(
    "--region_feature_file",
    type=str,
    default="region_feature.pt",
    help="Region-level node feature tensor.",
)

# ---- json graph / search files ----
parser.add_argument(
    "--region2rid_file",
    type=str,
    default="region2rid.json",
    help="Region-to-road mapping JSON.",
)

parser.add_argument(
    "--adjacent_list_file",
    type=str,
    default="adjacent_list.json",
    help="Road adjacency list JSON.",
)

parser.add_argument(
    "--rid_gps_file",
    type=str,
    default="rid_gps.json",
    help="Road GPS lookup JSON ([lon, lat]).",
)

parser.add_argument(
    "--road_length_file",
    type=str,
    default="road_length.json",
    help="Road length lookup JSON.",
)

parser.add_argument(
    "--region_adjacent_list_file",
    type=str,
    default="region_adjacent_list.json",
    help="Region adjacency + boundary-road lookup JSON.",
)

parser.add_argument(
    "--region_dist_file",
    type=str,
    default="region_count_dist.npy",
    help="Region distance matrix used during hierarchical search.",
)

parser.add_argument(
    "--region_transfer_file",
    type=str,
    default="region_transfer_prob.json",
    help="Region transfer probability JSON.",
)

parser.add_argument(
    "--rid2region_file",
    type=str,
    default="rid2region.json",
    help="Road-to-region mapping JSON.",
)

# ---- time distributions ----
parser.add_argument(
    "--road_time_distribution_file",
    type=str,
    default="road_time_distribution.npy",
    help="Road-level hourly travel-time distribution.",
)

parser.add_argument(
    "--region_time_distribution_file",
    type=str,
    default="region_time_distribution.npy",
    help="Region-level hourly travel-time distribution.",
)
parser.add_argument(
    "--config",
    type=Path,
    required=True,
    help="Path to TS-TrajGen YAML experiment configuration.",
)

args = parser.parse_args()

# local: bool = args.local
dataset_name: str = args.dataset_name
device: str = args.device

data_dir: Path = args.data_root / args.dataset_name

# trajectory input/output
true_traj_path: Path = data_dir / args.true_traj_file
generate_trace_path: Path = data_dir / args.generated_trace_output_file

# pretrained checkpoints
pretrain_gen_path = args.pretrain_gen_file
pretrain_gat_path = args.pretrain_gat_file
pretrain_region_gen_path = args.pretrain_region_gen_file
pretrain_region_gat_path = args.pretrain_region_gat_file

# map manager
geo_path: Path = args.geo_path
map_manager_cache_dir: Path = args.map_manager_cache_dir

# graph/features
node_feature_path: Path = data_dir / args.node_feature_file
adjacent_np_path: Path = data_dir / args.adjacent_np_file
region_adjacent_np_path: Path = data_dir / args.region_adjacent_np_file
region_feature_path: Path = data_dir / args.region_feature_file

# JSON lookups
region2rid_path: Path =  data_dir / args.region2rid_file
adjacent_list_path: Path =  data_dir / args.adjacent_list_file
rid_gps_path: Path =  data_dir / args.rid_gps_file
road_length_path: Path =  data_dir / args.road_length_file
region_adjacent_list_path: Path =  data_dir / args.region_adjacent_list_file
region_transfer_path: Path =  data_dir / args.region_transfer_file
rid2region_path: Path =  data_dir / args.rid2region_file

# numpy arrays
region_dist_path: Path = data_dir / args.region_dist_file
road_time_distribution_path: Path = data_dir / args.road_time_distribution_file
region_time_distribution_path: Path = data_dir / args.region_time_distribution_file

experiment_config = load_config(args.config)


# setup model configuration to match that of the pretrained
gen_config = deepcopy(experiment_config["road"]["generator"])
gen_config["function_g"]["device"] = device
gen_config["function_h"]["device"] = device

region_gen_config = deepcopy(experiment_config["region"]["generator"])
region_gen_config["function_g"]["device"] = device
region_gen_config["function_h"]["device"] = device

map_manager = MapManager(

    dataset_name=dataset_name,
    geo_path=geo_path,
    cache_dir=map_manager_cache_dir
)

# Load road level and region level data
# 读取路网邻接表
with open(adjacent_list_path, 'r') as f:
    adjacent_list = json.load(f)
# 读取路网 GPS
with open(rid_gps_path, 'r') as f:
    rid_gps = json.load(f)
# 读取路段长度信息
with open(road_length_path, 'r') as f:
    road_length = json.load(f)
# 区域相关信息
with open(region_adjacent_list_path, 'r') as f:
    region_adjacent_list = json.load(f)
region_dist = np.load(region_dist_path)
with open(region_transfer_path, 'r') as f:
    region_transfer_freq = json.load(f)
with open(rid2region_path, 'r') as f:
    rid2region = json.load(f)

road_time_distribution = np.load(road_time_distribution_path)

region_time_distribution = np.load(region_time_distribution_path)

true_traj = pd.read_csv(true_traj_path)

node_features = torch.load(node_feature_path, map_location=device)
adj_mx = sp.load_npz(adjacent_np_path)

region_adj_mx = sp.load_npz(region_adjacent_np_path)
region_features = torch.load(region_feature_path, map_location=device)

road_num = pd.read_csv(geo_path).shape[0]
time_size = experiment_config["data"]["time_size"]

loc_pad = road_num
time_pad = time_size

data_feature = {
    "road_num": road_num + 1,
    "time_size": time_size + 1,
    "road_pad": loc_pad,
    "time_pad": time_pad,
    "adj_mx": adj_mx,
    "node_features": node_features,
    "img_height": map_manager.img_height,
    "img_width": map_manager.img_width,
}

with open(region2rid_path, "r") as f:
    region2rid = json.load(f)

region_num = len(region2rid)

region_data_feature = {
    "road_num": region_num + 1,
    "time_size": time_size + 1,
    "road_pad": region_num,
    "time_pad": time_pad,
    "adj_mx": region_adj_mx,
    "node_features": region_features,
    "img_height": map_manager.img_height,
    "img_width": map_manager.img_width,
}

# 初始化生成器
road_generator = GeneratorV4(config=gen_config, data_feature=data_feature).to(device)
road_generatorv1_state = torch.load(pretrain_gen_path, map_location=device)
road_generator.function_g.load_state_dict(road_generatorv1_state)
road_gat_state = torch.load(pretrain_gat_path, map_location=device)
road_generator.function_h.load_state_dict(road_gat_state)
road_generator.train(False)

region_generator = GeneratorV4(config=region_gen_config, data_feature=region_data_feature).to(device)
region_generatorv1_state = torch.load(pretrain_region_gen_path, map_location=device)
region_generator.function_g.load_state_dict(region_generatorv1_state)
# region_gat_state = torch.load(pretrain_region_gat_file, map_location=device)
region_gat_state = torch.load(pretrain_region_gat_path, map_location=device)
region_generator.function_h.load_state_dict(region_gat_state)
region_generator.train(False)

searcher = DoubleLayerSearcher(device=device, adjacent_list=adjacent_list, road_center_gps=rid_gps, road_length=road_length,
                               region_adjacent_list=region_adjacent_list, region_dist=region_dist, region_transfer_freq=region_transfer_freq,
                               rid2region=rid2region, road_time_distribution=road_time_distribution,
                               region_time_distribution=region_time_distribution, region2rid=region2rid)
# 对每条轨迹都进行一个生成，并将生成结果保存至本地
f = open(generate_trace_path, 'w')
f.write("traj_id,rid_list,time_list\n")
fail_cnt = 0
region_astar_fail_cnt = 0
for index, row in tqdm(true_traj.iterrows(), total=true_traj.shape[0]):
    rid_list = [int(i) for i in row['rid_list'].split(',')]
    mm_id = row['traj_id']
    time_list = list(map(encode_time, row['time_list'].split(',')))
    with torch.no_grad():
        gen_trace_loc, gen_trace_tim, is_astar = searcher.astar_search(region_model=region_generator,
                                                                       road_model=road_generator,
                                                                       start_rid=rid_list[0], start_tim=time_list[0],
                                                                       des=rid_list[-1],
                                                                       default_len=len(rid_list), max_step=5000)
    f.write('{},\"{}\",\"{}\"\n'.format(str(mm_id), ','.join([str(rid) for rid in gen_trace_loc]),
                                        ','.join([str(time) for time in gen_trace_tim])))
    if gen_trace_loc[-1] != rid_list[-1]:
        fail_cnt += 1
    if is_astar == 0:
        region_astar_fail_cnt += 1

print('fail cnt ', fail_cnt)
print('region astar fail cnt ', region_astar_fail_cnt)
f.close()
searcher.save_fail_log()

