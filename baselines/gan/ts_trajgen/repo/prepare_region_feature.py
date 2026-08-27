# region_feature 由预训练好的路段 node_feature 聚类得到

"""
script:
  load road GAT
  compute road embeddings
  aggregate road embeddings by region
  save region_feature.pt
"""

from pathlib import Path
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
import torch
from generator.distance_gat_fc import DistanceGatFC
import json
import numpy as np
from utils.map_manager import MapManager
import argparse
from utils.refactor_utils import load_config

parser = argparse.ArgumentParser(
    description=(
        "Prepare region-level GAT node features from road-level node features, "
        "road-to-region mappings, and a pretrained road-level GAT checkpoint."
    )
)

parser.add_argument(
    "--dataset_name",
    type=str,
    required=True,
    help="Dataset folder name. Ex. nyc",
)

parser.add_argument("--device", type=str, default="cuda:0")

parser.add_argument(
    "--data_root",
    type=Path,
    default=Path("../datasets"),
    help="Root data directory containing dataset folders.",
)
parser.add_argument(
    "--geo_path",
    type=Path,
    required=True,
    help="Path to the active .geo file used to dynamically derive road_num.",
)

parser.add_argument(
    "--map_manager_cache_dir",
    type=Path,
    required=True,
    help="Directory used by MapManager to save/load city bounding-box cache.",
)

parser.add_argument(
    "--save_folder",
    type=Path,
    default=Path("./save/nyc"),
    required=True,
    help="Folder containing pretrained road-level GAT checkpoint.",
)
parser.add_argument(
    "--save_file_name",
    type=str,
    default="gat_fc.pt",
    help="Pretrained road-level GAT checkpoint filename.",
)

parser.add_argument(
    "--adjacent_np_filename",
    type=str,
    default="adjacent_mx.npz",
    help="Road-level adjacency sparse matrix filename.",
)
parser.add_argument(
    "--node_feature_filename",
    type=str,
    default="node_feature.pt",
    help="Road-level node feature tensor filename.",
)
parser.add_argument(
    "--rid2region_filename",
    type=str,
    default="rid2region.json",
    help="Mapping from road id to region id.",
)
parser.add_argument(
    "--region2rid_filename",
    type=str,
    default="region2rid.json",
    help="Mapping from region id to road ids.",
)
parser.add_argument(
    "--region_feature_filename",
    type=str,
    default="region_feature.pt",
    help="Output region-level feature tensor filename.",
)

parser.add_argument(
    "--config",
    type=Path,
    required=True,
    help="Path to TS-TrajGen YAML experiment configuration.",
)

args = parser.parse_args()

data_dir: Path = args.data_root / args.dataset_name

checkpoint_path: Path =  args.save_folder / args.save_file_name
adjacent_np_file: Path =  data_dir / args.adjacent_np_filename
node_feature_file: Path =  data_dir / args.node_feature_filename
rid2region_path: Path =  data_dir / args.rid2region_filename
region2rid_path: Path =  data_dir / args.region2rid_filename
region_feature_path: Path =  data_dir / args.region_feature_filename

geo_path: Path = args.geo_path
map_manager_cache_dir: Path = args.map_manager_cache_dir

# save_folder: Path = args.save_folder
# save_file_name: str = args.save_file_name

# 训练参数
# dataset_name = 'Xian'
dataset_name: str = args.dataset_name
device = args.device

experiment_config = load_config(args.config)
config = experiment_config["road"]["generator"]["function_h"].copy()
config["device"] = device

# config = {
#     'embed_dim': 128,
#     'gps_emb_dim': 5,
#     'num_of_heads': 4,
#     'concat': False,
#     'device': device,
#     'distance_mode': 'l2'
# }

# save_folder = './save/Xian/'
# save_folder: Path = args.save_folder
# save_file_name = 'gat_fc.pt'

# 加载 rel
# road_num = 17378
road_num = pd.read_csv(geo_path).shape[0]
# adjacent_np_file = './data/Xian/adjacent_mx.npz'
adj_mx = sp.load_npz(adjacent_np_file)
# 加载 node_feature
# node_feature_file = './data/Xian/node_feature.pt'
node_features = torch.load(node_feature_file).to(device)
# map_manager = MapManager(dataset_name=dataset_name)
map_manager = MapManager(
    dataset_name=dataset_name,
    geo_path=geo_path,
    cache_dir=map_manager_cache_dir,
)
data_feature = {
    'adj_mx': adj_mx,
    'node_features': node_features,
    'img_width': map_manager.img_width,
    'img_height': map_manager.img_height
}
# 加载模型
road_gat = DistanceGatFC(config=config, data_feature=data_feature).to(device)
# road_gat.load_state_dict(torch.load(os.path.join(save_folder, save_file_name), map_location=device))
road_gat.load_state_dict(torch.load(checkpoint_path, map_location=device))

road_gat._setup_node_emb()
# (road_num, feature_dim)
node_emb_feature = road_gat.node_emb_feature
# 构建 road 与 region 的映射矩阵
# with open('./data/Xian/rid2region.json', 'r') as f:
with open(rid2region_path, 'r') as f:
    rid2region = json.load(f)
# with open('./data/Xian/region2rid.json', 'r') as f:
with open(region2rid_path, 'r') as f:
    region2rid = json.load(f)
region_num = len(region2rid)
# (region_num, road_num)
region2rid_mat = np.zeros((region_num, road_num)) # NOTE: this can be very memory expensive mainly depending on large road_num
for rid in tqdm(rid2region):
    region = rid2region[rid]
    region2rid_mat[region][int(rid)] = 1.0

region2rid_mat = torch.FloatTensor(region2rid_mat).to(device)
region_feature = torch.matmul(region2rid_mat, node_emb_feature)
# 保存 region 的 feature
# torch.save(region_feature, './data/Xian/region_feature.pt')
torch.save(region_feature, region_feature_path)

