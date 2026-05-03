import json
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
import argparse
# 根据路网连通性，构建交通区域的连通性（邻接矩阵）
# Region adjacency = "which regions connect" & "which roads connect them"


parser = argparse.ArgumentParser(
    description="Construct TS-TrajGen region adjacency from road adjacency and road-to-region mappings."
)

parser.add_argument("--dataset_name", type=str, default="Xian")
parser.add_argument("--data_root", type=str, default="../data")

parser.add_argument("--rel_filename", type=str, default="xian.rel")
parser.add_argument("--adjacent_filename", type=str, default="adjacent_list.json")
parser.add_argument("--rid2region_filename", type=str, default="rid2region.json")
parser.add_argument("--region2rid_filename", type=str, default="region2rid.json")

parser.add_argument("--region_adj_mx_filename_output", type=str, default="region_adj_mx.npz")
parser.add_argument("--region_adjacent_filename_output", type=str, default="region_adjacent_list.json")

args = parser.parse_args()

data_dir: Path = Path(args.data_root) / args.dataset_name


rel_path: Path = data_dir / args.rel_filename
adjacent_path: Path = data_dir / args.adjacent_filename
rid2region_path: Path = data_dir / args.rid2region_filename
region2rid_path: Path = data_dir / args.region2rid_filename

region_adj_mx_path: Path = data_dir / args.region_adj_mx_filename_output
region_adjacent_path: Path = data_dir / args.region_adjacent_filename_output


# 读取路段邻接表
rid_rel = pd.read_csv(rel_path)
# rid_rel = pd.read_csv('../data/Xian/xian.rel')

if adjacent_path.exists():
    with open(adjacent_path, "r", encoding="utf-8") as f:
        rid_adjacent_list: dict[str, list[int]] = json.load(f)
else:
    print(f"WARNING: Missing adjacency file rebuilding it, load/save path: {adjacent_path = }")
    rid_adjacent_list: dict[str, list[int]] = {}
    for index, row in tqdm(rid_rel.iterrows(), total=rid_rel.shape[0], desc='cal road adjacent list'):
        f_rid = str(row['origin_id'])
        t_rid = row['destination_id']
        if f_rid not in rid_adjacent_list:
            rid_adjacent_list[f_rid] = [t_rid]
        else:
            rid_adjacent_list[f_rid].append(t_rid)
    # with open('../data/Xian/adjacent_list.json', 'w') as f:
    with open(adjacent_path, 'w') as f:
        json.dump(rid_adjacent_list, f)

# 读取路段与区域之间的映射关系
# with open('../data/Xian/rid2region.json', 'r') as f:
with open(rid2region_path, 'r') as f:
    rid2region = json.load(f)

# with open('../data/Xian/region2rid.json', 'r') as f:
with open(region2rid_path, 'r') as f:
    region2rid = json.load(f)

region_adjacent_list = {}
"""
区域的邻接表如下构建:
    当前区域: {
        下游区域: 边界路段集合（这个边界路段是下游区域中的路段）
    }
"""
"""
Region adjacency is built as:

Current region -> {
    downstream region -> set of boundary road segments
}
"""


# 使用稀疏矩阵构建邻接矩阵
region_adj_row = []
region_adj_col = []
region_adj_data = []

# all roads inside this region
for region in tqdm(region2rid, desc="cal region adjacent"):
    # region 是 str
    # 遍历该区域所包含的路段，这些路段的可达路段所属的区域即为可达区域
    next_region_dict = {}
    rid_set = region2rid[region]
    # For each road in the region
    for rid in rid_set:
        # rid 是 int
        if str(rid) in rid_adjacent_list:
            for next_rid in rid_adjacent_list[str(rid)]:
                # next_rid 是 int
                # 查找下游路段所属的
                # roads you can go to next
                next_region = rid2region[str(next_rid)]
                if int(region) != next_region:
                    # There is a boundary crossing between regions
                    # Store that relationship
                    # next_region 是当前区域的下游区域
                    if next_region not in next_region_dict:
                        next_region_dict[next_region] = set()
                        next_region_dict[next_region].add(next_rid)
                        # 将边加入稀疏邻接矩阵中
                        region_adj_row.append(int(region))
                        region_adj_col.append(next_region)
                        region_adj_data.append(1.0)
                    else:
                        next_region_dict[next_region].add(next_rid)
                        # 无需加入稀疏矩阵中
    # 将 set 转换为 list
    for next_region in next_region_dict:
        rid_set = next_region_dict[next_region]
        next_region_dict[next_region] = list(rid_set)
    region_adjacent_list[region] = next_region_dict

total_region = len(region2rid)
region_adj_mx = sp.coo_matrix((region_adj_data, (region_adj_row, region_adj_col)),
                              shape=(total_region, total_region))
# 保存生成结果
# sp.save_npz("../data/Xian/region_adj_mx", region_adj_mx)
sp.save_npz(region_adj_mx_path, region_adj_mx)
# with open('../data/Xian/region_adjacent_list.json', 'w') as f:
with open(region_adjacent_path, 'w') as f:
    json.dump(region_adjacent_list, f)

# 进行一些简单的统计
adjacent_cnt = []
border_rid_cnt = []
for region in region_adjacent_list:
    adjacent_cnt.append(len(region_adjacent_list[region]))
    for next_region in region_adjacent_list[region]:
        border_rid_cnt.append(len(region_adjacent_list[region][next_region]))
adjacent_cnt = np.array(adjacent_cnt)
border_rid_cnt = np.array(border_rid_cnt)
print('region adjacent avg: {}, max {}, min {}'.format(np.average(adjacent_cnt), np.max(adjacent_cnt),
                                                       np.min(adjacent_cnt)))
print('border road avg: {}, max {}, min {}'.format(np.average(border_rid_cnt), np.max(border_rid_cnt),
                                                   np.min(border_rid_cnt)))
