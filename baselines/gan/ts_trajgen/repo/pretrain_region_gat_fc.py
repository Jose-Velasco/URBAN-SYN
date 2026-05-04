import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
import torch
from generator.distance_gat_fc import DistanceGatFC
from torch.utils.data import DataLoader
from utils.ListDataset import ListDataset
from utils.utils import get_logger
import json
import numpy as np
from utils.parser import str2bool
import argparse


parser = argparse.ArgumentParser(
    description=(
        "Pretrain region-level GAT (Function H) using region adjacency, "
        "region features, and trajectory pretraining data (TS-TrajGen)."
    )
)

# parser.add_argument('--local', type=str2bool, default=True)
# parser.add_argument('--debug', type=str2bool, default=False)

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
    default=Path("./data"),
    help="Root directory containing dataset folders.",
)

parser.add_argument(
    "--device",
    type=str,
    default="cuda:0",
    help="Torch device (e.g., cuda:0, cpu).",
)

# ---- inputs ----
parser.add_argument(
    "--region2rid_filename",
    type=str,
    default="region2rid.json",
    help="Mapping from region id → list of road ids.",
)

parser.add_argument(
    "--adjacent_np_filename",
    type=str,
    default="region_adj_mx.npz",
    help="Region adjacency sparse matrix file.",
)

parser.add_argument(
    "--node_feature_filename",
    type=str,
    default="region_feature.pt",
    help="Region-level node feature tensor.",
)

parser.add_argument(
    "--region_dist_filename",
    type=str,
    default="region_count_dist.npy",
    help="Region-to-region distance matrix.",
)

parser.add_argument(
    "--train_filename",
    type=str,
    default="xianshi_region_pretrain_input_train.csv",
    help="Region-level pretrain training data.",
)

parser.add_argument(
    "--eval_filename",
    type=str,
    default="xianshi_region_pretrain_input_eval.csv",
    help="Region-level pretrain validation data.",
)

parser.add_argument(
    "--test_filename",
    type=str,
    default="xianshi_region_pretrain_input_test.csv",
    help="Region-level pretrain test data.",
)

# ---- training control ----
parser.add_argument(
    "--train",
    action="store_true",
    default=False,
    help="Enable training mode.",
)

# ---- outputs ----
parser.add_argument(
    "--save_dir",
    type=Path,
    default=Path("./save/Xian"),
    help="Directory to save/load model (default: ./save/<dataset_name>).",
)

parser.add_argument(
    "--save_file_name",
    type=str,
    default="region_gat_fc.pt",
    help="Model checkpoint filename.",
)

args = parser.parse_args()
# local = args.local
dataset_name: str = args.dataset_name
device: str = args.device
# debug = args.debug

data_dir: Path = args.data_root / args.dataset_name
save_dir: Path = args.save_dir

save_dir.mkdir(parents=True, exist_ok=True)

region2rid_path: Path = data_dir / args.region2rid_filename
adjacent_np_path: Path = data_dir / args.adjacent_np_filename
node_feature_path: Path = data_dir / args.node_feature_filename
region_dist_path: Path = data_dir / args.region_dist_filename
train_path: Path = data_dir / args.train_filename
eval_path: Path = data_dir / args.eval_filename
test_path: Path = data_dir / args.test_filename
save_path: Path = save_dir / args.save_file_name

# archive_data_folder = 'TS_TrajGen_data_archive'

# if local:
#     data_root = './data/'
# else:
#     data_root = '/mnt/data/jwj/TS_TrajGen_data_archive/'

# 训练参数
batch_size = 32
if dataset_name == 'BJ_Taxi' or dataset_name == 'Porto_Taxi':
    config = {
        'embed_dim': 128,
        'gps_emb_dim': 5,
        'num_of_heads': 5,
        'concat': False,
        'device': device,
        'distance_mode': 'l2',
        'no_gps_emb': True
    }
else:
    # Xian
    config = {
        'embed_dim': 68,
        'gps_emb_dim': 5,
        'num_of_heads': 4,
        'concat': False,
        'device': device,
        'distance_mode': 'l2',
        'no_gps_emb': True
    }
max_epoch = 50
learning_rate = 0.0005
weight_decay = 0.0001
lr_patience = 2
lr_decay_ratio = 0.01
early_stop_lr = 1e-6

# save_folder = './save/{}'.format(dataset_name)
save_folder: Path = save_dir
# save_file_name = 'region_gat_fc.pt'
temp_folder = './temp/{}/gat/'.format(dataset_name)
train: bool = args.train

logger = get_logger(name='RegionGatDis')
logger.info('read data')
# with open(os.path.join(data_root, dataset_name, 'region2rid.json'), 'r') as f:
with open(region2rid_path, 'r') as f:
    region2rid = json.load(f)
# 数据集的大小
road_num = len(region2rid)
road_num_with_pad = road_num + 1
# adjacent_np_file = os.path.join(data_root, dataset_name, 'region_adj_mx.npz')
# adjacent_np_file: Path = adjacent_np_path

adj_mx = sp.load_npz(adjacent_np_path)

# 加载区域 region_feature
# node_feature_file = os.path.join(data_root, dataset_name, 'region_feature.pt')
# node_features = torch.load(node_feature_file, map_location='cpu').to(device)
node_features = torch.load(node_feature_path, map_location='cpu').to(device)

data_feature = {
    'adj_mx': adj_mx,
    'node_features': node_features
}

# 加载模型
gat = DistanceGatFC(config=config, data_feature=data_feature).to(device)
logger.info('init gat')
logger.info(gat)
optimizer = torch.optim.Adam(gat.parameters(), lr=learning_rate, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=lr_patience, factor=lr_decay_ratio)
# 加载训练数据
# 读取训练输入数据
if dataset_name == 'BJ_Taxi':
    train_data = pd.read_csv('./data/201511_region_pretrain_input_train.csv')
    eval_data = pd.read_csv('./data/201511_region_pretrain_input_eval.csv')
    test_data = pd.read_csv('./data/201511_region_pretrain_input_test.csv')
else:
    # Xian
    # train_data = pd.read_csv(os.path.join(data_root, dataset_name, 'xianshi_region_pretrain_input_train.csv'))
    train_data = pd.read_csv(train_path)
    # eval_data = pd.read_csv(os.path.join(data_root, dataset_name, 'xianshi_region_pretrain_input_eval.csv'))
    eval_data = pd.read_csv(eval_path)
    # test_data = pd.read_csv(os.path.join(data_root, dataset_name, 'xianshi_region_pretrain_input_test.csv'))
    test_data = pd.read_csv(test_path)

train_data = train_data.values.tolist()
eval_data = eval_data.values.tolist()
test_data = test_data.values.tolist()

train_num = len(train_data)
eval_num = len(eval_data)
test_num = len(test_data)
total_data = train_num + eval_num + test_num
logger.info('total input record is {}. train set: {}, val set {}, test set {}'.format(total_data, train_num,
                                                                                      eval_num, test_num))

train_dataset = ListDataset(train_data)
eval_dataset = ListDataset(eval_data)
test_dataset = ListDataset(test_data)

# region_dist = np.load(os.path.join(data_root, dataset_name, 'region_count_dist.npy'))
region_dist = np.load(region_dist_path)


# 自定义收集函数
def collate_fn(indices):
    batch_des = []
    batch_candidate_set = []
    batch_candidate_dis = []
    batch_target = []
    candidate_set_len = []
    for item in indices:
        batch_des.append(item[2])
        candidate_set = [int(i) for i in item[3].split(',')]
        # 获取每个候选区域与目标区域的距离
        candidate_dis = []
        for c in candidate_set:
            dis = region_dist[c][item[2]]
            if dis == -1:
                # 不可能被选中的
                dis = 100000
            candidate_dis.append(dis/100)  # 转化为百米
        batch_candidate_set.append(candidate_set)
        batch_candidate_dis.append(candidate_dis)
        batch_target.append(item[5])
        candidate_set_len.append(len(candidate_set))
    # 补齐
    max_candidate_size = max(candidate_set_len)
    for i in range(len(batch_des)):
        # 对于候选集，选择非下一跳的点进行补齐
        while len(batch_candidate_set[i]) < max_candidate_size:
            # 因为我们已经干掉了 candidate_set len 为 1 的点了
            assert len(batch_candidate_set[i]) != 1, 'candidate set is 1!'
            pad_index = np.random.randint(len(batch_candidate_set[i]))
            if pad_index != batch_target[i]:
                batch_candidate_set[i].append(batch_candidate_set[i][pad_index])
                batch_candidate_dis[i].append(batch_candidate_dis[i][pad_index])
    return [torch.LongTensor(batch_des).to(device), torch.LongTensor(batch_candidate_set).to(device),
            torch.FloatTensor(batch_candidate_dis).to(device), torch.LongTensor(batch_target).to(device)]


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)


if train:
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    metrics = []
    for epoch in range(max_epoch):
        # train
        logger.info('start train epoch {}'.format(epoch))
        gat.train(True)
        train_loss = 0
        for des, candidate_set, candidate_distance, target in tqdm(train_loader, desc='train model'):
            optimizer.zero_grad()
            loss = gat.calculate_loss(candidate_set=candidate_set, candidate_distance=candidate_distance, des=des,
                                      target=target)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        # val
        gat.train(False)
        val_hit = 0
        for des, candidate_set, candidate_distance, target in tqdm(val_loader, desc='val model'):
            with torch.no_grad():
                candidate_score = gat.predict(candidate_set=candidate_set, des=des, candidate_distance=candidate_distance)
            target = target.tolist()
            val, index = torch.topk(candidate_score, 1, dim=1)
            for i, p in enumerate(index):
                if target[i] in p:
                    val_hit += 1
        val_ac = val_hit / eval_num
        metrics.append(val_ac)
        lr_scheduler.step(val_ac)
        # store temp model
        torch.save(gat.state_dict(), os.path.join(temp_folder, 'region_gat_{}.pt'.format(epoch)))
        lr = optimizer.param_groups[0]['lr']
        logger.info('==> Train Epoch {}: Train Loss {:.6f}, val ac {}, lr {}'.format(epoch, train_loss, val_ac, lr))
        if lr < early_stop_lr:
            logger.info('early stop')
            break
    # load best epoch
    best_epoch = np.argmin(metrics)
    load_temp_file = 'region_gat_{}.pt'.format(best_epoch)
    logger.info('load best from {}'.format(best_epoch))
    gat.load_state_dict(torch.load(os.path.join(temp_folder, load_temp_file)))
else:
    # gat.load_state_dict(torch.load(os.path.join(save_folder, save_file_name), map_location=device))
    gat.load_state_dict(torch.load(save_path, map_location=device))
# 开始评估
gat.train(False)
test_hit = 0
for des, candidate_set, candidate_distance, target in tqdm(test_loader, desc='test model'):
    with torch.no_grad():
        candidate_score = gat.predict(candidate_set=candidate_set, des=des, candidate_distance=candidate_distance)
    target = target.tolist()
    val, index = torch.topk(candidate_score, 1, dim=1)
    for i, p in enumerate(index):
        if target[i] in p:
            test_hit += 1
test_ac = test_hit / test_num
logger.info('==> Test Result: test ac {}'.format(test_ac))
# 保存模型
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
# torch.save(gat.state_dict(), os.path.join(save_folder, save_file_name))
torch.save(gat.state_dict(), save_path)
# 删除 temp 文件
for rt, dirs, files in os.walk(temp_folder):
    for name in files:
        remove_path = os.path.join(rt, name)
        os.remove(remove_path)
