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
import argparse
from utils.refactor_utils import load_config


parser = argparse.ArgumentParser(
    description=(
        "Pretrain region-level GAT (Function H) using region adjacency, "
        "region features, and trajectory pretraining data (TS-TrajGen)."
    )
)

# ---- dataset ----
parser.add_argument(
    "--dataset_name",
    type=str,
    required=True,
    help="Dataset folder name under --data_root (e.g., nyc).",
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

parser.add_argument(
    "--config",
    type=Path,
    required=True,
    help="Path to TS-TrajGen YAML experiment configuration.",
)

parser.add_argument(
    "--temp_dir",
    type=Path,
    required=True,
    help="Directory used for temporary epoch checkpoints.",
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
dataset_name: str = args.dataset_name
device: str = args.device

data_dir: Path = args.data_root / args.dataset_name
save_dir: Path = args.save_dir

temp_dir: Path = args.temp_dir

save_dir.mkdir(parents=True, exist_ok=True)
temp_dir.mkdir(parents=True, exist_ok=True)

region2rid_path: Path = data_dir / args.region2rid_filename
adjacent_np_path: Path = data_dir / args.adjacent_np_filename
node_feature_path: Path = data_dir / args.node_feature_filename
region_dist_path: Path = data_dir / args.region_dist_filename
train_path: Path = data_dir / args.train_filename
eval_path: Path = data_dir / args.eval_filename
test_path: Path = data_dir / args.test_filename
save_path: Path = save_dir / args.save_file_name

experiment_config = load_config(args.config)
model_config = experiment_config["region"]["generator"]["function_h"].copy()
model_config["device"] = device

train_config = experiment_config["training"]["region_function_h"]
optimizer_config = train_config["optimizer"]
scheduler_config = train_config["scheduler"]

max_epoch = train_config["max_epoch"]
batch_size = train_config["batch_size"]
learning_rate = optimizer_config["learning_rate"]
weight_decay = optimizer_config["weight_decay"]
lr_patience = scheduler_config["lr_patience"]
lr_decay_ratio = scheduler_config["lr_decay_ratio"]
early_stop_lr = scheduler_config["early_stop_lr"]

# save_folder = './save/{}'.format(dataset_name)
# save_file_name = 'region_gat_fc.pt'
# temp_folder = './temp/{}/gat/'.format(dataset_name)
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
gat = DistanceGatFC(config=model_config, data_feature=data_feature).to(device)
logger.info('init gat')
logger.info(gat)
optimizer = torch.optim.Adam(gat.parameters(), lr=learning_rate, weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=lr_patience, factor=lr_decay_ratio)
# 加载训练数据
# 读取训练输入数据

train_data = pd.read_csv(train_path)
eval_data = pd.read_csv(eval_path)
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
        # torch.save(gat.state_dict(), os.path.join(temp_folder, 'region_gat_{}.pt'.format(epoch)))
        temp_path = temp_dir / f"region_gat_{epoch}.pt"
        torch.save(gat.state_dict(), temp_path)
        lr = optimizer.param_groups[0]['lr']
        logger.info('==> Train Epoch {}: Train Loss {:.6f}, val ac {}, lr {}'.format(epoch, train_loss, val_ac, lr))
        if lr < early_stop_lr:
            logger.info('early stop')
            break
    # load best epoch
    '''
    original: best_epoch = np.argmin(metrics)
    BUG selects the worst val_acc but scheduler uses mode="max"
    '''
    best_epoch = np.argmax(metrics)
    # load_temp_file = 'region_gat_{}.pt'.format(best_epoch)
    logger.info('load best from {}'.format(best_epoch))
    # gat.load_state_dict(torch.load(os.path.join(temp_folder, load_temp_file)))
    temp_path = temp_dir / f"region_gat_{best_epoch}.pt"
    gat.load_state_dict(torch.load(temp_path, map_location=device))
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
# torch.save(gat.state_dict(), os.path.join(save_folder, save_file_name))
torch.save(gat.state_dict(), save_path)
# 删除 temp 文件
for rt, dirs, files in os.walk(temp_dir):
    for name in files:
        remove_path = os.path.join(rt, name)
        os.remove(remove_path)
