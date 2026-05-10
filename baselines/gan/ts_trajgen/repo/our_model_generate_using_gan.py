from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import yaml
from tqdm import tqdm

from generator.generator_v4 import GeneratorV4
from search import DoubleLayerSearcher
from utils.data_util import encode_time
from utils.map_manager import MapManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate trajectories using GAN-trained TS-TrajGen full-generator checkpoints."
    )

    parser.add_argument("--dataset_name", type=str, default="Xian")
    parser.add_argument("--data_root", type=Path, default=Path("./data"))
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument(
        "--model_config",
        type=Path,
        required=True,
        help="YAML file containing road_gen_config and region_gen_config.",
    )

    parser.add_argument("--true_traj_file", type=str, default="xianshi_partA_mm_test.csv")
    parser.add_argument("--generated_trace_output_file", type=str, default="TS_TrajGen_GAN_generate.csv")

    parser.add_argument(
        "--road_gan_generator_file",
        type=Path,
        required=True,
        help="GAN-trained full road GeneratorV4 checkpoint.",
    )
    parser.add_argument(
        "--region_gan_generator_file",
        type=Path,
        required=True,
        help="GAN-trained full region GeneratorV4 checkpoint.",
    )

    parser.add_argument("--geo_path", type=Path, required=True)
    parser.add_argument("--map_manager_cache_dir", type=Path, default=Path("./data/Xian"))

    parser.add_argument("--node_feature_file", type=str, default="node_feature.pt")
    parser.add_argument("--adjacent_np_file", type=str, default="adjacent_mx.npz")
    parser.add_argument("--region_adjacent_np_file", type=str, default="region_adj_mx.npz")
    parser.add_argument("--region_feature_file", type=str, default="region_feature.pt")

    parser.add_argument("--region2rid_file", type=str, default="region2rid.json")
    parser.add_argument("--rid2region_file", type=str, default="rid2region.json")
    parser.add_argument("--adjacent_list_file", type=str, default="adjacent_list.json")
    parser.add_argument("--rid_gps_file", type=str, default="rid_gps.json")
    parser.add_argument("--road_length_file", type=str, default="road_length.json")
    parser.add_argument("--region_adjacent_list_file", type=str, default="region_adjacent_list.json")
    parser.add_argument("--region_dist_file", type=str, default="region_count_dist.npy")
    parser.add_argument("--region_transfer_file", type=str, default="region_transfer_prob.json")
    parser.add_argument("--road_time_distribution_file", type=str, default="road_time_distribution.npy")
    parser.add_argument("--region_time_distribution_file", type=str, default="region_time_distribution.npy")

    parser.add_argument("--max_step", type=int, default=5000)

    return parser.parse_args()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_config(path: Path, device: str) -> tuple[dict, dict]:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    road_gen_config = config["road_gen_config"]
    region_gen_config = config["region_gen_config"]

    road_gen_config["function_g"]["device"] = device
    road_gen_config["function_h"]["device"] = device
    region_gen_config["function_g"]["device"] = device
    region_gen_config["function_h"]["device"] = device

    return road_gen_config, region_gen_config


def main() -> None:
    args = parse_args()
    data_dir = args.data_root / args.dataset_name
    device = args.device

    true_traj_path = data_dir / args.true_traj_file
    output_path = data_dir / args.generated_trace_output_file

    true_traj = pd.read_csv(true_traj_path)

    map_manager = MapManager(
        dataset_name=args.dataset_name,
        geo_path=args.geo_path,
        cache_dir=args.map_manager_cache_dir,
    )

    road_gen_config, region_gen_config = load_model_config(args.model_config, device)

    node_features = torch.load(data_dir / args.node_feature_file, map_location=device).to(device)
    adj_mx = sp.load_npz(data_dir / args.adjacent_np_file)

    region_features = torch.load(data_dir / args.region_feature_file, map_location=device).to(device)
    region_adj_mx = sp.load_npz(data_dir / args.region_adjacent_np_file)

    road_num = pd.read_csv(args.geo_path).shape[0]
    time_size = 2880

    data_feature = {
        "road_num": road_num + 1,
        "time_size": time_size + 1,
        "road_pad": road_num,
        "time_pad": time_size,
        "adj_mx": adj_mx,
        "node_features": node_features,
        "img_width": map_manager.img_width,
        "img_height": map_manager.img_height,
    }

    region2rid = load_json(data_dir / args.region2rid_file)
    region_num = len(region2rid)

    region_data_feature = {
        "road_num": region_num + 1,
        "time_size": time_size + 1,
        "road_pad": region_num,
        "time_pad": time_size,
        "adj_mx": region_adj_mx,
        "node_features": region_features,
        "img_width": map_manager.img_width,
        "img_height": map_manager.img_height,
    }

    adjacent_list = load_json(data_dir / args.adjacent_list_file)
    rid_gps = load_json(data_dir / args.rid_gps_file)
    road_length = load_json(data_dir / args.road_length_file)
    region_adjacent_list = load_json(data_dir / args.region_adjacent_list_file)
    region_transfer_freq = load_json(data_dir / args.region_transfer_file)
    rid2region = load_json(data_dir / args.rid2region_file)

    region_dist = np.load(data_dir / args.region_dist_file)
    road_time_distribution = np.load(data_dir / args.road_time_distribution_file)
    region_time_distribution = np.load(data_dir / args.region_time_distribution_file)

    road_generator = GeneratorV4(config=road_gen_config, data_feature=data_feature).to(device)
    road_generator.load_state_dict(torch.load(args.road_gan_generator_file, map_location=device))
    road_generator.eval()

    region_generator = GeneratorV4(config=region_gen_config, data_feature=region_data_feature).to(device)
    region_generator.load_state_dict(torch.load(args.region_gan_generator_file, map_location=device))
    region_generator.eval()

    searcher = DoubleLayerSearcher(
        device=device,
        adjacent_list=adjacent_list,
        road_center_gps=rid_gps,
        road_length=road_length,
        region_adjacent_list=region_adjacent_list,
        region_dist=region_dist,
        region_transfer_freq=region_transfer_freq,
        rid2region=rid2region,
        road_time_distribution=road_time_distribution,
        region_time_distribution=region_time_distribution,
        region2rid=region2rid,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fail_cnt = 0
    region_astar_fail_cnt = 0

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("traj_id,rid_list,time_list\n")

        for _, row in tqdm(true_traj.iterrows(), total=true_traj.shape[0]):
            rid_list = [int(i) for i in row["rid_list"].split(",")]
            time_list = list(map(encode_time, row["time_list"].split(",")))
            traj_id = row["traj_id"]

            with torch.no_grad():
                gen_trace_loc, gen_trace_tim, is_astar = searcher.astar_search(
                    region_model=region_generator,
                    road_model=road_generator,
                    start_rid=rid_list[0],
                    start_tim=time_list[0],
                    des=rid_list[-1],
                    default_len=len(rid_list),
                    max_step=args.max_step,
                )

            f.write(
                f'{traj_id},"{",".join(map(str, gen_trace_loc))}",'
                f'"{",".join(map(str, gen_trace_tim))}"\n'
            )

            if gen_trace_loc[-1] != rid_list[-1]:
                fail_cnt += 1
            if is_astar == 0:
                region_astar_fail_cnt += 1

    print("fail cnt ", fail_cnt)
    print("region astar fail cnt ", region_astar_fail_cnt)
    searcher.save_fail_log()


if __name__ == "__main__":
    main()