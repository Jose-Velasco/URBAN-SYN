from __future__ import annotations

import argparse
import ast
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from geopy import distance
from shapely.geometry import LineString
from tqdm import tqdm
from collections import defaultdict


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


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Convert matched trajectory CSVs into TS-TrajGen pretrain inputs."
    )

    parser.add_argument(
        "--local",
        type=str2bool,
        default=True,
        help="Use local relative dataset paths instead of original hardcoded archive paths.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="BJ_Taxi",
        help="Dataset name. BJ_Taxi and Porto_Taxi keep original handling; others use generic paths.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="../data/",
        help="Root directory containing dataset subfolders for local/custom datasets.",
    )
    parser.add_argument(
        "--dataset_prefix",
        type=str,
        default=None,
        help="Prefix for generic dataset files, e.g. 'nyc' -> nyc_mm_train.csv, nyc.geo, nyc.rel.",
    )
    parser.add_argument(
        "--max_step",
        type=int,
        default=4,
        help="Maximum random step size when random encoding is enabled.",
    )
    parser.add_argument(
        "--random_encode",
        type=str2bool,
        default=True,
        help="Randomly skip steps during encoding to reduce data volume and overfitting.",
    )
    parser.add_argument(
        "--train_rate",
        type=float,
        default=0.9,
        help="Fraction of train_data rows written to train output; the rest go to eval.",
    )

    return parser.parse_args()


def resolve_dataset_paths(args: argparse.Namespace) -> dict[str, Path | None]:
    """
    Resolve input and output paths for the selected dataset.

    Built-in datasets keep the original script's behavior. Generic datasets
    expect files like:
        <prefix>_mm_train.csv
        <prefix>_mm_test.csv
        <prefix>.geo
        <prefix>.rel

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    dict[str, Path | None]
        Resolved dataset paths and metadata.
    """
    dataset_name = args.dataset_name

    if args.local:
        data_root = Path(args.data_root)
    else:
        data_root = Path("/mnt/data/jwj/TS_TrajGen_data_archive/")

    if dataset_name == "BJ_Taxi":
        archive_root = Path("/mnt/data/jwj/TS_TrajGen_data_archive/")
        return {
            "dataset_dir": None,
            "train_data_path": Path("/mnt/data/jwj/BJ_Taxi/chaoyang_traj_mm_train.csv"),
            "test_data_path": Path("/mnt/data/jwj/BJ_Taxi/chaoyang_traj_mm_test.csv"),
            "adjacent_file": archive_root / "adjacent_list.json",
            "rid_gps_file": archive_root / "rid_gps.json",
            "geo_file": None,
            "rel_file": None,
            "train_output": archive_root / "bj_taxi_pretrain_input_train.csv",
            "eval_output": archive_root / "bj_taxi_pretrain_input_eval.csv",
            "test_output": archive_root / "bj_taxi_pretrain_input_test.csv",
        }

    if dataset_name == "Porto_Taxi":
        archive_root = Path("/mnt/data/jwj/TS_TrajGen_data_archive/")
        return {
            "dataset_dir": None,
            "train_data_path": Path("/mnt/data/jwj/Porto_Taxi/porto_mm_train.csv"),
            "test_data_path": Path("/mnt/data/jwj/Porto_Taxi/porto_mm_test.csv"),
            "adjacent_file": archive_root / "porto_adjacent_list.json",
            "rid_gps_file": archive_root / "porto_rid_gps.json",
            "geo_file": None,
            "rel_file": None,
            "train_output": archive_root / "porto_taxi_pretrain_input_train.csv",
            "eval_output": archive_root / "porto_taxi_pretrain_input_eval.csv",
            "test_output": archive_root / "porto_taxi_pretrain_input_test.csv",
        }

    # Generic/custom dataset branch, e.g. NYC.
    dataset_dir = data_root / dataset_name
    prefix = args.dataset_prefix or dataset_name.lower()

    return {
        "dataset_dir": dataset_dir,
        "train_data_path": dataset_dir / f"{prefix}_mm_train.csv",
        "test_data_path": dataset_dir / f"{prefix}_mm_test.csv",
        "adjacent_file": dataset_dir / "adjacent_list.json",
        "rid_gps_file": dataset_dir / "rid_gps.json",
        "geo_file": dataset_dir / f"{prefix}.geo",
        "rel_file": dataset_dir / f"{prefix}.rel",
        "train_output": dataset_dir / f"{prefix}_pretrain_input_train.csv",
        "eval_output": dataset_dir / f"{prefix}_pretrain_input_eval.csv",
        "test_output": dataset_dir / f"{prefix}_pretrain_input_test.csv",
    }


def load_train_test_data(paths: dict[str, Path | None]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and test trajectory CSVs.

    Parameters
    ----------
    paths : dict[str, Path | None]
        Resolved dataset paths.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Train and test trajectory dataframes.
    """
    train_data_path = paths["train_data_path"]
    test_data_path = paths["test_data_path"]
    if not (train_data_path and test_data_path):
        raise ValueError(f"Require both train_data_path and test_data_path but got: {train_data_path = }, {test_data_path = }")

    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    return train_data, test_data


def load_adjacent_list(paths: dict[str, Path | None]) -> dict[str, list[int]]:
    """
    Load or build the road adjacency list.

    For generic datasets, if `adjacent_list.json` does not exist, it is built
    from `<prefix>.rel`.

    Parameters
    ----------
    paths : dict[str, Path | None]
        Resolved dataset paths.

    Returns
    -------
    dict[str, list[int]]
        Mapping from road id string to list of reachable next road ids.
    """
    adjacent_file = paths["adjacent_file"]
    if adjacent_file is None:
        raise ValueError(f"Required adjacent_file but got: {adjacent_file = }")

    if adjacent_file.exists():
        with open(adjacent_file, "r", encoding="utf-8") as f:
            return json.load(f)

    rel_file = paths["rel_file"]
    if rel_file is None:
        raise FileNotFoundError(
            f"Missing adjacency file and no .rel fallback is available: {adjacent_file}"
        )

    rid_rel = pd.read_csv(rel_file)
    # road_adjacent_list: dict[str, list[int]] = {}
    road_adjacent_list: dict[str, list[int]] = defaultdict(list)

    for _, row in tqdm(rid_rel.iterrows(), total=rid_rel.shape[0], desc="build road adjacent list"):
        from_rid = str(row["origin_id"])
        to_rid = int(row["destination_id"])

        road_adjacent_list[from_rid].append(to_rid)
        # if from_rid not in road_adjacent_list:
        #     road_adjacent_list[from_rid] = [to_rid]
        # else:
        #     road_adjacent_list[from_rid].append(to_rid)

    with open(adjacent_file, "w", encoding="utf-8") as f:
        json.dump(road_adjacent_list, f)

    return dict(road_adjacent_list)


def load_rid_gps(paths: dict[str, Path | None]) -> dict[str, tuple[float, float]]:
    """
    Load or build the road centroid lookup.

    For generic datasets, if `rid_gps.json` does not exist, it is built from
    `<prefix>.geo` by taking each road geometry centroid.

    Parameters
    ----------
    paths : dict[str, Path | None]
        Resolved dataset paths.

    Returns
    -------
    dict[str, tuple[float, float]]
        Mapping from road id string to (lon, lat).
    """
    rid_gps_file = paths["rid_gps_file"]

    if rid_gps_file is None:
        raise ValueError(f"Required rid_gps_file but got: {rid_gps_file = }")

    if rid_gps_file.exists():
        with open(rid_gps_file, "r", encoding="utf-8") as f:
            return json.load(f)

    geo_file = paths["geo_file"]
    if geo_file is None:
        raise FileNotFoundError(
            f"Missing rid_gps file and no .geo fallback is available: {rid_gps_file}"
        )

    rid_info = pd.read_csv(geo_file)
    rid_gps: dict[str, tuple[float, float]] = {}

    for _, row in tqdm(rid_info.iterrows(), total=rid_info.shape[0], desc="build road gps dict"):
        rid = row["geo_id"]
        coordinate = ast.literal_eval(row["coordinates"])
        road_line = LineString(coordinates=coordinate)
        center_coord = road_line.centroid
        center_lon, center_lat = center_coord.x, center_coord.y
        rid_gps[str(rid)] = (center_lon, center_lat)

    with open(rid_gps_file, "w", encoding="utf-8") as f:
        json.dump(rid_gps, f)

    return rid_gps

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

# def encode_time(timestamp: str) -> int:
#     """
#     Encode a timestamp into the model's minute-level time slot.

#     Weekdays use [0, 1439], weekends use [1440, 2879], matching the
#     paper/code convention.

#     Parameters
#     ----------
#     timestamp : str
#         Timestamp in '%Y-%m-%dT%H:%M:%SZ' format.

#     Returns
#     -------
#     int
#         Encoded time slot.
#     """
#     time = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
#     if time.weekday() in (5, 6):
#         return time.hour * 60 + time.minute + 1440
#     return time.hour * 60 + time.minute


def parse_rid_list(value: str) -> list[int]:
    """
    Parse a road-id sequence from supported CSV formats.

    Supported examples
    ------------------
    - '1,2,3'
    - '[1, 2, 3]'

    Parameters
    ----------
    value : str
        Serialized road-id sequence.

    Returns
    -------
    list[int]
        Parsed road-id list.
    """
    if pd.isna(value):
        return []

    s = str(value).strip()
    if not s:
        return []

    if s.startswith("[") and s.endswith("]"):
        parsed = ast.literal_eval(s)
        return [int(i) for i in parsed]

    return [int(i.strip()) for i in s.split(",") if i.strip()]


def parse_time_list(value: str) -> list[int]:
    """
    Parse and encode a timestamp sequence from the trajectory CSV.

    Parameters
    ----------
    value : str
        Comma-separated ISO timestamps.

    Returns
    -------
    list[int]
        Encoded time slots.
    """
    if pd.isna(value):
        return []

    raw = str(value).strip()
    if not raw:
        return []

    return [encode_time(ts.strip()) for ts in raw.split(",") if ts.strip()]


def encode_trace(
    trace: pd.Series,
    fp,
    adjacent_list: dict[str, list[int]],
    rid_gps: dict[str, tuple[float, float]],
    random_encode: bool,
    max_step: int,
    stats: dict[str, int],
) -> None:
    """
    Encode one trajectory into TS-TrajGen pretraining supervision format.

    Parameters
    ----------
    trace : pd.Series
        One trajectory record containing rid_list and time_list.
    fp : TextIO
        Output file handle.
    adjacent_list : dict[str, list[int]]
        Road adjacency lookup.
    rid_gps : dict[str, tuple[float, float]]
        Road centroid lookup.
    random_encode : bool
        Whether to randomly skip steps during encoding.
    max_step : int
        Maximum random step size when random encoding is enabled.
    """
    rid_list = parse_rid_list(trace["rid_list"])
    time_list = parse_time_list(trace["time_list"])

    if len(rid_list) < 2 or len(time_list) < 2:
        print(f"Skipping current trace encode rid_list or time_list less than one: {len(rid_list) = }, {len(time_list)}")
        return

    # Keep sequences aligned if malformed rows sneak in.
    if len(rid_list) != len(time_list):
        print(f"Misaligned rid_list or time_list performing truncation: {len(rid_list) = }, {len(time_list) = }")
        min_len = min(len(rid_list), len(time_list))
        rid_list = rid_list[:min_len]
        time_list = time_list[:min_len]

    # if len(rid_list) < 2:
    #     return

    des = rid_list[-1]
    des_gps = rid_gps[str(des)]

    # NOTE: comments translated from original repo code (provided for context)
    """
    The training data still seems a bit too much.
    To avoid overfitting, let's use randomized encoding steps. 
    We can do a comparative experiment to see which one performs better.
    """
    if not random_encode:
        i = 1
        step_fn = lambda cur_i: cur_i + 1
    else:
        i = 1
        step_fn = lambda cur_i: cur_i + np.random.randint(1, max_step)

    while i < len(rid_list):
        cur_loc = rid_list[:i]
        cur_time = time_list[:i]
        cur_rid = cur_loc[-1]

        # If connectivity breaks, the rest of the supervision example is unreliable.
        if str(cur_rid) not in adjacent_list or rid_list[i] not in adjacent_list[str(cur_rid)]:
            # print("A path break has occurred, discarding the subsequent paths.")
            
            # Same-edge repeats can happen in our NYC data when multiple GPS anchors map
            # to the same road segment at different timestamps.
            if rid_list[i] == cur_rid:
                stats["same_edge_repeats"] += 1
                # NOTE:
                # to try to proceed but this changes original authors implementation and most likely results
                # for now align. 
                # NOTE:
                # uncomment and not use return below instead to NOT throw away the entire remaining trajectory after the first break.
                # this will change behavior from discard whole trace to skip that broken transition and continue
                # i = step_fn(i)
                # continue
                return

            # A true path break means the next road is not listed as reachable from
            # the current road in the .rel adjacency graph.
            stats["path_breaks"] += 1

            # NOTE:
            # uncomment below and use instead of return
            # i = step_fn(i)
            # continue
            return

        candidate_set = adjacent_list[str(cur_rid)]

        # Only useful when there is a real branching choice.
        if len(candidate_set) > 1:
            """Only points with multiple candidate points are worth learning from. (translated original comment)"""
            target = rid_list[i]
            target_index = 0
            candidate_dis = []

            for index, candidate in enumerate(candidate_set):
                if candidate == target:
                    target_index = index

                candidate_gps = rid_gps[str(candidate)]
                dis = distance.distance(
                    (des_gps[1], des_gps[0]),
                    (candidate_gps[1], candidate_gps[0]),
                ).kilometers * 10  # Unit: 100 meters
                candidate_dis.append(dis)

            cur_loc_str = ",".join(str(x) for x in cur_loc)
            cur_time_str = ",".join(str(x) for x in cur_time)
            candidate_set_str = ",".join(str(x) for x in candidate_set)
            candidate_dis_str = ",".join(str(x) for x in candidate_dis)

            fp.write(
                f"\"{cur_loc_str}\",\"{cur_time_str}\",{des},\"{candidate_set_str}\",\"{candidate_dis_str}\",{target_index}\n"
            )
            stats["encoded_examples"] += 1

        i = step_fn(i)


def main() -> None:
    """
    Main entry point for building TS-TrajGen pretraining CSVs.
    """
    args = parse_args()
    np.random.seed(101)

    paths = resolve_dataset_paths(args)

    train_data, test_data = load_train_test_data(paths)
    adjacent_list = load_adjacent_list(paths)
    rid_gps = load_rid_gps(paths)

    total_data_num = train_data.shape[0]
    train_num = int(total_data_num * args.train_rate)

    train_output = paths["train_output"]
    if train_output is None:
        raise ValueError(f"Required train_output but got: {train_output = }")

    # paths["train_output"].parent.mkdir(parents=True, exist_ok=True)
    train_output.parent.mkdir(parents=True, exist_ok=True)

    if not (paths["train_output"] and paths["eval_output"] and paths["test_output"]):
        raise ValueError(f"Require the following but got: {paths['train_output'] = }, {paths['eval_output'] = }, {paths['test_output'] = }")

    stats = {
        "path_breaks": 0,
        "same_edge_repeats": 0,
        "encoded_examples": 0,
    }
    with open(paths["train_output"], "w", encoding="utf-8") as train_output, \
         open(paths["eval_output"], "w", encoding="utf-8") as eval_output, \
         open(paths["test_output"], "w", encoding="utf-8") as test_output:

        header = "trace_loc,trace_time,des,candidate_set,candidate_dis,target\n"
        train_output.write(header)
        eval_output.write(header)
        test_output.write(header)

        for index, row in tqdm(train_data.iterrows(), total=train_data.shape[0], desc="encode train traj"):
            if index < train_num:
                encode_trace(
                    trace=row,
                    fp=train_output,
                    adjacent_list=adjacent_list,
                    rid_gps=rid_gps,
                    random_encode=args.random_encode,
                    max_step=args.max_step,
                    stats=stats,
                )
            else:
                encode_trace(
                    trace=row,
                    fp=eval_output,
                    adjacent_list=adjacent_list,
                    rid_gps=rid_gps,
                    random_encode=args.random_encode,
                    max_step=args.max_step,
                    stats=stats,
                )

        for _, row in tqdm(test_data.iterrows(), total=test_data.shape[0], desc="encode test traj"):
            encode_trace(
                trace=row,
                fp=test_output,
                adjacent_list=adjacent_list,
                rid_gps=rid_gps,
                random_encode=args.random_encode,
                max_step=args.max_step,
                stats=stats,
            )

    print("Done.")
    print("=== Pretrain Encoding Stats ===")
    print(f"Encoded examples:    {stats['encoded_examples']:,}")
    print(f"Path breaks skipped: {stats['path_breaks']:,}")
    print(f"Same-edge repeats:   {stats['same_edge_repeats']:,}")
    
    print(f"Train input: {paths['train_output']}")
    print(f"Eval input:  {paths['eval_output']}")
    print(f"Test input:  {paths['test_output']}")


if __name__ == "__main__":
    main()



# import pandas as pd
# from tqdm import tqdm
# import json
# from datetime import datetime
# from geopy import distance
# from shapely.geometry import LineString
# import numpy as np
# import argparse
# import os


# def str2bool(s):
#     if isinstance(s, bool):
#         return s
#     if s.lower() in ('yes', 'true'):
#         return True
#     elif s.lower() in ('no', 'false'):
#         return False
#     else:
#         raise argparse.ArgumentTypeError('bool value expected.')


# parser = argparse.ArgumentParser()
# parser.add_argument('--local', type=str2bool,
#                     default=True, help='whether save the trained model')
# parser.add_argument('--dataset_name', type=str,
#                     default='BJ_Taxi')

# args = parser.parse_args()
# local = args.local
# dataset_name = args.dataset_name
# max_step = 4
# random_encode = True  # 随机步数 encode，主要是减少数据量，避免过拟合

# if local:
#     data_root = '../data/'
# else:
#     data_root = '/mnt/data/jwj/TS_TrajGen_data_archive/'


# if dataset_name == 'BJ_Taxi':
#     train_data = pd.read_csv('/mnt/data/jwj/BJ_Taxi/chaoyang_traj_mm_train.csv')
#     test_data = pd.read_csv('/mnt/data/jwj/BJ_Taxi/chaoyang_traj_mm_test.csv')
# elif dataset_name == 'Porto_Taxi':
#     train_data = pd.read_csv('/mnt/data/jwj/Porto_Taxi/porto_mm_train.csv')
#     test_data = pd.read_csv('/mnt/data/jwj/Porto_Taxi/porto_mm_test.csv')
# else:
#     # Xian
#     train_data = pd.read_csv(os.path.join(data_root, dataset_name, 'xianshi_partA_mm_train.csv'))
#     test_data = pd.read_csv(os.path.join(data_root, dataset_name, 'xianshi_partA_mm_test.csv'))


# # 读取路网邻接表
# if dataset_name == 'BJ_Taxi':
#     with open('/mnt/data/jwj/TS_TrajGen_data_archive/adjacent_list.json', 'r') as f:
#         adjacent_list = json.load(f)
# elif dataset_name == 'Porto_Taxi':
#     with open('/mnt/data/jwj/TS_TrajGen_data_archive/porto_adjacent_list.json', 'r') as f:
#         adjacent_list = json.load(f)
# else:
#     adjacent_file = os.path.join(data_root, dataset_name, 'adjacent_list.json')
#     if os.path.exists(adjacent_file):
#         with open(adjacent_file, 'r') as f:
#             adjacent_list = json.load(f)
#     else:
#         rid_rel = pd.read_csv(os.path.join(data_root, dataset_name, 'xian.rel'))
#         road_adjacent_list = {}
#         for index, row in tqdm(rid_rel.iterrows(), total=rid_rel.shape[0], desc='cal road adjacent list'):
#             f_rid = str(row['origin_id'])
#             t_rid = row['destination_id']
#             if f_rid not in road_adjacent_list:
#                 road_adjacent_list[f_rid] = [t_rid]
#             else:
#                 road_adjacent_list[f_rid].append(t_rid)
#         with open(adjacent_file, 'w') as f:
#             json.dump(road_adjacent_list, f)

# # 读取路网信息表
# if dataset_name == 'BJ_Taxi':
#     with open('/mnt/data/jwj/TS_TrajGen_data_archive/rid_gps.json', 'r') as f:
#         rid_gps = json.load(f)
# elif dataset_name == 'Porto_Taxi':
#     with open('/mnt/data/jwj/TS_TrajGen_data_archive/porto_rid_gps.json', 'r') as f:
#         rid_gps = json.load(f)
# else:
#     # Xian
#     rid_gps_file = os.path.join(data_root, dataset_name, 'rid_gps.json')
#     if os.path.exists(rid_gps_file):
#         with open(rid_gps_file, 'r') as f:
#             rid_gps = json.load(f)
#     else:
#         rid_gps = {}
#         rid_info = pd.read_csv(os.path.join(data_root, dataset_name, 'xian.geo'))
#         for index, row in tqdm(rid_info.iterrows(), total=rid_info.shape[0], desc='cal road gps dict'):
#             rid = row['geo_id']
#             coordinate = eval(row['coordinates'])
#             road_line = LineString(coordinates=coordinate)
#             center_coord = road_line.centroid
#             center_lon, center_lat = center_coord.x, center_coord.y
#             rid_gps[str(rid)] = (center_lon, center_lat)
#         with open(rid_gps_file, 'w') as f:
#             json.dump(rid_gps, f)


# def encode_time(timestamp):
#     """
#     编码时间
#     """
#     # 按一分钟编码，周末与工作日区分开来
#     time = datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%SZ')
#     if time.weekday() == 5 or time.weekday() == 6:
#         return time.hour * 60 + time.minute + 1440
#     else:
#         return time.hour * 60 + time.minute


# def encode_trace(trace, fp):
#     """
#     编码轨迹

#     Args:
#         trace: 一条轨迹记录
#         fp: 写入编码结果的文件
#     """
#     rid_list = [int(i) for i in trace['rid_list'].split(',')]
#     time_list = [encode_time(i) for i in trace['time_list'].split(',')]
#     des = rid_list[-1]
#     des_gps = rid_gps[str(des)]
#     # 训练数据还是感觉有点多
#     # 这里为了避免过拟合，还是随机步数 encode 吧
#     # 可以做个对比实验看哪个效果好一点
#     if not random_encode:
#         for i in range(1, len(rid_list)):
#             cur_loc = rid_list[:i]
#             cur_time = time_list[:i]
#             cur_rid = cur_loc[-1]
#             if str(cur_rid) not in adjacent_list or rid_list[i] not in adjacent_list[str(cur_rid)]:
#                 # 这不应该发生，如果发生了则舍弃掉后面的路径
#                 # 说明发生了断路
#                 return
#             candidate_set = adjacent_list[str(cur_rid)]
#             if len(candidate_set) > 1:
#                 # 对于有多个候选点的才有学习的价值
#                 target = rid_list[i]
#                 target_index = 0
#                 candidate_dis = []
#                 for index, c in enumerate(candidate_set):
#                     if c == target:
#                         target_index = index
#                     c_gps = rid_gps[str(c)]
#                     dis = distance.distance((des_gps[1], des_gps[0]), (c_gps[1], c_gps[0])).kilometers * 10 # 单位为百米
#                     candidate_dis.append(dis)
#                 # 开始写入编码结果
#                 cur_loc_str = ",".join([str(i) for i in cur_loc])
#                 cur_time_str = ",".join([str(i) for i in cur_time])
#                 candidate_set_str = ",".join([str(i) for i in candidate_set])
#                 candidate_dis_str = ",".join([str(i) for i in candidate_dis])
#                 fp.write("\"{}\",\"{}\",{},\"{}\",\"{}\",{}\n".format(cur_loc_str, cur_time_str, des, candidate_set_str, candidate_dis_str, target_index))
#     else:
#         i = 1
#         while i < len(rid_list):
#             cur_loc = rid_list[:i]
#             cur_time = time_list[:i]
#             cur_rid = cur_loc[-1]
#             if str(cur_rid) not in adjacent_list or rid_list[i] not in adjacent_list[str(cur_rid)]:
#                 # 这不应该发生，如果发生了则舍弃掉后面的路径
#                 # 说明发生了断路
#                 return
#             candidate_set = adjacent_list[str(cur_rid)]
#             if len(candidate_set) > 1:
#                 # 对于有多个候选点的才有学习的价值
#                 target = rid_list[i]
#                 target_index = 0
#                 candidate_dis = []
#                 for index, c in enumerate(candidate_set):
#                     if c == target:
#                         target_index = index
#                     c_gps = rid_gps[str(c)]
#                     dis = distance.distance((des_gps[1], des_gps[0]), (c_gps[1], c_gps[0])).kilometers * 10 # 单位为百米
#                     candidate_dis.append(dis)
#                 # 开始写入编码结果
#                 cur_loc_str = ",".join([str(i) for i in cur_loc])
#                 cur_time_str = ",".join([str(i) for i in cur_time])
#                 candidate_set_str = ",".join([str(i) for i in candidate_set])
#                 candidate_dis_str = ",".join([str(i) for i in candidate_dis])
#                 fp.write("\"{}\",\"{}\",{},\"{}\",\"{}\",{}\n".format(cur_loc_str, cur_time_str, des, candidate_set_str, candidate_dis_str, target_index))
#             # i 不再是 ++ 而是随机加一定步数
#             step = np.random.randint(1, max_step)
#             i += step


# if __name__ == '__main__':
#     train_rate = 0.9
#     total_data_num = train_data.shape[0]
#     train_num = int(total_data_num * train_rate)
#     if dataset_name == 'BJ_Taxi':
#         train_output = open('/mnt/data/jwj/TS_TrajGen_data_archive/{}.csv'.format('bj_taxi_pretrain_input_train'), 'w')
#         eval_output = open('/mnt/data/jwj/TS_TrajGen_data_archive/{}.csv'.format('bj_taxi_pretrain_input_eval'), 'w')
#         test_output = open('/mnt/data/jwj/TS_TrajGen_data_archive/{}.csv'.format('bj_taxi_pretrain_input_test'), 'w')
#     elif dataset_name == 'Porto_Taxi':
#         train_output = open('/mnt/data/jwj/TS_TrajGen_data_archive/{}.csv'.format('porto_taxi_pretrain_input_train'), 'w')
#         eval_output = open('/mnt/data/jwj/TS_TrajGen_data_archive/{}.csv'.format('porto_taxi_pretrain_input_eval'), 'w')
#         test_output = open('/mnt/data/jwj/TS_TrajGen_data_archive/{}.csv'.format('porto_taxi_pretrain_input_test'), 'w')
#     else:
#         # Xian
#         train_output = open(os.path.join(data_root, dataset_name, 'xianshi_partA_pretrain_input_train.csv'), 'w')
#         eval_output = open(os.path.join(data_root, dataset_name, 'xianshi_partA_pretrain_input_eval.csv'), 'w')
#         test_output = open(os.path.join(data_root, dataset_name, 'xianshi_partA_pretrain_input_test.csv'), 'w')
#     train_output.write('trace_loc,trace_time,des,candidate_set,candidate_dis,target\n')
#     eval_output.write('trace_loc,trace_time,des,candidate_set,candidate_dis,target\n')
#     test_output.write('trace_loc,trace_time,des,candidate_set,candidate_dis,target\n')
#     for index, row in tqdm(train_data.iterrows(), total=train_data.shape[0], desc='encode train traj'):
#         if index <= train_num:
#             encode_trace(row, train_output)
#         else:
#             encode_trace(row, eval_output)
#     for index, row in tqdm(test_data.iterrows(), total=test_data.shape[0], desc='encode test traj'):
#         encode_trace(row, test_output)
#     train_output.close()
#     eval_output.close()
#     test_output.close()
