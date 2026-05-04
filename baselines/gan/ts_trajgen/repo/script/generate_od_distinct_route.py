from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from collections import defaultdict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build od_distinct_route.json for TS-TrajGen yaw loss."
    )

    parser.add_argument("--dataset_name", type=str, default="Xian")
    parser.add_argument("--data_root", type=Path, default=Path("./data"))

    parser.add_argument(
        "--traj_filename",
        type=str,
        default="xianshi_partA_traj_mm_processed.csv",
        help="Trajectory CSV containing rid_list.",
    )
    parser.add_argument(
        "--rid_gps_filename",
        type=str,
        default="rid_gps.json",
        help="Road id to GPS coordinate mapping.",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="od_distinct_route.json",
        help="Output OD-to-historical-routes JSON.",
    )

    return parser.parse_args()


def parse_rid_list(value: str) -> list[str]:
    """
    Parse a comma-separated rid_list and remove invalid padding tokens.
    """
    return [x.strip() for x in str(value).split(",") if x.strip() and x.strip() != "000"]


def build_od_distinct_route(
    traj_path: Path,
    rid_gps_path: Path,
    output_path: Path,
) -> None:
    """
    Build OD (origin-destination): distinct historical route mapping used for
    yaw loss computation in TS-TrajGen.

    This file is required by `Rollout.yaw_loss()` during GAN training, where
    generated trajectories are compared against historical trajectories with the
    same OD pair using DTW (Dynamic Time Warping).

    Parameters
    ----------
    traj_path : Path
        Path to processed trajectory CSV (must contain 'rid_list').

    rid_gps_path : Path
        Path to JSON mapping road ID -> [lon, lat].

    output_path : Path
        Output path for od_distinct_route.json.

    Returns
    ----------
    None: Writes JSON file to disk.

    Output format:
    {
        "originRid-destinationRid": [
            [[lat, lon], [lat, lon], ...], # one historical trajectory
            ...
        ]
    }

    Each OD key maps to a list of historical trajectories represented as GPS
    coordinate sequences.

    Important Coordinate Convention
    --------------------------------
    - The input `rid_gps.json` follows GeoJSON / LibCity convention:
        coordinates = [longitude, latitude]

    - However, the DTW + haversine implementation used in TS-TrajGen expects:
        [latitude, longitude]

    Specifically, the haversine function assumes:
        array = [lat, lon]

    Therefore, we MUST convert:
        [lon, lat] → [lat, lon]
    
    -   rollout.py's Roll.yaw_loss(...) appends generated GPS as [lat, lon], so we save
    historical routes in the same coordinate order.

    before saving routes, otherwise distance calculations (yaw loss) will be incorrect.

    This function aggregates all trajectories sharing the same OD pair.

    Notes
    -----
    - This replaces the missing `od_distinct_route.json` file in the original repo.
    - Using combined train+test trajectories provides better statistical coverage
      but may introduce slight evaluation leakage. Document this choice if used.

    """
    traj_df = pd.read_csv(traj_path)

    with open(rid_gps_path, "r", encoding="utf-8") as f:
        rid_gps = json.load(f)

    od_route: dict[str, list[list[list[float]]]] = defaultdict(list)

    skipped_short = 0
    skipped_missing_gps = 0

    for _, row in tqdm(traj_df.iterrows(), total=traj_df.shape[0], desc="build OD routes"):
        rid_list = parse_rid_list(row["rid_list"])

        if len(rid_list) < 2:
            skipped_short += 1
            continue

        missing = [rid for rid in rid_list if rid not in rid_gps]
        if missing:
            skipped_missing_gps += 1
            continue

        origin = rid_list[0]
        destination = rid_list[-1]
        od_key = f"{origin}-{destination}"

        # rid_gps is usually [lon, lat], while rollout.py compares [lat, lon].
        gps_route = [[rid_gps[rid][1], rid_gps[rid][0]] for rid in rid_list]

        # od_route.setdefault(od_key, []).append(gps_route)
        od_route[od_key].append(gps_route)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(od_route, f)

    print(f"Saved: {output_path}")
    print(f"OD pairs: {len(od_route):,}")
    print(f"Skipped short: {skipped_short:,}")
    print(f"Skipped missing GPS: {skipped_missing_gps:,}")


def main() -> None:
    args = parse_args()
    data_dir = args.data_root / args.dataset_name

    build_od_distinct_route(
        traj_path=data_dir / args.traj_filename,
        rid_gps_path=data_dir / args.rid_gps_filename,
        output_path=data_dir / args.output_filename,
    )


if __name__ == "__main__":
    main()