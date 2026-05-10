from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for creating a combined region-level trajectory file.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Create a combined region-level trajectory CSV from train/eval/test "
            "region trajectory files."
        )
    )

    parser.add_argument("--dataset_name", type=str, default="Xian")
    parser.add_argument("--data_root", type=Path, default=Path("../data"))

    parser.add_argument(
        "--processed_filename",
        type=str,
        default="xianshi_region_traj_mm_processed.csv",
        help="Output combined region-level trajectory CSV.",
    )
    parser.add_argument(
        "--train_filename",
        type=str,
        default="xianshi_mm_region_train.csv",
        help="Region-level training trajectory CSV.",
    )
    parser.add_argument(
        "--eval_filename",
        type=str,
        default="xianshi_mm_region_eval.csv",
        help="Region-level validation trajectory CSV.",
    )
    parser.add_argument(
        "--test_filename",
        type=str,
        default="xianshi_mm_region_test.csv",
        help="Region-level test trajectory CSV.",
    )

    return parser.parse_args()


def ensure_region_processed_traj_file(
    data_dir: Path,
    processed_filename: str = "xianshi_region_traj_mm_processed.csv",
    train_filename: str = "xianshi_mm_region_train.csv",
    eval_filename: str = "xianshi_mm_region_eval.csv",
    test_filename: str = "xianshi_mm_region_test.csv",
) -> Path:
    """
    Ensure a combined region-level trajectory CSV exists.

    This file is for region-level utilities that require `region_list`, such as
    region-level OD route generation. It should not replace the road-level
    `xianshi_partA_traj_mm_processed.csv`, which contains `rid_list`.
    """
    processed_path = data_dir / processed_filename

    if processed_path.exists():
        print(f"[INFO] Using existing region processed file: {processed_path}")
        return processed_path

    frames: list[pd.DataFrame] = []
    for filename in [train_filename, eval_filename, test_filename]:
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing region trajectory file: {path}")
        frames.append(pd.read_csv(path))

    traj_df = pd.concat(frames, ignore_index=True)

    if "region_list" not in traj_df.columns:
        raise ValueError("Expected column 'region_list' not found")

    traj_df.to_csv(processed_path, index=False)

    print(f"[INFO] Saved region processed file: {processed_path}")
    print(f"[INFO] Shape: {traj_df.shape}")

    return processed_path


def main() -> None:
    """
    Create the combined region-level trajectory CSV.
    """
    args = parse_args()
    data_dir = args.data_root / args.dataset_name

    ensure_region_processed_traj_file(
        data_dir=data_dir,
        processed_filename=args.processed_filename,
        train_filename=args.train_filename,
        eval_filename=args.eval_filename,
        test_filename=args.test_filename,
    )


if __name__ == "__main__":
    main()