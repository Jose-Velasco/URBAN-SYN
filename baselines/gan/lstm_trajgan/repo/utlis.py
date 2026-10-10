import argparse
import json
from pathlib import Path

import pandas as pd


def train_file_parse_args():
    """
    Parse command-line arguments for LSTM-TrajGAN training.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Train the LSTM-TrajGAN model using preprocessed "
            "trajectory CSV and NPY files."
        )
    )

    parser.add_argument(
        "--train_csv",
        type=Path,
        default=Path("data/train_latlon.csv"),
        help=(
            "Path to the training CSV containing semantic "
            "trajectory data."
        ),
    )

    parser.add_argument(
        "--test_csv",
        type=Path,
        default=Path("data/test_latlon.csv"),
        help=(
            "Path to the test CSV containing semantic "
            "trajectory data."
        ),
    )

    parser.add_argument(
        "--train_npy",
        type=Path,
        default=Path("data/final_train.npy"),
        help=(
            "Path to the encoded training NPY generated "
            "by csv2npy.py."
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("training_params"),
        help=(
            "Directory where model checkpoints, generated "
            "trajectories, and logs will be saved."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2000,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Training batch size.",
    )


    parser.add_argument(
        "--save_params_rate",
        type=int,
        default=100,
        help="The parameter saving interval",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help=(
            "Maximum trajectory sequence length. "
            "If omitted, it is derived from the train and test CSV files."
        ),
    )

    return parser.parse_args()

def predict_file_parse_args():
    """
    Parse command-line arguments for LSTM-TrajGAN prediction.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic trajectories using a trained "
            "LSTM-TrajGAN generator checkpoint."
        )
    )

    parser.add_argument(
        "--load_checkpoint_epochs",
        type=int,
        required=True,
        help=(
            "Checkpoint epoch number to load. "
            "Example: 2000 loads G_model_2000.h5"
        ),
    )

    parser.add_argument(
        "--train_csv",
        type=Path,
        default=Path("data/train_latlon.csv"),
        help=(
            "Path to the training CSV used for model metadata "
            "and normalization."
        ),
    )

    parser.add_argument(
        "--test_csv",
        type=Path,
        default=Path("data/test_latlon.csv"),
        help=(
            "Path to the test CSV containing semantic "
            "trajectory data."
        ),
    )

    parser.add_argument(
        "--test_npy",
        type=Path,
        default=Path("data/final_test.npy"),
        help=(
            "Path to the encoded test NPY generated "
            "by csv2npy.py."
        ),
    )

    parser.add_argument(
        "--encoded_test_csv",
        type=Path,
        default=Path("data/dev_test_encoded_final.csv"),
        help=(
            "Path to the encoded test CSV used for reconstructing "
            "generated trajectories."
        ),
    )

    parser.add_argument(
        "--generator_weights_dir",
        type=Path,
        default=Path("training_params"),
        help=(
            "Directory containing trained generator "
            "checkpoint weights."
        ),
    )

    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("results/syn_traj_test.csv"),
        help=(
            "Path where generated synthetic trajectories "
            "will be saved."
        ),
    )

    return parser.parse_args()

def get_max_trajectory_length(
    train_df,
    test_df,
    tid_col: str = "tid",
) -> int:
    """Return the longest trajectory length across train and test data."""

    train_max = train_df.groupby(tid_col).size().max()
    test_max = test_df.groupby(tid_col).size().max()

    max_length = max(train_max, test_max)

    if max_length <= 0:
        raise ValueError("Could not determine a valid maximum trajectory length.")

    return int(max_length)


def save_max_length(max_length: int, output_dir: Path) -> Path:
    """Cache the maximum trajectory length used to build the model."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata.json"

    with metadata_path.open("w") as file:
        json.dump(
            {"max_length": max_length},
            file,
            indent=2,
        )

    return metadata_path


def load_max_length(checkpoint_dir: Path) -> int:
    """Load the cached maximum trajectory length for a trained model."""

    metadata_path = Path(checkpoint_dir) / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Model metadata not found: {metadata_path}"
        )

    with metadata_path.open("r") as file:
        metadata = json.load(file)

    return int(metadata["max_length"])

def get_max_trajectory_length(
    train_df,
    test_df,
    tid_col="tid",
):
    """Return the longest trajectory length across train and test data."""

    train_max = train_df.groupby(tid_col).size().max()
    test_max = test_df.groupby(tid_col).size().max()

    if pd.isna(train_max) or pd.isna(test_max):
        raise ValueError(
            "Could not determine max_length from the trajectory data."
        )

    max_length = int(max(train_max, test_max))

    if max_length <= 0:
        raise ValueError(
            f"max_length must be positive, got {max_length}."
        )

    return max_length