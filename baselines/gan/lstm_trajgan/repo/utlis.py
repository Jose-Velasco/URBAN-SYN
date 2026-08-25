import argparse
from pathlib import Path


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