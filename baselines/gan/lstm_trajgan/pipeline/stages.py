"""Construct the concrete commands used by the LSTM-TrajGAN wrapper."""

from __future__ import print_function

import sys

from pipeline.stage import Stage


VALID_TARGETS = {
    "build",
    "encode-train",
    "encode-test",
    "prepare",
    "train",
    "generate",
    "all",
}


def stages_for_target(config, target):
    """Return the ordered stages for one primitive or convenience target."""
    if target not in VALID_TARGETS:
        allowed = ", ".join(sorted(VALID_TARGETS))
        raise ValueError(
            "Unknown target '{}'. Expected one of: {}".format(target, allowed)
        )

    stage_map = _build_stage_map(config)

    if target == "prepare":
        names = ("build", "encode-train", "encode-test")
    elif target == "all":
        names = (
            "build",
            "encode-train",
            "encode-test",
            "train",
            "generate",
        )
    else:
        names = (target,)

    return tuple(stage_map[name] for name in names)

def _build_source_arguments(config):
    """Build dataset-source CLI arguments for the adapter."""
    if config.uses_pre_split_inputs:
        return [
            "--train_input_path",
            str(config.train_source_path),
            "--test_input_path",
            str(config.test_source_path),
        ]

    return [
        "--input_path",
        str(config.source_path),
        "--train_ratio",
        str(config.train_ratio),
    ]


def _build_builder_command(config, python):
    """Build the complete build_lstm_trajgan_inputs.py command."""
    command = [
        python,
        str(config.builder_path),
    ]

    command.extend(_build_source_arguments(config))

    command.extend(
        [
            "--out_dir",
            str(config.data_dir),
            "--dataset_name",
            config.dataset_name,
            "--random_state",
            str(config.random_state),
            "--category_mode",
            config.category_mode,
            "--constant_category",
            str(config.constant_category),
            "--log_dir",
            str(config.log_dir),
        ]
    )

    if config.max_trajectories is not None:
        command.extend(
            [
                "--max_trajectories",
                str(config.max_trajectories),
            ]
        )

    return command


def _build_required_paths(config):
    """Return files required before the adapter build stage can run."""
    return [config.builder_path] + list(config.source_paths)

def _build_stage_map(config):
    python = sys.executable

    build_command = _build_builder_command(
        config,
        python,
    )

    train_command = [
        python,
        str(config.train_script),
        "--train_csv",
        str(config.train_csv),
        "--test_csv",
        str(config.test_csv),
        "--train_npy",
        str(config.train_npy),
        "--output_dir",
        str(config.checkpoint_dir),
        "--epochs",
        str(config.epochs),
        "--batch_size",
        str(config.batch_size),
        "--save_params_rate",
        str(config.save_params_rate),
    ]

    if config.max_length is not None:
        train_command.extend(["--max_length", str(config.max_length)])

    generate_command = [
        python,
        str(config.predict_script),
        "--load_checkpoint_epochs",
        str(config.checkpoint_epoch),
        "--train_csv",
        str(config.train_csv),
        "--test_csv",
        str(config.test_csv),
        "--test_npy",
        str(config.test_npy),
        "--encoded_test_csv",
        str(config.test_encoded_csv),
        "--generator_weights_dir",
        str(config.checkpoint_dir),
        "--output_csv",
        str(config.generated_csv),
    ]

    if config.max_length is not None:
        generate_command.extend(["--max_length", str(config.max_length)])

    final_epoch = config.epochs

    return {
        "build": Stage(
            name="build",
            group="prepare",
            description=(
                "Convert trajectory source data to the CSV inputs expected "
                "by LSTM-TrajGAN."
            ),
            cwd=config.workspace_dir,
            command=build_command,
            required_paths=_build_required_paths(config),
            expected_paths=[
                config.train_csv,
                config.test_csv,
                config.train_encoded_csv,
                config.test_encoded_csv,
                config.builder_metadata,
            ],
        ),
        "encode-train": Stage(
            name="encode-train",
            group="prepare",
            description=(
                "Convert the encoded training CSV to the original "
                "LSTM-TrajGAN NPY representation."
            ),
            cwd=config.workspace_dir,
            command=[
                python,
                str(config.csv2npy_script),
                "--load_path",
                str(config.train_encoded_csv),
                "--save_path",
                str(config.train_npy),
                "--tid_col",
                config.tid_col,
            ],
            required_paths=[
                config.csv2npy_script,
                config.train_encoded_csv,
            ],
            expected_paths=[config.train_npy],
        ),
        "encode-test": Stage(
            name="encode-test",
            group="prepare",
            description=(
                "Convert the encoded test CSV to the seven-field NPY "
                "representation required by predict.py."
            ),
            cwd=config.workspace_dir,
            command=[
                python,
                str(config.csv2npy_script),
                "--load_path",
                str(config.test_encoded_csv),
                "--save_path",
                str(config.test_npy),
                "--tid_col",
                config.tid_col,
                "--generate_test_npy",
            ],
            required_paths=[
                config.csv2npy_script,
                config.test_encoded_csv,
            ],
            expected_paths=[config.test_npy],
        ),
        "train": Stage(
            name="train",
            group="train",
            description="Train LSTM-TrajGAN and save periodic checkpoints.",
            cwd=config.workspace_dir,
            command=train_command,
            required_paths=[
                config.train_script,
                config.train_csv,
                config.test_csv,
                config.train_npy,
            ],
            expected_paths=[
                config.checkpoint_dir
                / "C_model_{}.h5".format(final_epoch),
                config.checkpoint_dir
                / "G_model_{}.h5".format(final_epoch),
                config.checkpoint_dir
                / "D_model_{}.h5".format(final_epoch),
                config.checkpoint_dir / "metadata.json",
            ],
        ),
        "generate": Stage(
            name="generate",
            group="generate",
            description=(
                "Generate synthetic trajectories from a saved generator "
                "checkpoint."
            ),
            cwd=config.workspace_dir,
            command=generate_command,
            required_paths=[
                config.predict_script,
                config.train_csv,
                config.test_csv,
                config.test_npy,
                config.test_encoded_csv,
                config.generator_checkpoint,
            ],
            expected_paths=[config.generated_csv],
        ),
    }
