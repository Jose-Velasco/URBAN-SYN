#!/usr/bin/env python3
"""Run the LSTM-TrajGAN baseline through a small config-driven wrapper."""

from __future__ import print_function

import argparse
from pathlib import Path

from pipeline.config import load_pipeline_config
from pipeline.runner import PipelineRunner
from pipeline.stages import VALID_TARGETS, stages_for_target


def parse_args():
    """Parse wrapper command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run LSTM-TrajGAN preparation, training, and generation stages."
    )
    parser.add_argument(
        "target",
        choices=sorted(VALID_TARGETS),
        help=(
            "Pipeline target: build, encode-train, encode-test, prepare, "
            "train, generate, or all."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("pipeline_configs/nyc.yaml"),
        help=(
            "Wrapper YAML configuration "
            "(default: pipeline_configs/nyc.yaml)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved stages and commands without running them.",
    )
    return parser.parse_args()


def main():
    """Load the wrapper config and execute the selected target."""
    args = parse_args()
    config = load_pipeline_config(args.config)
    stages = stages_for_target(config, args.target)

    print("=" * 80)
    print("LSTM-TrajGAN pipeline")
    print("=" * 80)
    print("Dataset: {}".format(config.dataset_name))
    print("Target: {}".format(args.target))
    print("Workspace: {}".format(config.workspace_dir))
    print("Repository: {}".format(config.repo_dir))
    print("Data directory: {}".format(config.data_dir))
    print("Output directory: {}".format(config.outputs_dir))
    print("Config: {}".format(args.config))
    print("Stages: {}".format(len(stages)))
    print("Dry run: {}".format(args.dry_run))

    runner = PipelineRunner(dry_run=args.dry_run)
    runner.run(stages)


if __name__ == "__main__":
    main()
