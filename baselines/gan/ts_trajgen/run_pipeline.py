from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline.config import PipelineConfig, load_pipeline_config
from pipeline.runner import PipelineRunner, PipelineStageError
from pipeline.stages import (
    PIPELINE_GROUP_ORDER,
    stages_for_all,
    stages_for_group,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the TS-TrajGen wrapper."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the refactored TS-TrajGen baseline pipeline inside the "
            "ts-trajgen container."
        )
    )

    parser.add_argument(
        "target",
        choices=(*PIPELINE_GROUP_ORDER, "all"),
        help=(
            "Pipeline group to run. Use 'all' to execute every group in "
            "pipeline order."
        ),
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the wrapper pipeline YAML configuration.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected stages and commands without executing them.",
    )

    return parser.parse_args()


def select_stages(
    config: PipelineConfig,
    target: str,
):
    """Select the ordered stages requested by the CLI target."""
    if target == "all":
        return stages_for_all(config)

    return stages_for_group(config, target)


def print_run_summary(
    config: PipelineConfig,
    *,
    target: str,
    stage_count: int,
    dry_run: bool,
) -> None:
    """Print the high-level wrapper configuration before execution."""
    print("=" * 80)
    print("TS-TrajGen Pipeline")
    print("=" * 80)
    print(f"Dataset:          {config.dataset_name}")
    print(f"Target:           {target}")
    print(f"Stages selected:  {stage_count}")
    print(f"Device:           {config.device}")
    print(f"Generation mode:  {config.generation_mode}")
    print(f"Dry run:          {dry_run}")
    print(f"Workspace:        {config.workspace_dir}")
    print(f"Repository:       {config.repo_dir}")
    print(f"Dataset dir:      {config.dataset_dir}")
    print(f"Model config:     {config.model_config_path}")
    print("=" * 80, flush=True)


def main() -> int:
    """Run the requested TS-TrajGen pipeline group."""
    args = parse_args()

    try:
        config = load_pipeline_config(args.config)
        stages = select_stages(config, args.target)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    print_run_summary(
        config,
        target=args.target,
        stage_count=len(stages),
        dry_run=args.dry_run,
    )

    runner = PipelineRunner(dry_run=args.dry_run)

    try:
        runner.run(stages)
    except PipelineStageError as exc:
        print(
            f"Pipeline stopped at stage {exc.stage.name!r}.",
            file=sys.stderr,
        )
        return exc.returncode if exc.returncode != 0 else 1
    except (FileNotFoundError, NotADirectoryError) as exc:
        print(f"Pipeline setup error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.", file=sys.stderr)
        return 130

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
