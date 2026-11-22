"""Lightweight stage runner with dry-run and input-only validation."""

import argparse
import sys
from collections.abc import Sequence

from .config import ResolvedExperimentConfig, load_experiment_config
from urban_syn.pipeline.runner import PipelineRunner
from .stages import TARGETS, stages_for_target



def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser used inside Docker or Apptainer.

    Returns
    -------
    argparse.ArgumentParser
        Parser for stage selection and experiment configuration.
    """
    parser = argparse.ArgumentParser(description="Run URBAN-SYN experiment stages.")
    parser.add_argument("target", choices=TARGETS)
    parser.add_argument("--config", required=True, help="Experiment YAML path")
    parser.add_argument("--project-root", default=None, help="Repository root (default: cwd)")
    parser.add_argument(
        "--encoder-checkpoint", default=None,
        help="Pretrained encoder.pt selected by the study or explicitly for this run"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Show stage plan without checking files")
    mode.add_argument("--validate-only", action="store_true", help="Check source files without executing")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve an experiment and dispatch the selected pipeline stages.

    Parameters
    ----------
    argv : Sequence[str] | None, default=None
        Command-line arguments, or process arguments when None.

    Returns
    -------
    int
        Exit code: zero on success, one on a handled stage error.
    """
    args = build_parser().parse_args(argv)
    try:
        config = load_experiment_config(
            args.config, project_root=args.project_root,
            encoder_checkpoint=args.encoder_checkpoint
        )
        stages = stages_for_target(config, args.target)
        # Keep stage outputs within this invocation, not in module-global state.
        training_run_dir = None
        generated_path = None

        def prepare_graph_handler(experiment: ResolvedExperimentConfig) -> None:
            """Reuse or generate the cached road graph before model loading.

            Parameters
            ----------
            experiment : ResolvedExperimentConfig
                Settings locating the source GeoPackage and graph cache.

            Returns
            -------
            None
                Both graph artifacts are available for downstream stages.
            """
            from .graph_preparation import prepare_experiment_graph

            prepare_experiment_graph(experiment)

        def train_handler(experiment: ResolvedExperimentConfig) -> None:
            """Train and retain the run directory for the generation stage.

            Parameters
            ----------
            experiment : ResolvedExperimentConfig
                Validated experiment configuration.

            Returns
            -------
            None
                Saves the training run directory for this pipeline execution.
            """
            nonlocal training_run_dir
            from .tracking import build_tracker
            from .trainer import run_joint_training

            training_run_dir = run_joint_training(experiment, build_tracker(experiment))

        def generate_handler(experiment: ResolvedExperimentConfig) -> None:
            """Generate GPS trajectories from the selected trained checkpoint.

            Parameters
            ----------
            experiment : ResolvedExperimentConfig
                Validated experiment configuration.

            Returns
            -------
            None
                Retains the generated Parquet path for the evaluator handoff.
            """
            nonlocal generated_path
            from .generation import run_generation
            from .tracking import build_tracker

            generated_path = run_generation(
                experiment,
                build_tracker(experiment),
                training_run_dir=training_run_dir
            )

        def export_handler(experiment: ResolvedExperimentConfig) -> None:
            """Write the shared evaluator's path handoff manifest.

            Parameters
            ----------
            experiment : ResolvedExperimentConfig
                Validated experiment configuration.

            Returns
            -------
            None
                Writes an evaluation handoff JSON next to generated Parquet.
            """
            from .generation import export_evaluation_handoff, resolve_generation_checkpoint

            source = generated_path
            if source is None:
                checkpoint = resolve_generation_checkpoint(experiment, training_run_dir)
                source = checkpoint.parent.parent / "generated" / experiment.settings.generation.output_name
            handoff = export_evaluation_handoff(experiment, source)
            print(f"  Evaluator handoff: {handoff}")

        PipelineRunner(config, handlers={
            "prepare-graph": prepare_graph_handler,
            "train": train_handler,
            "generate": generate_handler,
            "export-eval": export_handler
        }).run(
            stages,
            dry_run=args.dry_run,
            validate_only=args.validate_only
        )
    except (FileNotFoundError, NotImplementedError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0
