"""Lightweight stage runner with dry-run and input-only validation."""

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence

from .config import ResolvedExperimentConfig, load_experiment_config
from .stage import Stage
from .stages import TARGETS, stages_for_target


StageHandler = Callable[[ResolvedExperimentConfig], None]


class PipelineRunner:
    """Check prerequisites and dispatch stages through injected handlers.

    The validation stage works immediately. Stage-specific execution handlers are injected by the entry point.
    They are never routed through the old smoke-test CLI.
    """

    def __init__(self, config: ResolvedExperimentConfig, handlers: Mapping[str, StageHandler] | None = None):
        """Register stage handlers without constructing runtime dependencies.

        Parameters
        ----------
        config : ResolvedExperimentConfig
            Validated experiment configuration.
        handlers : Mapping[str, StageHandler] | None, default=None
            Stage-specific execution functions supplied by the caller.
        """
        self.config = config
        self.handlers = dict(handlers or {})

    def run(self, stages: Sequence[Stage], dry_run: bool = False, validate_only: bool = False) -> None:
        """Plan, preflight, or execute an ordered sequence of stages.

        Parameters
        ----------
        stages : Sequence[Stage]
            Ordered stages to inspect or execute.
        dry_run : bool, default=False
            Print a plan without file or execution checks.
        validate_only : bool, default=False
            Verify existing inputs without executing handlers.

        Returns
        -------
        None
            Prints status or raises when a prerequisite fails.
        """
        if dry_run and validate_only:
            raise ValueError("Choose either dry_run or validate_only, not both.")

        mode = "DRY RUN" if dry_run else "VALIDATE ONLY" if validate_only else "EXECUTE"
        print(f"Experiment: {self.config.settings.experiment.name} | Mode: {mode}")
        print(f"Config: {self.config.config_path}")
        print(f"Project root: {self.config.project_root}")
        print(f"Output root: {self.config.settings.paths.output_root}")

        for index, stage in enumerate(stages, start=1):
            print(f"\n[{index}/{len(stages)}] {stage.name}: {stage.description}")
            if stage.depends_on:
                print(f"  Prior stage(s): {', '.join(stage.depends_on)}")
            if stage.required_files:
                print("  Existing inputs:")
                for path in stage.required_files:
                    print(f"    {path}")
            if stage.missing_settings:
                print(f"  Settings needed: {', '.join(stage.missing_settings)}")

            if dry_run:
                print("  Status: ready to inspect (file checks skipped)")
                continue

            self._validate_inputs(stage)
            print("  Input checks passed")
            if validate_only or stage.name == "validate":
                continue

            handler = self.handlers.get(stage.name)
            if handler is None:
                raise NotImplementedError(
                    f"Stage '{stage.name}' is planned but has no execution handler yet. "
                    "Use --dry-run or --validate-only until integration is implemented."
                )
            handler(self.config)
            print(f"  Stage completed: {stage.name}")

        print(f"\n{mode} finished for {len(stages)} stage(s).")

    @staticmethod
    def _validate_inputs(stage: Stage) -> None:
        """Verify pre-existing files and settings required by a stage.

        Parameters
        ----------
        stage : Stage
            Stage whose prerequisites are checked.

        Returns
        -------
        None
            Raises if any input or required setting is unavailable.
        """
        problems = [f"Required setting is missing: {name}" for name in stage.missing_settings]
        problems.extend(
            f"Required file does not exist: {path}"
            for path in stage.required_files if not path.is_file()
        )
        if problems:
            details = "\n  - ".join(problems)
            raise FileNotFoundError(f"Cannot run stage '{stage.name}':\n  - {details}")


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
        config = load_experiment_config(args.config, project_root=args.project_root)
        stages = stages_for_target(config, args.target)
        # Keep stage outputs within this invocation, not in module-global state.
        training_run_dir = None
        generated_path = None

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
