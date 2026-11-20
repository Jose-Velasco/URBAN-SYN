"""Reusable lightweight stage runner for URBAN-SYN experiment pipelines."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .stage import Stage


StageHandler = Callable[[Any], None]


class PipelineRunner:
    """Check stage prerequisites and dispatch injected execution handlers.

    Configuration objects must provide ``settings.experiment.name``,
    ``settings.paths.output_root``, ``config_path``, and ``project_root``.
    ``emit`` can be a logger's ``info`` method for console/file logging.
    """

    def __init__(
        self,
        config: Any,
        handlers: Mapping[str, StageHandler] | None = None,
        emit: Callable[[str], None] | None = None
    ):
        """Register stage handlers without constructing runtime dependencies."""
        self.config = config
        self.handlers = dict(handlers or {})
        self.emit = emit if emit is not None else print

    def run(self, stages: Sequence[Stage], dry_run: bool = False, validate_only: bool = False) -> None:
        """Plan, preflight, or execute an ordered sequence of stages.

        Dry runs never check files or invoke handlers. ``validate_only`` checks
        declared prerequisites and invokes an optional ``validate`` handler,
        but skips all artifact-producing stages.
        """
        if dry_run and validate_only:
            raise ValueError("Choose either dry_run or validate_only, not both.")

        mode = "DRY RUN" if dry_run else "VALIDATE ONLY" if validate_only else "EXECUTE"
        self.emit(f"Experiment: {self.config.settings.experiment.name} | Mode: {mode}")
        self.emit(f"Config: {self.config.config_path}")
        self.emit(f"Project root: {self.config.project_root}")
        self.emit(f"Output root: {self.config.settings.paths.output_root}")

        for index, stage in enumerate(stages, start=1):
            self.emit(f"[{index}/{len(stages)}] {stage.name}: {stage.description}")
            if stage.depends_on:
                self.emit(f"  Prior stage(s): {', '.join(stage.depends_on)}")
            if stage.required_files:
                self.emit("  Existing inputs:")
                for path in stage.required_files:
                    self.emit(f"    {path}")
            if stage.missing_settings:
                self.emit(f"  Settings needed: {', '.join(stage.missing_settings)}")

            if dry_run:
                self.emit("  Status: ready to inspect (file checks skipped)")
                continue

            self._validate_inputs(stage)
            self.emit("  Input checks passed")
            # A configured validate handler can perform deeper read-only checks.
            # The joint experiment has no validate handler and retains its
            # previous input-only validation behavior.
            if validate_only and stage.name != "validate":
                continue
            handler = self.handlers.get(stage.name)
            if handler is None:
                if stage.name == "validate":
                    continue
                raise NotImplementedError(
                    f"Stage '{stage.name}' is planned but has no execution handler yet. "
                    "Use --dry-run or --validate-only until integration is implemented."
                )
            handler(self.config)
            if not validate_only:
                self.emit(f"  Stage completed: {stage.name}")

        self.emit(f"{mode} finished for {len(stages)} stage(s).")

    @staticmethod
    def _validate_inputs(stage: Stage) -> None:
        """Reject missing required settings or pre-existing input files."""
        problems = [f"Required setting is missing: {name}" for name in stage.missing_settings]
        problems.extend(
            f"Required file does not exist: {path}"
            for path in stage.required_files if not path.is_file()
        )
        if problems:
            details = "\n  - ".join(problems)
            raise FileNotFoundError(f"Cannot run stage '{stage.name}':\n  - {details}")
