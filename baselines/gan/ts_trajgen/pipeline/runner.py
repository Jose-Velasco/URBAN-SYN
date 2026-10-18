from __future__ import annotations

import shlex
import subprocess
from collections.abc import Iterable

from pipeline.stage import Stage


class PipelineStageError(RuntimeError):
    """Raised when a pipeline stage exits with a non-zero return code."""

    def __init__(self, stage: Stage, returncode: int) -> None:
        self.stage = stage
        self.returncode = returncode

        super().__init__(
            f"Pipeline stage {stage.name!r} failed with exit code {returncode}."
        )


class PipelineRunner:
    """Execute TS-TrajGen pipeline stages sequentially.

    The first wrapper iteration intentionally keeps execution simple:

    - stages run in the order supplied by the caller;
    - commands execute directly with ``shell=False``;
    - stdout/stderr are inherited by the current terminal;
    - execution stops immediately when a stage fails;
    - ``dry_run`` prints the commands without executing them.
    """

    def __init__(self, *, dry_run: bool = False) -> None:
        """Initialize the runner.

        Args:
            dry_run:
                If True, print each stage exactly as it would be executed but
                do not start any subprocesses.
        """
        self.dry_run = dry_run

    def run(self, stages: Iterable[Stage]) -> None:
        """Run the supplied stages sequentially.

        Args:
            stages:
                Ordered stages to execute.

        Raises:
            FileNotFoundError:
                If a stage working directory does not exist.
            PipelineStageError:
                If a stage command exits with a non-zero return code.
        """
        stage_list = tuple(stages)

        if not stage_list:
            print("No pipeline stages selected.", flush=True)
            return

        total = len(stage_list)

        for index, stage in enumerate(stage_list, start=1):
            self.run_stage(stage, index=index, total=total)

        if self.dry_run:
            print(
                f"\nDry run complete: {total} stage(s) inspected.",
                flush=True,
            )
        else:
            print(
                f"\nPipeline complete: {total} stage(s) succeeded.",
                flush=True,
            )

    def run_stage(
        self,
        stage: Stage,
        *,
        index: int | None = None,
        total: int | None = None,
    ) -> None:
        """Run one pipeline stage.

        Args:
            stage:
                Stage to execute.
            index:
                Optional one-based position used for progress display.
            total:
                Optional total number of stages used for progress display.

        Raises:
            FileNotFoundError:
                If ``stage.cwd`` does not exist or is not a directory.
            PipelineStageError:
                If the subprocess exits unsuccessfully.
        """
        self._validate_working_directory(stage)
        self._print_stage(stage, index=index, total=total)

        if self.dry_run:
            return

        try:
            subprocess.run(
                list(stage.command),
                cwd=stage.cwd,
                check=True,
                shell=False,
            )
        except subprocess.CalledProcessError as exc:
            print(
                f"\nFAILED: {stage.name} "
                f"(exit code {exc.returncode})",
                flush=True,
            )
            raise PipelineStageError(stage, exc.returncode) from exc

        print(f"SUCCESS: {stage.name}", flush=True)

    @staticmethod
    def _validate_working_directory(stage: Stage) -> None:
        """Fail early when a configured stage working directory is invalid."""
        if not stage.cwd.exists():
            raise FileNotFoundError(
                f"Working directory for stage {stage.name!r} does not exist: "
                f"{stage.cwd}"
            )

        if not stage.cwd.is_dir():
            raise NotADirectoryError(
                f"Working directory for stage {stage.name!r} is not a directory: "
                f"{stage.cwd}"
            )

    @staticmethod
    def _print_stage(
        stage: Stage,
        *,
        index: int | None,
        total: int | None,
    ) -> None:
        """Print the stage metadata and shell-readable command."""
        if index is not None and total is not None:
            heading = f"[{index}/{total}] {stage.name}"
        else:
            heading = stage.name

        print("\n" + "=" * 80, flush=True)
        print(heading, flush=True)
        print(stage.description, flush=True)
        print(f"Group: {stage.group}", flush=True)
        print(f"Working directory: {stage.cwd}", flush=True)
        print(f"Command: {shlex.join(stage.command)}", flush=True)
        print("=" * 80, flush=True)
