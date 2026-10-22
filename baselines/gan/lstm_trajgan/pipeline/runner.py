"""Sequential subprocess runner for the LSTM-TrajGAN wrapper."""

from __future__ import print_function

import shlex
import subprocess
import time


class PipelineStageError(RuntimeError):
    """Raised when one pipeline stage exits unsuccessfully."""

    def __init__(self, stage, returncode):
        self.stage = stage
        self.returncode = returncode
        RuntimeError.__init__(
            self,
            "Pipeline stage '{}' failed with exit code {}.".format(
                stage.name,
                returncode,
            ),
        )


class PipelineRunner(object):
    """Run stages in order, stopping immediately when one fails."""

    def __init__(self, dry_run=False):
        self.dry_run = bool(dry_run)

    def run(self, stages):
        """Run an ordered sequence of Stage objects."""
        stage_list = tuple(stages)

        if not stage_list:
            print("No pipeline stages selected.", flush=True)
            return

        total = len(stage_list)
        total_start = time.time()

        for index, stage in enumerate(stage_list, 1):
            self.run_stage(stage, index=index, total=total)

        elapsed = time.time() - total_start

        if self.dry_run:
            print(
                "\nDry run complete: {} stage(s) inspected.".format(total),
                flush=True,
            )
        else:
            print(
                "\nPipeline complete: {} stage(s) succeeded in {:.2f} s.".format(
                    total,
                    elapsed,
                ),
                flush=True,
            )

    def run_stage(self, stage, index=None, total=None):
        """Print, preflight, execute, and verify one stage."""
        self._validate_working_directory(stage)
        self._print_stage(stage, index=index, total=total)

        if self.dry_run:
            return

        self._validate_required_paths(stage)

        start = time.time()

        try:
            subprocess.run(
                list(stage.command),
                cwd=str(stage.cwd),
                check=True,
                shell=False,
            )
        except subprocess.CalledProcessError as exc:
            print(
                "\nFAILED: {} (exit code {})".format(
                    stage.name,
                    exc.returncode,
                ),
                flush=True,
            )
            raise PipelineStageError(stage, exc.returncode)

        self._validate_expected_paths(stage)

        elapsed = time.time() - start
        print(
            "SUCCESS: {} ({:.2f} s)".format(stage.name, elapsed),
            flush=True,
        )

    @staticmethod
    def _validate_working_directory(stage):
        if not stage.cwd.exists():
            raise FileNotFoundError(
                "Working directory for stage '{}' does not exist: {}".format(
                    stage.name,
                    stage.cwd,
                )
            )

        if not stage.cwd.is_dir():
            raise NotADirectoryError(
                "Working directory for stage '{}' is not a directory: {}".format(
                    stage.name,
                    stage.cwd,
                )
            )

    @staticmethod
    def _validate_required_paths(stage):
        missing = [path for path in stage.required_paths if not path.exists()]

        if not missing:
            return

        message = [
            "Stage '{}' is missing required input(s):".format(stage.name)
        ]
        message.extend("  {}".format(path) for path in missing)
        raise FileNotFoundError("\n".join(message))

    @staticmethod
    def _validate_expected_paths(stage):
        missing = [path for path in stage.expected_paths if not path.exists()]

        if not missing:
            return

        message = [
            "Stage '{}' completed but expected output(s) are missing:".format(
                stage.name
            )
        ]
        message.extend("  {}".format(path) for path in missing)
        raise RuntimeError("\n".join(message))

    @staticmethod
    def _print_stage(stage, index=None, total=None):
        if index is not None and total is not None:
            heading = "[{}/{}] {}".format(index, total, stage.name)
        else:
            heading = stage.name

        print("\n" + "=" * 80, flush=True)
        print(heading, flush=True)
        print(stage.description, flush=True)
        print("Group: {}".format(stage.group), flush=True)
        print("Working directory: {}".format(stage.cwd), flush=True)
        print(
            "Command: {}".format(
                " ".join(shlex.quote(part) for part in stage.command)
            ),
            flush=True,
        )

        if stage.expected_paths:
            print("Expected output(s):", flush=True)
            for path in stage.expected_paths:
                print("  {}".format(path), flush=True)

        print("=" * 80, flush=True)
