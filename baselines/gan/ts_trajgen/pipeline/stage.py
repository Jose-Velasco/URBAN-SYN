from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Stage:
    """A single executable stage in the TS-TrajGen wrapper pipeline.

    The stage is intentionally small for the first wrapper iteration.
    Command construction happens in ``pipeline.stages`` after the pipeline
    configuration has been loaded, so the runner only needs to know what
    command to execute and where to execute it.

    Attributes:
        name:
            Unique, human-readable stage identifier.
        group:
            Pipeline group that owns the stage, such as ``prepare`` or
            ``pretrain-road``.
        description:
            Short explanation printed by the runner before execution.
        cwd:
            Working directory used when executing the command.
        command:
            Fully resolved command and arguments. Each item is a separate
            subprocess argument; commands are never executed through a shell.
    """

    name: str
    group: str
    description: str
    cwd: Path
    command: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the minimal invariants required by the runner."""
        if not self.name.strip():
            raise ValueError("Stage name cannot be empty.")

        if not self.group.strip():
            raise ValueError(f"Stage {self.name!r} must belong to a group.")

        if not self.command:
            raise ValueError(f"Stage {self.name!r} must define a command.")

        if not self.command[0].strip():
            raise ValueError(
                f"Stage {self.name!r} has an invalid executable in its command."
            )