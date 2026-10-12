"""Definition of one executable LSTM-TrajGAN pipeline stage."""

from pathlib import Path


class Stage(object):
    """A small immutable-ish description of one subprocess stage."""

    def __init__(
        self,
        name,
        group,
        description,
        cwd,
        command,
        required_paths=None,
        expected_paths=None,
    ):
        self.name = str(name)
        self.group = str(group)
        self.description = str(description)
        self.cwd = Path(cwd)
        self.command = tuple(str(part) for part in command)
        self.required_paths = tuple(
            Path(path) for path in (required_paths or ())
        )
        self.expected_paths = tuple(
            Path(path) for path in (expected_paths or ())
        )

        if not self.name.strip():
            raise ValueError("Stage name cannot be empty.")

        if not self.group.strip():
            raise ValueError(
                "Stage '{}' must belong to a group.".format(self.name)
            )

        if not self.command:
            raise ValueError(
                "Stage '{}' must define a command.".format(self.name)
            )
