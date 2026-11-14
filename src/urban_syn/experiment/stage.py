"""Small description of a pipeline stage and its external requirements."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Stage:
    """Describe work without importing models or starting a subprocess.

    ``required_files`` are inputs that must already exist on disk.
    Produced artifacts are checked by their eventual stage handlers.
    """

    name: str
    description: str
    required_files: tuple[Path, ...] = ()
    missing_settings: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
