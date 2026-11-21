"""Canonical NYC input names and the legacy names shipped with this baseline."""

from pathlib import Path

CANONICAL_INPUTS = ('nyc.geo', 'nyc.rel', 'nyc_mm_train.csv', 'nyc_mm_test.csv')
ALIASES = {
    'nyc.geo': ('roadmap.geo',),
    'nyc.rel': ('roadmap.rel',),
    'nyc_mm_train.csv': ('NYC_Taxi_trajectory_train.csv',),
    'nyc_mm_test.csv': ('NYC_Taxi_trajectory_test.csv',),
}


def resolve_input(directory, name):
    """Return the canonical NYC input path, accepting an existing legacy alias."""
    directory = Path(directory)
    for candidate in (name, *ALIASES.get(name, ())):
        path = directory / candidate
        if path.is_file():
            return path
    aliases = ', '.join(ALIASES.get(name, ()))
    suffix = f" or alias {aliases}" if aliases else ''
    raise FileNotFoundError(f'Missing NYC input {name}{suffix} in {directory}')


def input_paths(directory):
    return {name: resolve_input(directory, name) for name in CANONICAL_INPUTS}
