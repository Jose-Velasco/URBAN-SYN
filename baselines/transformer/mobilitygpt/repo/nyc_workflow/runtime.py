"""Installed-source provenance and path validation, independent of the caller's cwd."""
import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parent
EXCLUDED_SOURCE_DIRS = {'output', 'outputs', '.git', '.venv', 'venv', 'env',
    '__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache', 'build', 'dist',
    'node_modules'}


def source_files(directory):
    """Only importable source trees; never descend into run snapshots or environments."""
    for current, directories, files in os.walk(directory):
        current = Path(current)
        if '.pipeline.lock' in files or 'pyvenv.cfg' in files:
            directories[:] = []
            continue
        directories[:] = sorted(name for name in directories
            if name not in EXCLUDED_SOURCE_DIRS and not name.endswith('.egg-info')
            and (current/name/'__init__.py').is_file()
            and not (current/name).is_symlink())
        yield from (current/name for name in sorted(files) if name.endswith('.py'))


def require_output(path, inputs=()):
    path = Path(path).expanduser().resolve()
    if any(part in {'.git', '.venv', 'venv', '__pycache__'} for part in path.parts):
        raise ValueError(f'Output overlaps protected metadata/environment: {path}')
    if path.exists() and not path.is_dir():
        raise ValueError(f'Refusing to overwrite existing output file: {path}')
    # A fresh child of a source tree is allowed, but never a source-containing
    # directory itself. Resolving first also prevents symlink aliases bypassing this.
    for name in ('nyc_workflow', 'mobilitygpt', 'finetuners', 'datasets', 'preprocessing'):
        directory = PACKAGE.parent/name
        if directory.is_dir():
            if directory.is_relative_to(path):
                raise ValueError(f'Output overlaps source files: {path}')
            if path.is_relative_to(directory) and any(p.is_relative_to(path) for p in source_files(directory)):
                raise ValueError(f'Output overlaps source files: {path}')
    for value in inputs:
        value = Path(value).expanduser().resolve()
        if path.is_relative_to(value) or value.is_relative_to(path):
            raise ValueError(f'Output and input must be separate: {path}, {value}')
    return path


def module_command(module, *arguments):
    return [str(Path(sys.executable).absolute()), '-u', '-m', module, *map(str, arguments)]


def source_entries():
    entries = {}
    for name in ('nyc_workflow', 'mobilitygpt', 'finetuners', 'datasets', 'preprocessing'):
        module = importlib.import_module(name)
        directory = (Path(module.__file__).resolve().parent if module.__file__ else
                     Path(next(iter(module.__path__))).resolve())
        entries.update({f'{name}/{p.relative_to(directory)}': p for p in source_files(directory)})
    for name in ('requirements.txt', 'notebook_provenance.json'):
        entries[f'nyc_workflow/{name}'] = PACKAGE/name
    project = PACKAGE.parent/'pyproject.toml'
    if project.is_file():
        entries['pyproject.toml'] = project
    return dict(sorted(entries.items()))


def snapshot_sources(directory):
    directory = Path(directory)
    for name, source in source_entries().items():
        target = directory/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def git_record(*arguments):
    project = PACKAGE.parent
    if not (project/'.git').exists():
        return 'Git metadata unavailable; installed source snapshots are authoritative.\n'
    process = subprocess.run(['git', '-C', str(project), *arguments], capture_output=True, text=True)
    return process.stdout if process.returncode == 0 else 'Git record unavailable: '+process.stderr
