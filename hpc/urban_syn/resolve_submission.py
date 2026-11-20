"""Resolve an individual experiment or study selection for Slurm submission.

The program only reads YAML; it does not import PyTorch or submit jobs.
Both the login-node submitter and the container can use the same decisions.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Submission:
    """Validated experiment selection and optional Slurm defaults."""

    config: Path
    pipeline: str
    partition: str = ""
    gres: str = ""
    time: str = ""
    memory: str = ""
    cpus: str = ""
    mail_user: str = ""
    mail_type: str = ""
    study_name: str = ""
    experiment_key: str = ""


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read a mapping from YAML, rejecting empty or malformed documents."""
    if not path.is_file():
        raise FileNotFoundError(f"YAML file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return data


def _inside_project(value: str, root: Path) -> Path:
    """Resolve repository-relative paths without allowing outside files."""
    path = Path(value).expanduser()
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Experiment files must be inside the project: {path}")
    return path


def _pipeline(experiment: dict[str, Any], config_path: Path) -> str:
    """Identify an existing pipeline without introducing new YAML keys.

    Both experiment schemas reject unknown top-level keys. We distinguish
    them by their existing 'pretraining' and 'model.diffusion' sections.
    """
    pretrain = "pretraining" in experiment
    joint = isinstance(experiment.get("model"), dict) and "diffusion" in experiment["model"]
    if pretrain == joint:
        raise ValueError(f"Cannot distinguish link pretraining from joint diffusion: {config_path}")
    return "link_pretrain" if pretrain else "joint"


def resolve_submission(
    root: Path, *, config: str | None = None,
    study: str | None = None, experiment_key: str | None = None
) -> Submission:
    """Resolve YAML selection and scheduling defaults without side effects.

    Args:
        root: Repository root containing the experiment YAML files.
        config: Standalone experiment YAML path, relative to the root.
        study: Optional YAML grouping several experiments and Slurm defaults.
        experiment_key: Name of one experiment listed in the study.

    Returns:
        Submission with an absolute config path, pipeline, and scheduling defaults.

    Raises:
        ValueError: For invalid selection, paths, or unsupported settings.
    """
    root = root.resolve()
    if (config is None) == (study is None):
        raise ValueError("Choose either --config or --study, not both.")

    defaults: dict[str, Any] = {}
    study_name = ""
    if study is not None:
        if not experiment_key:
            raise ValueError("--experiment is required with --study.")
        study_data = _read_yaml(_inside_project(study, root))
        if set(study_data) - {"study", "hpc"}:
            raise ValueError("Study YAML accepts only 'study' and 'hpc' sections.")
        study_info = study_data.get("study")
        if not isinstance(study_info, dict):
            raise ValueError("Study YAML requires a 'study' mapping.")
        study_name = str(study_info.get("name", ""))
        experiments = study_info.get("experiments")
        if not isinstance(experiments, dict) or experiment_key not in experiments:
            raise ValueError(f"Experiment '{experiment_key}' is not listed in {study}.")
        entry = experiments[experiment_key]
        # Either a path string or an explicit {config: path} is supported.
        config = entry.get("config") if isinstance(entry, dict) else entry
        if not isinstance(config, str) or not config.strip():
            raise ValueError(f"Experiment '{experiment_key}' needs a config path.")
        defaults = study_data.get("hpc") or {}
        if not isinstance(defaults, dict):
            raise ValueError("Study hpc settings must be a mapping.")
        supported = {"partition", "gres", "time", "memory", "cpus_per_task", "mail_user", "mail_type"}
        if set(defaults) - supported:
            raise ValueError(f"Unknown HPC settings: {sorted(set(defaults) - supported)}")

    if not isinstance(config, str) or not config.strip():
        raise ValueError("An experiment config path is required.")
    path = _inside_project(config, root)
    data = _read_yaml(path)
    pipeline = _pipeline(data, path)

    # A study's shared graph is a consistency check, not a second copy of the
    # graph feature settings. The individual YAML remains standalone-valid.
    if study is not None:
        graph_ref = study_info.get("graph_config")
        if graph_ref is not None:
            config_graph = (data.get("paths") or {}).get("graph_config")
            if not isinstance(config_graph, str):
                raise ValueError(f"Missing paths.graph_config in {path}")
            if _inside_project(graph_ref, root) != _inside_project(config_graph, root):
                raise ValueError(f"Graph YAML differs from the study for {experiment_key}: {config_graph}")

    def setting(name: str) -> str:
        value = defaults.get(name)
        if value is None:
            return ""
        if not isinstance(value, (str, int)):
            raise ValueError(f"hpc.{name} must be a string or integer.")
        result = str(value)
        if "\n" in result or "\r" in result:
            raise ValueError(f"hpc.{name} cannot contain line breaks.")
        return result

    return Submission(
        config=path, pipeline=pipeline,
        partition=setting("partition"), gres=setting("gres"),
        time=setting("time"), memory=setting("memory"),
        cpus=setting("cpus_per_task"), mail_user=setting("mail_user"),
        mail_type=setting("mail_type"), study_name=study_name,
        experiment_key=experiment_key or ""
    )


def main() -> int:
    """Emit newline-separated fields for the shell or a single requested field."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--config")
    selection.add_argument("--study")
    parser.add_argument("--experiment")
    parser.add_argument("--field", choices=("pipeline", "config"))
    args = parser.parse_args()
    try:
        selected = resolve_submission(
            args.project_root, config=args.config, study=args.study,
            experiment_key=args.experiment
        )
    except (ValueError, FileNotFoundError, yaml.YAMLError) as error:
        parser.exit(2, f"ERROR: {error}\n")

    if args.field:
        print(getattr(selected, args.field))
    else:
        # One field per line, in a fixed order. The submitter never evaluates
        # this output as shell code, including paths with spaces.
        for value in (
            selected.config, selected.pipeline, selected.partition, selected.gres,
            selected.time, selected.memory, selected.cpus, selected.mail_user,
            selected.mail_type, selected.study_name, selected.experiment_key
        ):
            print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
