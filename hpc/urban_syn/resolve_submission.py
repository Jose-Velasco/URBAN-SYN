"""Resolve an individual experiment or study selection for Slurm submission.

The program only reads YAML; it does not import PyTorch or submit jobs.
Both the login-node submitter and the container can use the same decisions.
"""

from __future__ import annotations

import argparse
import json
import re
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
    exclude_nodes: str = ""
    encoder_checkpoint: str = ""
    source_run: str = ""


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
    encoder_checkpoint = ""
    source_run = ""
    pretrained_from = None
    experiments: dict[str, Any] = {}
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
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", study_name):
            raise ValueError("study.name must be a nonempty directory-safe name.")
        experiments = study_info.get("experiments")
        if not isinstance(experiments, dict) or experiment_key not in experiments:
            raise ValueError(f"Experiment '{experiment_key}' is not listed in {study}.")
        entry = experiments[experiment_key]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", experiment_key):
            raise ValueError("Study experiment keys must be directory-safe names.")
        if isinstance(entry, dict):
            unknown = set(entry) - {"config", "pretrained_from"}
            if unknown:
                raise ValueError(f"Unknown settings for experiment '{experiment_key}': {sorted(unknown)}")
            pretrained_from = entry.get("pretrained_from")
        # Either a path string or an explicit {config: path} is supported.
        config = entry.get("config") if isinstance(entry, dict) else entry
        if not isinstance(config, str) or not config.strip():
            raise ValueError(f"Experiment '{experiment_key}' needs a config path.")
        defaults = study_data.get("hpc") or {}
        if not isinstance(defaults, dict):
            raise ValueError("Study hpc settings must be a mapping.")
        supported = {
            "partition", "gres", "time", "memory", "cpus_per_task",
            "mail_user", "mail_type", "exclude_nodes"
        }
        if set(defaults) - supported:
            raise ValueError(f"Unknown HPC settings: {sorted(set(defaults) - supported)}")

    if not isinstance(config, str) or not config.strip():
        raise ValueError("An experiment config path is required.")
    path = _inside_project(config, root)
    data = _read_yaml(path)
    pipeline = _pipeline(data, path)

    if pretrained_from is not None:
        if pipeline != "joint":
            raise ValueError("pretrained_from is only supported for joint diffusion experiments.")
        if not isinstance(pretrained_from, str) or pretrained_from not in experiments:
            raise ValueError(f"pretrained_from must name an experiment in {study}.")
        source_entry = experiments[pretrained_from]
        source_config = source_entry.get("config") if isinstance(source_entry, dict) else source_entry
        if not isinstance(source_config, str):
            raise ValueError(f"Source experiment '{pretrained_from}' needs a config path.")
        source_path = _inside_project(source_config, root)
        source_data = _read_yaml(source_path)
        if _pipeline(source_data, source_path) != "link_pretrain":
            raise ValueError(f"pretrained_from '{pretrained_from}' must reference link pretraining.")
        source_graph = (source_data.get("paths") or {}).get("graph_config")
        study_graph = study_info.get("graph_config")
        if (not isinstance(source_graph, str) or not isinstance(study_graph, str) or
                _inside_project(source_graph, root) != _inside_project(study_graph, root)):
            raise ValueError(
                f"Pretraining source '{pretrained_from}' must use the study's graph YAML."
            )
        encoder_options = (data.get("model") or {}).get("road_encoder") or {}
        if encoder_options.get("initialization") != "pretrained":
            raise ValueError("Study pretrained_from requires model.road_encoder.initialization=pretrained.")
        if encoder_options.get("checkpoint") is not None:
            raise ValueError(
                "Set model.road_encoder.checkpoint: null when using study pretrained_from "
                "to avoid two competing checkpoint selections."
            )
        reference_path = (
            root / "outputs/experiments/studies" / study_name /
            pretrained_from / "selected_encoder.json"
        )
        if not reference_path.is_file():
            raise FileNotFoundError(
                f"No completed encoder has been published for '{pretrained_from}' in "
                f"study '{study_name}'. Submit that pretraining experiment first, "
                f"then retry. Expected: {reference_path}"
            )
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        if (not isinstance(reference, dict) or reference.get("format_version") != 1 or
                reference.get("study_name") != study_name or
                reference.get("experiment_key") != pretrained_from):
            raise ValueError(f"Invalid or mismatched pretraining reference: {reference_path}")
        source_run = str(reference.get("run_id", ""))
        run_dir = reference.get("run_dir")
        encoded = reference.get("encoder_checkpoint")
        if not source_run or not isinstance(run_dir, str) or not isinstance(encoded, str):
            raise ValueError(f"Incomplete pretraining reference: {reference_path}")
        run_path = _inside_project(run_dir, root)
        selected_path = _inside_project(encoded, root)
        if selected_path != run_path / "checkpoints/encoder.pt":
            raise ValueError(f"Pretraining reference has an unexpected encoder path: {reference_path}")
        if run_path.name != source_run or not selected_path.is_file():
            raise FileNotFoundError(
                f"Selected pretrained encoder is missing or does not match its run: {selected_path}"
            )
        encoder_checkpoint = str(selected_path)

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

    # Slurm expects a comma-separated host list for --exclude. Let the study
    # express it as a readable YAML list, and validate names before forwarding.
    excluded = defaults.get("exclude_nodes", [])
    if not isinstance(excluded, list):
        raise ValueError("hpc.exclude_nodes must be a list of node names.")
    for node in excluded:
        if not isinstance(node, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", node):
            raise ValueError("hpc.exclude_nodes contains an invalid node name.")

    return Submission(
        config=path, pipeline=pipeline,
        partition=setting("partition"), gres=setting("gres"),
        time=setting("time"), memory=setting("memory"),
        cpus=setting("cpus_per_task"), mail_user=setting("mail_user"),
        mail_type=setting("mail_type"), study_name=study_name,
        experiment_key=experiment_key or "", exclude_nodes=",".join(excluded),
        encoder_checkpoint=encoder_checkpoint, source_run=source_run
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
            selected.mail_type, selected.study_name, selected.experiment_key,
            selected.exclude_nodes, selected.encoder_checkpoint, selected.source_run
        ):
            print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
