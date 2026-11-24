"""Resolve selected study experiments and their Slurm checkpoint dependencies.

All substantive input validation lives in config_validation.py. The batch-node
checkpoint and snapshot helpers remain standard-library-only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# The resolver is also loaded directly by path from HPC scripts and regression tests.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from config_validation import (
    HPCSettings, StudyExperiment, StudyFile, StudySettings,
    detect_pipeline, inside_project, read_yaml_mapping,
    validate_encoder_source, validate_pretraining_dependency,
    validate_published_encoder_reference, validate_source_graph,
    validate_study_entry, validate_study_graph
)

_read_yaml = read_yaml_mapping
_inside_project = inside_project
_pipeline = detect_pipeline


@dataclass(frozen=True)
class Submission:
    """Resolved experiment selection and shell-compatible Slurm defaults."""

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
    pretrained_from: str = ""
    pretrain_config: str = ""
    source_study_name: str = ""
    job_name: str = ""
    pretrain_job_name: str = ""
    composed_config: dict[str, Any] | None = None


def _compose(root: Path, config_dir: Path, preset: str) -> dict[str, Any]:
    """Load the standalone Hydra composer without requiring an HPC package."""
    source = root / "hpc/urban_syn/compose_experiment.py"
    spec = importlib.util.spec_from_file_location("urban_syn_hydra_composer", source)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Hydra composer not found: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.compose_experiment(config_dir, preset)


def _job_names(config: dict[str, Any]) -> tuple[str, str]:
    """Generate compact job names from the validated preset metadata."""
    tags = config["tracking"]["tags"]
    architecture = tags["gnn_architecture"]
    objective = tags["connectivity_loss"]
    strategy = tags["gnn_update"]
    mode = ("base-warm" if strategy == "warmup" else "base") if objective == "none" else (
        f"{objective}-{'warm' if strategy == 'warmup' else 'frz'}"
    )
    if tags.get("profile") == "smoke":
        mode = mode.replace("warm", "wrm")
    suffix = "-s" if tags.get("profile") == "smoke" else ""
    return (f"us-j-{architecture}-{mode}{suffix}",
            f"us-p-{architecture}-{tags['pretraining_objective']}")


def _study_experiment(
    root: Path, study: str, experiment_key: str
) -> tuple[StudyFile, StudyExperiment, str]:
    """Validate the study and return the selected entry without file writes."""
    data = StudyFile.model_validate(_read_yaml(_inside_project(study, root)))
    if experiment_key not in data.study.experiments:
        raise ValueError(f"Experiment '{experiment_key}' is not listed in {study}.")
    entry = data.study.experiments[experiment_key]
    if isinstance(entry, str):
        entry = StudyExperiment(config=entry)
    validate_study_entry(entry, experiment_key)
    return data, entry, data.study.name


def _selected_published_encoder(
    root: Path, source_study: str, source_experiment: str
) -> tuple[str, str]:
    """Resolve the exact encoder referenced by a completed study selection."""
    reference_path = (root / "outputs/experiments/studies" / source_study / source_experiment / "selected_encoder.json")
    if not reference_path.is_file():
        raise FileNotFoundError(
            f"No completed encoder has been published for '{source_experiment}' in "
            f"study '{source_study}'. Submit that pretraining experiment first, "
            f"then retry. Expected: {reference_path}"
        )
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    return validate_published_encoder_reference(root, reference, reference_path, source_study, source_experiment)


def _pretraining_source(
    root: Path, settings: StudySettings, source_name: str, study_path: str
) -> tuple[Path, dict[str, Any]]:
    """Load a study's link-pretraining experiment, without resolving weights."""
    entry = settings.experiments.get(source_name)
    if entry is None:
        raise ValueError(f"pretrained_from must name an experiment in {study_path}.")

    config = entry if isinstance(entry, str) else entry.config
    if config is None:
        raise ValueError(f"Source experiment '{source_name}' needs a config path.")

    path = _inside_project(config, root)
    source = _read_yaml(path)
    if _pipeline(source, path) != "link_pretrain":
        raise ValueError(f"pretrained_from '{source_name}' must reference link pretraining.")
    return path, source


def _resolve_pretraining_dependency(
    root: Path, settings: StudySettings, study_path: str, source_name: str,
    source_study: str, joint: dict[str, Any], hydra_preset: bool, pending: bool
) -> tuple[Path, str, str]:
    """Validate the producing study entry and optionally find its published encoder."""
    source_path, source = _pretraining_source(root, settings, source_name, study_path)
    validate_source_graph(root, settings, source, joint, source_name, hydra_preset)
    validate_encoder_source(joint)

    if pending:
        return source_path, "", ""
    checkpoint, run_id = _selected_published_encoder(root, source_study, source_name)
    return source_path, checkpoint, run_id


def resolve_submission(
    root: Path, *, study: str, experiment_key: str, pending_pretrain: bool = False
) -> Submission:
    """Resolve a named study experiment and its pretraining dependency.

    Args:
        root: Repository root containing the study and its referenced inputs.
        study: Study catalog YAML, relative to the repository or absolute.
        experiment_key: Experiment name in the catalog.
        pending_pretrain: Permit a pending upstream pretraining job without weights.

    Returns:
        Resolved experiment, checkpoint source, and Slurm defaults.
    """
    root = root.resolve()
    if not study or not experiment_key:
        raise ValueError("--study and --experiment are required.")

    document, entry, study_name = _study_experiment(root, study, experiment_key)
    settings = document.study
    pretrained_from = entry.pretrained_from
    source_study_name = entry.source_study or study_name
    composed: dict[str, Any] | None = None
    job_name = ""
    pretrain_job_name = ""

    if entry.preset is not None:
        if settings.composition is None:
            raise ValueError("Hydra entries require study.composition to name the config directory.")
        config_dir = _inside_project(settings.composition, root)
        composed = _compose(root, config_dir, entry.preset)
        path = _inside_project(config_dir / "base.yaml", root)
        job_name, pretrain_job_name = _job_names(composed)
    else:
        if entry.config is None:
            raise ValueError(f"Study experiment '{experiment_key}' needs a config path.")
        path = _inside_project(entry.config, root)

    data = composed if composed is not None else _read_yaml(path)
    pipeline = _pipeline(data, path)
    if settings.composition and pipeline == "link_pretrain":
        # Named link pretraining entries reference standalone training YAMLs.
        # Their resolved configs can also be snapshotted by the submitter.
        composed = data
        graph = _read_yaml(_inside_project(data["paths"]["graph_config"], root))
        architecture = graph["model"]["gnn"]["architecture"]
        job_name = f"us-p-{architecture}-{data['pretraining']['objective']}"

    encoder_checkpoint = ""
    source_run = ""
    source_path: Path | None = None
    if pretrained_from is not None:
        validate_pretraining_dependency(pipeline, settings, study)
        source_path, encoder_checkpoint, source_run = _resolve_pretraining_dependency(
            root, settings, study, pretrained_from, source_study_name,
            data, composed is not None, pending_pretrain
        )

    validate_study_graph(root, settings.graph_config, data, experiment_key)

    hpc = document.hpc
    return Submission(
        config=path, pipeline=pipeline,
        partition=hpc.shell_value("partition"), gres=hpc.shell_value("gres"),
        time=hpc.shell_value("time"), memory=hpc.shell_value("memory"),
        cpus=hpc.shell_value("cpus_per_task"), mail_user=hpc.shell_value("mail_user"),
        mail_type=hpc.shell_value("mail_type"), study_name=study_name,
        experiment_key=experiment_key, exclude_nodes=",".join(hpc.exclude_nodes),
        encoder_checkpoint=encoder_checkpoint, source_run=source_run,
        pretrained_from=pretrained_from or "",
        pretrain_config=str(source_path) if source_path is not None else "",
        source_study_name=source_study_name,
        job_name=job_name or ("urban-syn-joint" if pipeline == "joint" else "urban-syn-pretrain"),
        pretrain_job_name=pretrain_job_name or "urban-syn-pretrain",
        composed_config=composed
    )


def main() -> int:
    """Emit fields for the existing Bash submitter without executing jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--study", required=True, help="Study catalog YAML")
    parser.add_argument("--experiment", required=True, help="Experiment name in the study")
    parser.add_argument("--field", choices=("pipeline", "config"))
    parser.add_argument("--snapshot", action="store_true", help="Save immutable Hydra submission files")
    parser.add_argument("--source-job-id", default="", help="Producing Slurm job ID for the snapshot")
    parser.add_argument("--selected-encoder", default="", help="Manual encoder reference for the snapshot")
    parser.add_argument("--pending-pretrain", action="store_true")
    args = parser.parse_args()
    try:
        selected = resolve_submission(
            args.project_root, study=args.study,
            experiment_key=args.experiment, pending_pretrain=args.pending_pretrain
        )
        if args.snapshot and selected.composed_config is not None:
            from submission_snapshot import save_submission_snapshot
            selected = save_submission_snapshot(
                args.project_root, selected, source_job_id=args.source_job_id,
                selected_encoder=args.selected_encoder
            )
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, yaml.YAMLError) as error:
        parser.exit(2, f"ERROR: {error}\n")

    if args.field:
        print(getattr(selected, args.field))
    else:
        for value in (
            selected.config, selected.pipeline, selected.partition, selected.gres,
            selected.time, selected.memory, selected.cpus, selected.mail_user,
            selected.mail_type, selected.study_name, selected.experiment_key,
            selected.exclude_nodes, selected.encoder_checkpoint, selected.source_run,
            selected.pretrained_from, selected.pretrain_config,
            selected.source_study_name, selected.job_name, selected.pretrain_job_name
        ):
            print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
