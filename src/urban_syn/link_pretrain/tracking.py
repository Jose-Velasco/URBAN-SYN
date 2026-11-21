"""Pretraining-specific configuration for the shared experiment tracking port.

The MLflow backend and run continuation are owned by ``experiment.tracking``.
Training and testing depend only on ``ExperimentTracker`` and ``TrackingRun``.
"""

from __future__ import annotations

import json
from pathlib import Path

from urban_syn.experiment.tracking import (
    ConfiguredMLflowTracker,
    ExperimentTracker,
    MLflowRunOptions,
    NullTracker,
    TrackingReference
)

from .config import ResolvedLinkPretrainConfig


def _flatten_settings(prefix: str, values: dict) -> dict[str, object]:
    """Convert nested experiment options into searchable dotted MLflow parameters.

    Args:
        prefix: Parameter namespace, such as ``pretraining``.
        values: JSON-compatible settings dictionary.

    Returns:
        Flat parameter mapping, preserving values for backend serialization.
    """
    flattened: dict[str, object] = {}
    for name, value in values.items():
        path = f"{prefix}.{name}"
        if isinstance(value, dict):
            flattened.update(_flatten_settings(path, value))
        else:
            flattened[path] = value
    return flattened


def build_pretraining_tracker(config: ResolvedLinkPretrainConfig, run_dir: Path) -> ExperimentTracker:
    """Build the configured tracking adapter without starting a run.

    Args:
        config: Resolved pretraining and GNN options.
        run_dir: Directory containing the configuration snapshots and outputs.

    Returns:
        Shared no-op or MLflow ``ExperimentTracker`` interface.
    """
    settings = config.settings
    if not settings.tracking.enabled:
        return NullTracker()

    parameters: dict[str, object] = {}
    sections = {
        "experiment": {"seed": settings.experiment.seed},
        "gnn": config.graph_settings["model"]["gnn"],
        "pretraining": settings.pretraining.model_dump(mode="json"),
        "training": settings.training.model_dump(
            mode="json", exclude={"checkpoint": {"resume_from"}}
        ),
        "runtime": settings.runtime.model_dump(mode="json"),
        "diagnostics": settings.tracking.logging.model_dump(mode="json")
    }
    for name, values in sections.items():
        parameters.update(_flatten_settings(name, values))

    tags = dict(settings.tracking.tags)
    tags.update({
        "pipeline": "road_link_pretraining",
        "gnn.architecture": str(config.graph_settings["model"]["gnn"].get("architecture", "gatv2")),
        "message_graph": "training_only",
        "experiment_config": settings.experiment.name,
        "run_directory": str(run_dir)
    })
    artifacts = tuple(
        (path, "configuration")
        for path in (
            run_dir / "config" / "experiment.yaml",
            run_dir / "config" / "graph.yaml",
            run_dir / "resolved_config.json"
        )
        if path.is_file()
    )
    options = MLflowRunOptions(
        experiment_name=settings.tracking.experiment_name,
        output_root=settings.paths.output_root,
        run_name=settings.tracking.run_name or run_dir.name,
        tracking_uri=settings.tracking.tracking_uri,
        storage_scope=settings.tracking.storage_scope,
        tags=tags,
        parameters=parameters,
        artifacts=artifacts
    )
    return ConfiguredMLflowTracker(options)


def read_tracking_reference(run_dir: Path) -> TrackingReference | None:
    """Read a legacy run reference, if available, for checkpoint compatibility.

    Args:
        run_dir: Directory containing the previous run's ``mlflow_run.json``.

    Returns:
        Stored backend-independent reference or None when no file exists.
    """
    path = run_dir / "mlflow_run.json"
    if not path.is_file():
        return None
    saved = json.loads(path.read_text(encoding="utf-8"))
    return {
        "provider": saved.get("provider", "mlflow"),
        "run_id": saved.get("run_id"),
        "uri": saved.get("uri", saved.get("tracking_uri"))
    }


def write_tracking_reference(run_dir: Path, reference: TrackingReference) -> None:
    """Persist the active run reference for tools using an older checkpoint.

    Args:
        run_dir: Existing run directory.
        reference: Provider-independent identity returned by the tracking port.
    """
    if reference["provider"] == "none":
        return
    path = run_dir / "mlflow_run.json"
    path.write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")
