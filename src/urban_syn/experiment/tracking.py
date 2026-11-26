"""Experiment-tracking port and MLflow/disabled adapters.

Training code depends only on TrackingRun and ExperimentTracker. MLflow is
imported lazily, and backend-specific run continuation stays inside its adapter.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict

from .config import ResolvedExperimentConfig


class TrackingReference(TypedDict):
    """Serializable identity of one run in a particular tracking backend.

    Attributes
    ----------
    provider : str
        Tracking backend identifier (for example, ``mlflow`` or ``none``).
    run_id : str | None
        Backend-assigned run identifier, if one exists.
    uri : str | None
        Backend location used to look up the recorded run.
    """

    provider: str
    run_id: str | None
    uri: str | None


class TrackingRun(Protocol):
    """Operations consumed by the training engine during an active run."""

    reference: TrackingReference
    run_id: str | None

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Record run parameters.

        Parameters
        ----------
        params : Mapping[str, Any]
            Parameters to record.
        """
        ...

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        """Record metrics at a specified training step.

        Parameters
        ----------
        metrics : Mapping[str, float]
            Numeric values to record.
        step : int
            Step associated with the values.
        """
        ...

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        """Save or upload an existing file as a run artifact.

        Parameters
        ----------
        path : str | Path
            Existing artifact file.
        artifact_path : str | None, default=None
            Destination group within the run.
        """
        ...

    def set_tags(self, tags: Mapping[str, str]) -> None:
        """Record searchable run metadata.

        Parameters
        ----------
        tags : Mapping[str, str]
            Key/value metadata to record.
        """
        ...


class ExperimentTracker(Protocol):
    """Open a tracking run, optionally continuing a previously recorded run."""

    def start_run(
        self,
        previous: TrackingReference | None = None
    ) -> AbstractContextManager[TrackingRun]:
        """Return a context manager for an active run.

        Parameters
        ----------
        previous : TrackingReference | None, default=None
            Reference saved in a training checkpoint, when resuming.

        Returns
        -------
        AbstractContextManager[TrackingRun]
            Context that manages backend-specific run lifecycle.
        """
        ...


class NullRun:
    """No-op run for training without an experiment-tracking backend."""

    def __init__(self):
        """Initialize an empty, provider-neutral run reference."""
        self.run_id: str | None = None
        self.reference: TrackingReference = {
            "provider": "none", "run_id": None, "uri": None
        }

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Ignore parameter logging when tracking is disabled.

        Parameters
        ----------
        params : Mapping[str, Any]
            Unused parameter values.
        """
        pass

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        """Ignore metric logging when tracking is disabled.

        Parameters
        ----------
        metrics : Mapping[str, float]
            Unused metrics.
        step : int
            Unused step index.
        """
        pass

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        """Ignore artifact logging when tracking is disabled.

        Parameters
        ----------
        path : str | Path
            Unused artifact location.
        artifact_path : str | None, default=None
            Unused destination path.
        """
        pass

    def set_tags(self, tags: Mapping[str, str]) -> None:
        """Ignore tags when tracking is disabled.

        Parameters
        ----------
        tags : Mapping[str, str]
            Unused metadata.
        """
        pass


class NullTracker:
    """Provide a working tracking context without any external dependencies."""

    @contextmanager
    def start_run(self, previous: TrackingReference | None = None) -> Iterator[TrackingRun]:
        """Open an untracked run, even when training resumes from a tracked run.

        Parameters
        ----------
        previous : TrackingReference | None, default=None
            Prior tracking reference; ignored when tracking is disabled.

        Yields
        ------
        TrackingRun
            No-op run that satisfies the tracking port.
        """
        yield NullRun()


class MLflowRun:
    """Adapt MLflow's run API to the provider-independent TrackingRun port."""

    def __init__(self, mlflow: Any, run_id: str, tracking_uri: str):
        """Create a run adapter for an active MLflow run.

        Parameters
        ----------
        mlflow : Any
            Imported MLflow module.
        run_id : str
            MLflow run identifier.
        tracking_uri : str
            Location of the MLflow tracking backend.
        """
        self._mlflow = mlflow
        self.run_id = run_id
        self.reference: TrackingReference = {
            "provider": "mlflow", "run_id": run_id, "uri": tracking_uri
        }

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Record flat parameters, serializing nested values as JSON strings.

        Parameters
        ----------
        params : Mapping[str, Any]
            Parameter names and values to log.

        Returns
        -------
        None
            Writes parameters into the active MLflow run.
        """
        self._mlflow.log_params({key: _param_string(value) for key, value in params.items()})

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        """Record numeric metrics at the supplied optimizer step.

        Parameters
        ----------
        metrics : Mapping[str, float]
            Numeric metric values.
        step : int
            Training step associated with the values.

        Returns
        -------
        None
            Writes metrics into the active MLflow run.
        """
        self._mlflow.log_metrics({key: float(value) for key, value in metrics.items()}, step=step)

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        """Record an existing file as a run artifact.

        Parameters
        ----------
        path : str | Path
            Local file to record.
        artifact_path : str | None, default=None
            Optional destination folder within the run's artifacts.

        Returns
        -------
        None
            Writes the artifact into the active MLflow run.
        """
        artifact = Path(path)
        if not artifact.is_file():
            raise FileNotFoundError(f"Cannot log missing artifact: {artifact}")
        self._mlflow.log_artifact(str(artifact), artifact_path=artifact_path)

    def set_tags(self, tags: Mapping[str, str]) -> None:
        """Record searchable string tags.

        Parameters
        ----------
        tags : Mapping[str, str]
            Tags associated with this run.

        Returns
        -------
        None
            Writes tags into the active MLflow run.
        """
        self._mlflow.set_tags(dict(tags))


def _param_string(value: Any) -> str:
    """Serialize a parameter value for the MLflow parameter API.

    Parameters
    ----------
    value : Any
        Scalar or JSON-like parameter value.

    Returns
    -------
    str
        Stable string representation.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)



def _flatten_parameters(prefix: str, values: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten nested configuration mappings into searchable parameter names.

    Parameters
    ----------
    prefix : str
        Namespace for the parameter names, such as ``training``.
    values : Mapping[str, Any]
        JSON-compatible configuration values.

    Returns
    -------
    dict[str, Any]
        Flat parameter mapping, with dotted keys for nested settings.
    """
    flattened: dict[str, Any] = {}
    for name, value in values.items():
        key = f"{prefix}.{name}"
        if isinstance(value, Mapping):
            flattened.update(_flatten_parameters(key, value))
        else:
            flattened[key] = value
    return flattened


def _experiment_parameters(config: ResolvedExperimentConfig) -> dict[str, Any]:
    """Collect effective model and training settings for the MLflow Parameters UI.

    Full resolved YAML remains a separate artifact. Exclude transient paths,
    tracking backend details, and ``resume_from`` because MLflow parameters
    cannot change within an existing run.

    Parameters
    ----------
    config : ResolvedExperimentConfig
        Experiment settings after YAML and graph configuration resolution.

    Returns
    -------
    dict[str, Any]
        Searchable settings for the GNN, diffusion model, training and sampling.
    """
    settings = config.settings
    gnn = config.joint_settings.gnn.model_dump(mode="json")

    # Only log options for the chosen architecture, not inactive alternatives.
    inactive_architecture = "gine" if gnn["architecture"] == "gatv2" else "gatv2"
    gnn.pop(inactive_architecture, None)

    sections = {
        "experiment": {"seed": settings.experiment.seed},
        "gnn": gnn,
        "diffusion": settings.model.diffusion.model_dump(mode="json"),
        "model": {
            "road_encoder": settings.model.road_encoder.model_dump(mode="json"),
            "conditioning": settings.model.conditioning.model_dump(mode="json")
        },
        "training": settings.training.model_dump(
            mode="json",
            exclude={"checkpoint": {"resume_from"}}
        ),
        "generation": settings.generation.model_dump(
            mode="json",
            exclude={"checkpoint"}
        ),
        "runtime": settings.runtime.model_dump(mode="json")
    }

    parameters: dict[str, Any] = {}
    for name, values in sections.items():
        parameters.update(_flatten_parameters(name, values))
    return parameters


def _slurm_store_suffix(scope: str) -> str | None:
    """Derive a job-specific store directory when requested by YAML.

    Parameters
    ----------
    scope : str
        Validated ``tracking.storage_scope`` value: ``job`` or ``shared``.

    Returns
    -------
    str | None
        Sanitized Slurm job/array identifier, or None for shared/local runs.
    """
    if scope == "shared":
        return None

    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return None
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    parts = [job_id]
    if task_id is not None:
        parts.append(task_id)
    return "_".join(
        "".join(char for char in part if char.isalnum() or char in "-_")
        for part in parts
    )


def _git_info(project_root: Path) -> dict[str, str]:
    """Read version-control metadata without requiring a Git checkout.

    Parameters
    ----------
    project_root : Path
        Directory containing the experiment's source code.

    Returns
    -------
    dict[str, str]
        Commit and dirty-state tags when available.
    """
    def git_command(*arguments: str) -> str | None:
        """Run one Git query and return its output on success.

        Parameters
        ----------
        *arguments : str
            Git CLI arguments.

        Returns
        -------
        str | None
            Git standard output, or None when unavailable.
        """
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=5,
                check=False
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = git_command("rev-parse", "HEAD")
    if commit is None:
        return {}
    status = git_command("status", "--porcelain")
    tags = {"git_commit": commit}
    if status is not None:
        tags["git_dirty"] = str(bool(status)).lower()
    return tags


def build_config_tags(config: ResolvedExperimentConfig) -> dict[str, str]:
    """Create compact, searchable tags from the resolved model configuration.

    Tag names are stable and do not depend on manually authored run labels.
    Only enabled losses with nonzero weight contribute to the active objective.
    """
    experiment = config.settings
    losses = experiment.training.losses
    choices = (
        ("noise_mse", losses.noise_mse),
        ("reconstruction_mse", losses.reconstruction_mse),
        ("hard_connectivity", losses.hard_connectivity),
    )
    included = [
        name for name, choice in choices
        if choice.enabled and choice.include_in_total and choice.weight > 0
    ]
    diagnostic = [
        name for name, choice in choices
        if choice.enabled and not (choice.include_in_total and choice.weight > 0)
    ]
    return {
        "gnn.architecture": str(config.joint_settings.gnn.architecture),
        "road_encoder.mode": experiment.model.road_encoder.mode,
        "road_encoder.initialization": experiment.model.road_encoder.initialization,
        "road_encoder.trainable": str(experiment.model.road_encoder.trainable).lower(),
        "losses.in_total": ",".join(included),
        "losses.diagnostic_only": ",".join(diagnostic),
        "optimizer": experiment.training.optimizer.name,
        "scheduler": experiment.training.scheduler.name,
        "validation.enabled": str(experiment.training.validation.enabled).lower(),
        "conditioning.enabled": str(experiment.model.conditioning.enabled).lower(),
    }


def execution_hardware_tags() -> dict[str, str]:
    """Read the GPU assigned by Slurm and optional study information.

    The Slurm wrapper writes its detected GPU to a JSON file before launching
    Python. This avoids guessing that CUDA index 0 is physical GPU index 0.
    Missing metadata is normal for local training and CPU-only tests.

    Returns:
        String tags safe to attach to any experiment-tracking run.
    """
    tags: dict[str, str] = {}
    for environment, key in (
        ("SLURM_JOB_ID", "slurm.job_id"),
        ("SLURMD_NODENAME", "slurm.node"),
        ("URBAN_SYN_STUDY_NAME", "study.name"),
        ("URBAN_SYN_EXPERIMENT_KEY", "study.experiment")
    ):
        value = os.environ.get(environment)
        if value:
            tags[key] = value

    metadata_path = os.environ.get("URBAN_SYN_GPU_METADATA")
    if not metadata_path:
        return tags
    try:
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return tags
    if not isinstance(metadata, dict):
        return tags

    fields = (
        "gpu.name", "gpu.physical_index", "gpu.uuid", "gpu.cuda_device",
        "slurm.job_id", "slurm.node", "slurm.cuda_visible_devices"
    )
    for key in fields:
        value = metadata.get(key)
        if value is not None and str(value):
            tags[key] = str(value)
    return tags


@dataclass(frozen=True)
class MLflowRunOptions:
    """Backend-neutral values required to create an MLflow experiment run.

    Args:
        experiment_name: MLflow experiment under which the run is recorded.
        output_root: Root for ordinary experiment outputs.
        store_root: Optional independent storage root; overrides MLFLOW_TRACKING_URI.
        run_name: Human-readable run label.
        tracking_uri: Explicit backend URI; takes priority over all storage defaults.
        storage_scope: ``shared`` storage or separate storage for Slurm jobs.
        tags: Searchable metadata supplied by the calling training pipeline.
        parameters: Flat experiment parameters logged once for a new run.
        artifacts: Existing files with optional destination directories.
        documents: Mappings saved through MLflow as JSON artifacts.
        log_system_metrics: Whether MLflow should record system telemetry.
    """

    experiment_name: str
    output_root: Path
    run_name: str
    tracking_uri: str | None = None
    storage_scope: str = "shared"
    store_root: Path | None = None
    tags: Mapping[str, str] = field(default_factory=dict)
    parameters: Mapping[str, Any] = field(default_factory=dict)
    artifacts: tuple[tuple[Path, str | None], ...] = ()
    documents: tuple[tuple[str, Mapping[str, Any]], ...] = ()
    log_system_metrics: bool = False


class ConfiguredMLflowTracker:
    """MLflow adapter reused by independent training pipelines.

    Pipeline-specific configuration parsing happens before constructing this
    adapter. Trainers depend only on ``ExperimentTracker`` and ``TrackingRun``.
    """

    def __init__(self, options: MLflowRunOptions):
        """Resolve backend storage without importing MLflow or creating files.

        Args:
            options: Validated run location, parameters, tags, and artifacts.
        """
        self.options = options
        root = options.store_root if options.store_root is not None else options.output_root / "mlflow"
        suffix = _slurm_store_suffix(options.storage_scope)
        if suffix is not None:
            root = root / "jobs" / suffix
        self.store_root = root
        self.artifacts_root = root / "artifacts"
        environment_uri = os.environ.get("MLFLOW_TRACKING_URI") if options.store_root is None else None
        self.tracking_uri = (
            options.tracking_uri
            or environment_uri
            or f"sqlite:///{(root / 'tracking.db').as_posix()}"
        )

    def _configure(self, mlflow: Any) -> None:
        """Select a tracking URI and initialize local artifact storage as needed."""
        mlflow.set_tracking_uri(self.tracking_uri)
        if self.tracking_uri.startswith("sqlite:///"):
            self.artifacts_root.mkdir(parents=True, exist_ok=True)
            if mlflow.get_experiment_by_name(self.options.experiment_name) is None:
                mlflow.create_experiment(
                    self.options.experiment_name,
                    artifact_location=self.artifacts_root.resolve().as_uri()
                )
        mlflow.set_experiment(self.options.experiment_name)

    @contextmanager
    def start_run(self, previous: TrackingReference | None = None) -> Iterator[TrackingRun]:
        """Open or resume a run, preserving the backend-independent run reference.

        Args:
            previous: Optional reference stored in an earlier checkpoint.

        Yields:
            Active ``TrackingRun`` adapter. Parameters and configuration files
            are only recorded for a newly created run.

        Raises:
            RuntimeError: If MLflow is enabled but not installed.
        """
        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError("MLflow tracking is enabled; install mlflow in the container.") from exc

        self._configure(mlflow)
        resume_id = None
        if previous is not None:
            if previous.get("provider") == "mlflow" and previous.get("uri") == self.tracking_uri:
                resume_id = previous.get("run_id")

        if resume_id is None:
            context = mlflow.start_run(
                run_name=self.options.run_name,
                tags=dict(self.options.tags),
                log_system_metrics=self.options.log_system_metrics
            )
        else:
            context = mlflow.start_run(
                run_id=resume_id,
                log_system_metrics=self.options.log_system_metrics
            )

        with context as active:
            run = MLflowRun(mlflow, active.info.run_id, self.tracking_uri)
            # Attach the actual GPU allocation for *every* invocation, including
            # standalone generation and resumed training on a different node.
            hardware = execution_hardware_tags()
            if hardware:
                run.set_tags(hardware)
            metadata_file = os.environ.get("URBAN_SYN_GPU_METADATA")
            if metadata_file and Path(metadata_file).is_file():
                run.log_artifact(metadata_file, "hardware")
            if resume_id is None:
                run.log_params(self.options.parameters)
                if previous is not None and previous.get("run_id"):
                    run.set_tags({
                        "resumed_from_provider": previous.get("provider") or "unknown",
                        "resumed_from_run_id": previous["run_id"]
                    })
                for path, destination in self.options.artifacts:
                    if path.is_file():
                        run.log_artifact(path, destination)
                for destination, content in self.options.documents:
                    mlflow.log_dict(dict(content), destination)
            yield run


class MLflowTracker:
    """Configure the shared MLflow adapter for joint diffusion training.

    Existing callers still provide ``ResolvedExperimentConfig``. New training
    pipelines may use ``ConfiguredMLflowTracker`` directly.
    """

    def __init__(self, config: ResolvedExperimentConfig):
        """Retain the joint experiment's storage, tags, and configuration rules."""
        self.config = config
        settings = config.settings.tracking
        self.experiment_name = settings.experiment_name
        self._backend = ConfiguredMLflowTracker(MLflowRunOptions(
            experiment_name=settings.experiment_name,
            output_root=config.settings.paths.output_root,
            run_name=settings.run_name or (
                f"{config.settings.experiment.name}_{config.joint_settings.gnn.architecture}"
            ),
            tracking_uri=settings.tracking_uri,
            storage_scope=settings.storage_scope,
            store_root=settings.store_root,
            tags=self._run_tags(),
            parameters=_experiment_parameters(config),
            documents=(("configuration/resolved_experiment.json", {
                "experiment": config.settings.model_dump(mode="json"),
                "model_settings": config.joint_settings.model_dump(mode="json"),
                "project_root": str(config.project_root),
                "source_yaml": str(config.config_path)
            }),) if settings.logging.log_resolved_config else (),
            log_system_metrics=settings.logging.log_system_metrics
        ))
        self.store_root = self._backend.store_root
        self.artifacts_root = self._backend.artifacts_root
        self.tracking_uri = self._backend.tracking_uri

    def _configure(self, mlflow: Any) -> None:
        """Configure the backend using the same shared implementation."""
        self._backend._configure(mlflow)

    def _run_tags(self) -> dict[str, str]:
        """Combine experiment tags with Git and Slurm metadata."""
        tags = dict(self.config.settings.tracking.tags)
        tags.update(build_config_tags(self.config))
        tags["experiment_config"] = self.config.settings.experiment.name
        tags["mlflow_storage_scope"] = self.config.settings.tracking.storage_scope
        for environment, tag in (
            ("SLURM_JOB_ID", "slurm_job_id"),
            ("SLURM_ARRAY_TASK_ID", "slurm_array_task_id")
        ):
            value = os.environ.get(environment)
            if value:
                tags[tag] = value
        if self.config.settings.tracking.logging.log_git_info:
            tags.update(_git_info(self.config.project_root))
        return tags

    @contextmanager
    def start_run(self, previous: TrackingReference | None = None) -> Iterator[TrackingRun]:
        """Start or continue a joint-training run through the shared adapter."""
        with self._backend.start_run(previous=previous) as run:
            yield run


def build_tracker(config: ResolvedExperimentConfig) -> ExperimentTracker:
    """Construct the configured tracking adapter without starting a run.

    Parameters
    ----------
    config : ResolvedExperimentConfig
        Validated experiment configuration.

    Returns
    -------
    ExperimentTracker
        MLflow adapter or disabled-tracking implementation.
    """
    if not config.settings.tracking.enabled:
        return NullTracker()
    return MLflowTracker(config)
