"""Validated YAML settings for joint GNN + Diff-RNTraj experiments.

This module only loads configuration. It does not construct the graph/model,
initialize MLflow, or start training. Existing graph YAML supplies graph construction and GNN settings.
The experiment YAML owns Diff-RNTraj hyperparameters and training settings.

All relative paths in an experiment YAML are relative to ``project_root``
(default: the caller's working directory). This convention matches the planned
Docker/Apptainer bind of the repository root at ``/workspace``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from urban_syn.diffusion.config import (
    DiffusionSettings,
    JointTrainingSettings,
    TrainingSettings,
)
from urban_syn.gnn.config import GNNSettings


class _Settings(BaseModel):
    """Reject unknown keys and keep parsed configuration immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class ExperimentInfo(_Settings):
    name: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
    seed: int = Field(default=101, ge=0)


class ExperimentPaths(_Settings):
    """Paths to pre-existing artifacts; existence is checked by later stages."""

    graph_config: Path
    train_sequences: Path
    test_sequences: Path | None = None
    trip_id_map: Path | None = None
    road_network: Path | None = None
    length_distribution: Path | None = None
    test_reference_gps: Path | None = None
    output_root: Path = Path("outputs/experiments")


class RoadEncoderSettings(_Settings):
    """V1 supports only an unfrozen jointly trained road encoder."""

    mode: Literal["joint"] = "joint"
    # Pretraining/frozen modes will be admitted when their code exists.


class ConditioningSettings(_Settings):
    """Reserve a stable toggle without pretending conditioning is implemented."""

    enabled: bool = False

    @model_validator(mode="after")
    def reject_unsupported_conditioning(self) -> Self:
        if self.enabled:
            raise ValueError("Metadata conditioning is not supported in the V1 joint trainer yet.")
        return self


class ModelOptions(_Settings):
    """Experiment-owned diffusion architecture and optional model behavior.

    The GNN architecture stays in the referenced graph YAML, but diffusion
    hyperparameters must be explicitly configured per experiment.
    """

    diffusion: DiffusionSettings
    road_encoder: RoadEncoderSettings = Field(default_factory=RoadEncoderSettings)
    conditioning: ConditioningSettings = Field(default_factory=ConditioningSettings)


class OptimizerSettings(_Settings):
    name: Literal["adam", "adamw"] = "adamw"
    learning_rate: float = Field(default=1e-3, gt=0, allow_inf_nan=False)
    weight_decay: float = Field(default=0.01, ge=0, allow_inf_nan=False)


class SchedulerSettings(_Settings):
    """Scheduler selection. Training implementation will interpret parameters."""

    name: Literal["none", "cosine", "reduce_on_plateau"] = "none"
    min_lr: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    patience_epochs: int = Field(default=5, ge=1)


class WeightedLossSettings(_Settings):
    """Options shared by individually selectable loss components."""

    enabled: bool = True
    weight: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    include_in_total: bool = True
    every_steps: int = Field(default=1, ge=1)


class LossSettings(_Settings):
    """Currently available loss implementations and their selection policy."""

    noise_mse: WeightedLossSettings = Field(default_factory=WeightedLossSettings)
    reconstruction_mse: WeightedLossSettings = Field(default_factory=WeightedLossSettings)
    hard_connectivity: WeightedLossSettings = Field(default_factory=WeightedLossSettings)

    @model_validator(mode="after")
    def validate_objectives(self) -> Self:
        """Require a differentiable objective and stable total-loss cadence.

        Returns
        -------
        Self
            Validated settings.
        """
        choices = {
            "noise_mse": self.noise_mse,
            "reconstruction_mse": self.reconstruction_mse,
            "hard_connectivity": self.hard_connectivity
        }
        for name, choice in choices.items():
            if choice.enabled and choice.include_in_total and choice.every_steps != 1:
                raise ValueError(f"{name} contributes to total and must run every step.")

        gradient_losses = (self.noise_mse, self.reconstruction_mse)
        if not any(
            item.enabled and item.include_in_total and item.weight > 0
            for item in gradient_losses
        ):
            raise ValueError("At least one differentiable loss must have positive weight in total.")
        return self


class ValidationSettings(_Settings):
    enabled: bool = False
    fraction: float = Field(default=0.1, gt=0, lt=1, allow_inf_nan=False)
    split_group_columns: tuple[str, ...] = ("uid", "tid")

    @model_validator(mode="after")
    def require_original_trip_groups(self) -> Self:
        # Current NYC FMM pieces share their original uid/tid trip identity.
        if self.split_group_columns != ("uid", "tid"):
            raise ValueError("The current NYC validation split must use [uid, tid].")
        return self


class CheckpointSettings(_Settings):
    monitor: Literal["auto", "train_loss", "val_loss"] = "auto"
    save_last: bool = True
    save_best: bool = True
    every_epochs: int = Field(default=1, ge=1)
    resume_from: Path | None = None


class TrainingExperimentSettings(_Settings):
    """Experiment controls; convertible to the existing training settings.

    The original ``TrainingSettings`` keeps scalar learning_rate/weight_decay
    for compatibility with the old CLI. V1 experiment YAML has one source of
    truth under ``training.optimizer`` and is converted explicitly below.
    """

    epochs: int = Field(default=30, gt=0)
    batch_size: int = Field(default=256, gt=0)
    max_batches: int | None = Field(default=None, gt=0)
    grad_clip_norm: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    optimizer: OptimizerSettings = Field(default_factory=OptimizerSettings)
    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)
    losses: LossSettings = Field(default_factory=LossSettings)
    validation: ValidationSettings = Field(default_factory=ValidationSettings)
    checkpoint: CheckpointSettings = Field(default_factory=CheckpointSettings)

    def to_legacy_settings(self) -> TrainingSettings:
        """Adapt the new YAML to the existing training loop's parameter API."""
        return TrainingSettings(
            epochs=self.epochs,
            batch_size=self.batch_size,
            max_batches=self.max_batches,
            grad_clip_norm=self.grad_clip_norm,
            learning_rate=self.optimizer.learning_rate,
            weight_decay=self.optimizer.weight_decay
        )

    @model_validator(mode="after")
    def validate_checkpoint_monitor(self) -> Self:
        if self.checkpoint.monitor == "val_loss" and not self.validation.enabled:
            raise ValueError("checkpoint.monitor=val_loss requires validation.enabled=true.")
        if self.scheduler.min_lr >= self.optimizer.learning_rate and self.scheduler.name == "cosine":
            raise ValueError("cosine scheduler min_lr must be smaller than optimizer.learning_rate.")
        return self


class TrackingLoggingSettings(_Settings):
    metrics_every_steps: int = Field(default=100, ge=1)
    gnn_diagnostics_every_steps: int = Field(default=100, ge=1)
    gnn_embedding_probe_size: int = Field(default=128, ge=1)
    log_system_metrics: bool = False
    log_resolved_config: bool = True
    log_git_info: bool = True


class TrackingArtifactSettings(_Settings):
    log_best_checkpoint: bool = True
    log_last_checkpoint: bool = False
    log_generated_trajectories: bool = True
    log_training_logs: bool = True


class TrackingSettings(_Settings):
    """MLflow client settings, not tracking-server database configuration."""

    enabled: bool = True
    provider: Literal["mlflow"] = "mlflow"
    # None delegates to MLFLOW_TRACKING_URI or a runtime-specific default.
    tracking_uri: str | None = None
    experiment_name: str = Field(default="gnn_diff_rntraj_nyc", min_length=1)
    run_name: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    logging: TrackingLoggingSettings = Field(default_factory=TrackingLoggingSettings)
    artifacts: TrackingArtifactSettings = Field(default_factory=TrackingArtifactSettings)


class DDPMSamplerSettings(_Settings):
    """DDPM sampling options for the experiment generation stage.

    DDPM uses the diffusion schedule saved with the trained model; it has
    no independent inference-step or solver-order option in this version.
    Other samplers can introduce their own validated settings later.
    """

    name: Literal["ddpm"] = "ddpm"


class GenerationSettings(_Settings):
    """Generation options; standalone generation requires a checkpoint path."""

    enabled: bool = False
    checkpoint: Path | None = None
    checkpoint_choice: Literal["best", "last"] = "best"
    road_layer: str = Field(default="roads", min_length=1)
    output_name: str = Field(default="nyc_generated_gps.parquet", pattern=r"^[^/\\]+\.parquet$")
    num_trajectories: int | None = Field(default=None, ge=1)
    batch_size: int = Field(default=2, ge=1)
    sampler: DDPMSamplerSettings = Field(default_factory=DDPMSamplerSettings)


class RuntimeSettings(_Settings):
    device: Literal["auto", "cpu", "cuda"] = "auto"
    precision: Literal["fp32"] = "fp32"  # AMP added after supporting/tested.
    num_workers: int = Field(default=4, ge=0)


class ExperimentSettings(_Settings):
    """Top-level experiment YAML schema; no graph/model objects are built."""

    experiment: ExperimentInfo
    paths: ExperimentPaths
    model: ModelOptions
    training: TrainingExperimentSettings = Field(default_factory=TrainingExperimentSettings)
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)

    @model_validator(mode="after")
    def validate_cross_section_requirements(self) -> Self:
        if self.training.validation.enabled and self.paths.trip_id_map is None:
            raise ValueError("Validation requires paths.trip_id_map to group by original trip.")
        return self


@dataclass(frozen=True)
class ResolvedExperimentConfig:
    """Parsed settings, absolute paths, and reused architecture settings."""

    settings: ExperimentSettings
    joint_settings: JointTrainingSettings
    config_path: Path
    project_root: Path


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    """Read a YAML mapping with a useful error for empty/incorrect roots."""
    with path.open("r", encoding="utf-8") as source:
        value = yaml.safe_load(source)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping at the root of {path}.")
    return value


def _absolute(path: Path, root: Path) -> Path:
    """Resolve a user path relative to the configured project root."""
    return (path if path.is_absolute() else root / path).expanduser().resolve()


def _resolve_paths(paths: ExperimentPaths, root: Path) -> ExperimentPaths:
    """Resolve only declared path fields without changing experiment values."""
    return paths.model_copy(
        update={
            name: _absolute(value, root) if value is not None else None
            for name, value in paths
        }
    )


def _read_model_settings(
    graph_path: Path, model_options: ModelOptions, training: TrainingExperimentSettings
) -> JointTrainingSettings:
    """Combine graph-owned GNN and experiment-owned diffusion settings.

    The graph YAML's legacy ``model.diffusion`` and ``training`` sections are
    ignored. They may remain temporarily for the old smoke-test entry point.
    """
    graph_yaml = _read_yaml_mapping(graph_path)
    if not isinstance(graph_yaml.get("dataset"), dict) or not isinstance(graph_yaml.get("features"), dict):
        raise ValueError("Referenced graph YAML must contain dataset and features mappings.")
    model = graph_yaml.get("model")
    if not isinstance(model, dict):
        raise ValueError("Referenced graph YAML must contain a model mapping.")
    if "gnn" not in model:
        raise ValueError("Referenced graph YAML requires model.gnn.")

    return JointTrainingSettings(
        gnn=GNNSettings.model_validate(model["gnn"]),
        diffusion=model_options.diffusion,
        training=training.to_legacy_settings()
    )


def load_experiment_config(
    path: str | Path,
    project_root: str | Path | None = None
) -> ResolvedExperimentConfig:
    """Validate a joint experiment and its referenced model architecture.

    Parameters
    ----------
    path:
        Experiment YAML path. Relative values use the caller's working directory.
    project_root:
        Repository root for resolving all paths *inside* the experiment YAML.
        Defaults to the current working directory, e.g. /workspace in Docker.

    Notes
    -----
    This validates structure and the referenced graph YAML. Existence of data,
    graph cache, checkpoint, etc. belongs to the pipeline's later validate stage.
    No training, MLflow initialization, or heavy graph construction occurs.
    """
    source_path = Path(path).expanduser().resolve()
    root = Path(project_root).expanduser().resolve() if project_root is not None else Path.cwd().resolve()
    raw = _read_yaml_mapping(source_path)
    parsed = ExperimentSettings.model_validate(raw)
    resolved = parsed.model_copy(update={"paths": _resolve_paths(parsed.paths, root)})
    joint = _read_model_settings(
        resolved.paths.graph_config, resolved.model, resolved.training
    )

    return ResolvedExperimentConfig(
        settings=resolved,
        joint_settings=joint,
        config_path=source_path,
        project_root=root
    )
