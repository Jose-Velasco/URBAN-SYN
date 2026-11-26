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
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from urban_syn.diffusion.config import (
    DiffusionSettings,
    JointTrainingSettings,
    TrainingSettings,
)
from urban_syn.gnn.config import GNNSettings
from urban_syn.link_pretrain.config import NegativeSamplingSettings


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


class GradientWarmupSettings(_Settings):
    """Configure diffusion-to-GNN gradient scaling during joint training.

    The schedule follows the normalized sigmoid from LDLM, Appendix A.2.
    A disabled schedule leaves the existing GNN training behavior unchanged.
    """

    enabled: bool = False
    schedule: Literal["normalized_sigmoid"] = "normalized_sigmoid"
    warmup_steps: int = Field(default=10000, ge=1)
    gamma_min: float = Field(default=0.0, ge=0, lt=1, allow_inf_nan=False)
    k: float = Field(default=10.0, gt=0, allow_inf_nan=False)
    c: float = Field(default=0.8, gt=0, lt=1, allow_inf_nan=False)


class RoadEncoderSettings(_Settings):
    """Select the source of GNN weights and whether diffusion updates them.

    ``mode: joint`` is retained so existing experiment YAMLs still load.
    It does not determine how weights are initialized or trained.
    """

    mode: Literal["joint"] = "joint"
    initialization: Literal["random", "pretrained"] = "random"
    trainable: bool = True
    checkpoint: Path | None = None
    gradient_warmup: GradientWarmupSettings = Field(default_factory=GradientWarmupSettings)

    @model_validator(mode="after")
    def validate_initialization(self) -> Self:
        """Reject ambiguous checkpoint and trainability combinations."""
        if self.initialization == "pretrained" and self.checkpoint is None:
            raise ValueError(
                "model.road_encoder.checkpoint is required when "
                "initialization=pretrained (use the pretraining run's encoder.pt)."
            )
        if self.initialization == "random" and self.checkpoint is not None:
            raise ValueError(
                "model.road_encoder.checkpoint must be null when initialization=random."
            )
        if self.initialization == "random" and not self.trainable:
            raise ValueError(
                "A randomly initialized, frozen road encoder is not a supported "
                "comparison. Use initialization=pretrained or trainable=true."
            )
        if self.gradient_warmup.enabled and not self.trainable:
            raise ValueError(
                "model.road_encoder.gradient_warmup requires a trainable GNN."
            )
        return self


class ConnectivityScorerSettings(_Settings):
    """Load a frozen pretrained road-pair scorer for joint experiments.

    ``checkpoint: auto`` selects ``best.pt`` next to the configured
    pretrained ``encoder.pt``. An explicit path is also supported. The scorer
    is not optimized during the initial connectivity-guidance experiments.
    """

    enabled: bool = False
    checkpoint: Literal["auto"] | Path = "auto"
    trainable: bool = False

    @model_validator(mode="after")
    def reject_scorer_finetuning(self) -> Self:
        if self.trainable:
            raise ValueError(
                "Scorer fine-tuning is not implemented. Set "
                "model.connectivity_scorer.trainable=false."
            )
        return self


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
    connectivity_scorer: ConnectivityScorerSettings = Field(
        default_factory=ConnectivityScorerSettings
    )
    conditioning: ConditioningSettings = Field(default_factory=ConditioningSettings)

    @model_validator(mode="after")
    def validate_scorer_encoder(self) -> Self:
        if (self.connectivity_scorer.enabled
                and self.road_encoder.initialization == "random"
                and self.connectivity_scorer.checkpoint == "auto"):
            raise ValueError(
                "A randomly initialized GNN with connectivity guidance requires "
                "a pretrained scorer checkpoint (best.pt)."
            )
        return self


class OptimizerSettings(_Settings):
    name: Literal["adam", "adamw"] = "adamw"
    learning_rate: float = Field(default=1e-3, gt=0, allow_inf_nan=False)
    weight_decay: float = Field(default=0.01, ge=0, allow_inf_nan=False)


NonNegativeLearningRate = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class SchedulerSettings(_Settings):
    """Configure an epoch-level learning-rate scheduler.

    ReduceLROnPlateau options follow PyTorch's constructor. The deprecated
    ``patience_epochs`` key is accepted in older YAML files and normalized to
    ``patience`` so existing experiments keep their original behavior.
    """

    name: Literal["none", "cosine", "reduce_on_plateau"] = "none"
    min_lr: NonNegativeLearningRate | list[NonNegativeLearningRate] = 0.0

    # ReduceLROnPlateau settings (ignored when the scheduler is not selected).
    mode: Literal["min", "max"] = "min"
    factor: float = Field(default=0.1, gt=0, lt=1, allow_inf_nan=False)
    patience: int = Field(default=5, ge=0)
    threshold: float = Field(default=1e-4, ge=0, allow_inf_nan=False)
    threshold_mode: Literal["rel", "abs"] = "rel"
    cooldown: int = Field(default=0, ge=0)
    eps: float = Field(default=1e-8, ge=0, allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_patience(cls, value: Any) -> Any:
        """Accept older ``patience_epochs`` configs without duplicate settings.

        Parameters
        ----------
        value : Any
            Incoming scheduler settings.

        Returns
        -------
        Any
            Settings with the legacy key migrated, if present.
        """
        if not isinstance(value, dict) or "patience_epochs" not in value:
            return value
        if "patience" in value:
            raise ValueError("Specify either scheduler.patience or patience_epochs, not both.")
        migrated = dict(value)
        migrated["patience"] = migrated.pop("patience_epochs")
        return migrated

    @model_validator(mode="after")
    def validate_min_lr_shape(self) -> Self:
        """Restrict cosine annealing to one scalar minimum learning rate."""
        if isinstance(self.min_lr, list):
            if not self.min_lr:
                raise ValueError("scheduler.min_lr list cannot be empty.")
            if self.name == "cosine":
                raise ValueError("cosine scheduler requires a scalar min_lr.")
        return self


class WeightedLossSettings(_Settings):
    """Options shared by individually selectable loss components."""

    enabled: bool = True
    weight: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    include_in_total: bool = True
    every_steps: int = Field(default=1, ge=1)


class OptionalLossSettings(WeightedLossSettings):
    """Weighted loss disabled unless explicitly requested by an experiment."""

    enabled: bool = False


class BPRLossSettings(OptionalLossSettings):
    """Ranking guidance with reusable graph-aware negative-sampling settings."""

    negatives: NegativeSamplingSettings = Field(default_factory=NegativeSamplingSettings)


class LossSettings(_Settings):
    """Currently available loss implementations and their selection policy."""

    noise_mse: WeightedLossSettings = Field(default_factory=WeightedLossSettings)
    reconstruction_mse: WeightedLossSettings = Field(default_factory=WeightedLossSettings)
    hard_connectivity: WeightedLossSettings = Field(default_factory=WeightedLossSettings)
    connectivity_bce: OptionalLossSettings = Field(default_factory=OptionalLossSettings)
    connectivity_bpr: BPRLossSettings = Field(default_factory=BPRLossSettings)

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
            "hard_connectivity": self.hard_connectivity,
            "connectivity_bce": self.connectivity_bce,
            "connectivity_bpr": self.connectivity_bpr
        }
        if self.connectivity_bce.enabled and self.connectivity_bpr.enabled:
            raise ValueError("Enable either connectivity_bce or connectivity_bpr, not both.")
        for name, choice in choices.items():
            if choice.enabled and choice.include_in_total and choice.every_steps != 1:
                raise ValueError(f"{name} contributes to total and must run every step.")

        gradient_losses = (
            self.noise_mse, self.reconstruction_mse,
            self.connectivity_bce, self.connectivity_bpr
        )
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
        if self.scheduler.name == "cosine":
            # Scalar min_lr is guaranteed by SchedulerSettings validation.
            if self.scheduler.min_lr >= self.optimizer.learning_rate:
                raise ValueError("cosine scheduler min_lr must be smaller than optimizer.learning_rate.")
        return self


class ScorerDiagnosticsSettings(_Settings):
    """Read-only graph-link checks for the frozen pretrained scorer."""

    enabled: bool = False
    every_epochs: int = Field(default=10, ge=1)
    sample_size: int = Field(default=256, ge=1)
    chunk_size: int = Field(default=128, ge=1)
    negatives: NegativeSamplingSettings = Field(default_factory=NegativeSamplingSettings)


class TrackingLoggingSettings(_Settings):
    scorer: ScorerDiagnosticsSettings = Field(default_factory=ScorerDiagnosticsSettings)
    metrics_every_steps: int = Field(default=100, ge=1)
    # 0 disables gradient diagnostics. Step metrics remain unaffected.
    gnn_diagnostics_every_steps: int = Field(default=100, ge=0)
    # Embedding diagnostics have their own cadence, independent of checkpoint writes.
    gnn_embedding_enabled: bool = True
    gnn_embedding_every_epochs: int = Field(default=1, ge=1)
    gnn_embedding_sample_size: int = Field(default=128, ge=1)
    gnn_embedding_source: Literal["graph", "trajectories"] = "graph"

    @model_validator(mode="before")
    @classmethod
    def accept_previous_sample_name(cls, value: Any) -> Any:
        """Read existing experiment YAMLs while using the clearer setting name."""
        if not isinstance(value, dict) or "gnn_embedding_probe_size" not in value:
            return value
        if "gnn_embedding_sample_size" in value:
            raise ValueError("Use only tracking.logging.gnn_embedding_sample_size.")
        updated = dict(value)
        updated["gnn_embedding_sample_size"] = updated.pop("gnn_embedding_probe_size")
        return updated
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
    # Independently locate the SQLite database and MLflow artifact store.
    store_root: Path | None = None
    # Choose the default local SQLite layout when no explicit URI is provided.
    # Shared SQLite is the default; job scope isolates Slurm jobs when selected.
    storage_scope: Literal["job", "shared"] = "shared"
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
    prefetch_factor: int = Field(default=2, ge=1)
    persistent_workers: bool = True
    pin_memory: bool = True


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
        if self.tracking.logging.scorer.enabled and not self.model.connectivity_scorer.enabled:
            raise ValueError("Scorer diagnostics require model.connectivity_scorer.enabled=true.")
        if (
            self.training.losses.connectivity_bce.enabled
            or self.training.losses.connectivity_bpr.enabled
        ) and not self.model.connectivity_scorer.enabled:
            raise ValueError(
                "Connectivity guidance requires model.connectivity_scorer.enabled=true."
            )
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


def _resolve_model_options(options: ModelOptions, root: Path) -> ModelOptions:
    """Resolve explicit pretrained paths using the shared project root."""
    selected = options.road_encoder
    road_encoder = selected
    if selected.checkpoint is not None:
        road_encoder = selected.model_copy(
            update={"checkpoint": _absolute(selected.checkpoint, root)}
        )

    scorer = options.connectivity_scorer
    if isinstance(scorer.checkpoint, Path):
        scorer = scorer.model_copy(
            update={"checkpoint": _absolute(scorer.checkpoint, root)}
        )
    return options.model_copy(update={
        "road_encoder": road_encoder,
        "connectivity_scorer": scorer
    })


def _resolve_tracking(tracking: TrackingSettings, root: Path) -> TrackingSettings:
    """Resolve an explicitly configured MLflow store independently of run outputs."""
    if tracking.store_root is None:
        return tracking
    return tracking.model_copy(update={"store_root": _absolute(tracking.store_root, root)})


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
    project_root: str | Path | None = None,
    encoder_checkpoint: str | Path | None = None,
    generation_checkpoint: str | Path | None = None
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
    An optional encoder checkpoint passed by the study launcher replaces a
    null YAML checkpoint before normal settings validation.
    """
    source_path = Path(path).expanduser().resolve()
    root = Path(project_root).expanduser().resolve() if project_root is not None else Path.cwd().resolve()
    raw = _read_yaml_mapping(source_path)
    if encoder_checkpoint is not None:
        model = raw.get("model")
        encoder = model.get("road_encoder") if isinstance(model, dict) else None
        if not isinstance(encoder, dict):
            raise ValueError("--encoder-checkpoint requires model.road_encoder settings.")
        if encoder.get("checkpoint") is not None:
            raise ValueError(
                "Checkpoint is specified both in the joint YAML and by the launcher. "
                "Set model.road_encoder.checkpoint: null when selecting it at submission."
            )
        # Copy only the edited nested dictionaries, leaving the input YAML
        # and its other settings unchanged for reproducible run snapshots.
        raw = dict(raw)
        raw["model"] = dict(model)
        if encoder.get("initialization") == "pretrained":
            raw["model"]["road_encoder"] = dict(encoder)
            raw["model"]["road_encoder"]["checkpoint"] = str(encoder_checkpoint)
        elif encoder.get("initialization") == "random":
            scorer = model.get("connectivity_scorer") or {}
            source = Path(encoder_checkpoint)
            if not scorer.get("enabled") or scorer.get("checkpoint") != "auto":
                raise ValueError(
                    "--encoder-checkpoint requires model.road_encoder.initialization=pretrained "
                    "or a scratch GNN with an automatic pretrained scorer."
                )
            if source.name != "encoder.pt" or source.parent.name != "checkpoints":
                raise ValueError("Select the pretraining run's checkpoints/encoder.pt for its paired scorer.")
            raw["model"]["connectivity_scorer"] = {
                **scorer, "checkpoint": str(source.with_name("best.pt"))
            }
        else:
            raise ValueError("Unknown road encoder initialization for pretraining checkpoint selection.")
    if generation_checkpoint is not None:
        raw = {**raw, "generation": {**raw.get("generation", {}), "checkpoint": str(generation_checkpoint)}}
    parsed = ExperimentSettings.model_validate(raw)
    resolved = parsed.model_copy(update={
        "paths": _resolve_paths(parsed.paths, root),
        "model": _resolve_model_options(parsed.model, root),
        "tracking": _resolve_tracking(parsed.tracking, root)
    })
    joint = _read_model_settings(
        resolved.paths.graph_config, resolved.model, resolved.training
    )

    return ResolvedExperimentConfig(
        settings=resolved,
        joint_settings=joint,
        config_path=source_path,
        project_root=root
    )
