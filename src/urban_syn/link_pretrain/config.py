"""Validated settings for standalone road-link pretraining.

Only experiment choices belong in YAML. Smaller operational defaults remain in
these settings so the initial configuration stays short. Paths in YAML resolve
from the repository root, consistent with the existing joint experiment runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Settings(BaseModel):
    """Reject unsupported configuration while keeping settings immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)


class ExperimentInfo(_Settings):
    name: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
    seed: int = Field(default=101, ge=0)


class PathsSettings(_Settings):
    graph_config: Path
    output_root: Path = Path("outputs/experiments")


class GraphSplitSettings(_Settings):
    """Fractions of original directed positive links used by RandomLinkSplit."""

    val_fraction: float = Field(default=0.10, gt=0, lt=1, allow_inf_nan=False)
    test_fraction: float = Field(default=0.10, gt=0, lt=1, allow_inf_nan=False)
    disjoint_train_ratio: float = Field(default=0.20, gt=0, lt=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_fractions(self) -> Self:
        """Reject link splits that leave no training-positive partition."""
        if self.val_fraction + self.test_fraction >= 1:
            raise ValueError("graph_split val_fraction + test_fraction must be below 1.")
        return self


class NegativeSamplingSettings(_Settings):
    strategy: Literal["khop", "mixed"] = "mixed"
    amount: int = Field(default=2, ge=1)
    k_hops: int = Field(default=2, ge=2)
    hard_fraction: float = Field(default=0.75, ge=0, le=1, allow_inf_nan=False)
    evaluation_chunk_size: int = Field(default=128, ge=1)


class DecoderSettings(_Settings):
    """Architecture of the directional PyG MLP link scorer.

    The final layer always produces unrestricted logits without activation,
    normalization, or dropout. Other options apply to hidden layers.
    """

    hidden_dim: int = Field(default=128, ge=1)
    num_layers: int = Field(default=2, ge=1)
    activation: str = Field(default="relu", min_length=1)
    dropout: float = Field(default=0.0, ge=0, lt=1, allow_inf_nan=False)
    normalization: Literal["layer_norm", "batch_norm"] | None = None


class PretrainingSettings(_Settings):
    graph_split: GraphSplitSettings = Field(default_factory=GraphSplitSettings)
    positives_per_seed: Literal[1] = 1
    negatives: NegativeSamplingSettings = Field(default_factory=NegativeSamplingSettings)
    decoder: DecoderSettings = Field(default_factory=DecoderSettings)
    objective: Literal["bce", "bpr"] = "bpr"


class OptimizerSettings(_Settings):
    """Use the same field names as the existing optimizer builder."""

    name: Literal["adam", "adamw"] = "adamw"
    learning_rate: float = Field(default=0.001, gt=0, allow_inf_nan=False)
    weight_decay: float = Field(default=0.01, ge=0, allow_inf_nan=False)


class SchedulerSettings(_Settings):
    """Optional epoch scheduler, matching the joint runner's naming."""

    name: Literal["none", "cosine", "reduce_on_plateau"] = "reduce_on_plateau"
    min_lr: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    factor: float = Field(default=0.5, gt=0, lt=1, allow_inf_nan=False)
    patience: int = Field(default=10, ge=0)
    mode: Literal["min"] = "min"
    threshold: float = Field(default=1e-4, ge=0, allow_inf_nan=False)
    threshold_mode: Literal["rel", "abs"] = "rel"
    cooldown: int = Field(default=0, ge=0)
    eps: float = Field(default=1e-8, ge=0, allow_inf_nan=False)


class EarlyStoppingSettings(_Settings):
    """Stop when fixed validation loss stops improving."""

    enabled: bool = True
    patience: int = Field(default=15, ge=1)
    min_delta: float = Field(default=0.0, ge=0, allow_inf_nan=False)


class CheckpointSettings(_Settings):
    """Validation-loss monitoring is mandatory for standalone pretraining."""

    save_best: bool = True
    save_last: bool = True
    every_epochs: int = Field(default=10, ge=1)
    resume_from: Path | None = None


class TrainingSettings(_Settings):
    epochs: int = Field(default=100, ge=1)
    batch_size: int = Field(default=128, ge=1)
    max_batches: int | None = Field(default=None, ge=1)
    grad_clip_norm: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    optimizer: OptimizerSettings = Field(default_factory=OptimizerSettings)
    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)
    early_stopping: EarlyStoppingSettings = Field(default_factory=EarlyStoppingSettings)
    checkpoint: CheckpointSettings = Field(default_factory=CheckpointSettings)

    @model_validator(mode="after")
    def validate_scheduler(self) -> Self:
        """Ensure cosine scheduling has a reachable minimum learning rate."""
        if (
            self.scheduler.name == "cosine"
            and self.scheduler.min_lr >= self.optimizer.learning_rate
        ):
            raise ValueError("Cosine scheduler min_lr must be lower than the learning rate.")
        return self


class DiagnosticLoggingSettings(_Settings):
    """Cadence and sampling controls for shared road embedding diagnostics."""

    gnn_diagnostics_every_steps: int = Field(default=100, ge=0)
    gnn_embedding_enabled: bool = True
    gnn_embedding_every_epochs: int = Field(default=1, ge=1)
    gnn_embedding_sample_size: int = Field(default=128, ge=1)


class TrackingSettings(_Settings):
    enabled: bool = True
    provider: Literal["mlflow"] = "mlflow"
    experiment_name: str = Field(default="road_link_pretrain_nyc", min_length=1)
    tracking_uri: str | None = None
    storage_scope: Literal["shared", "job"] = "shared"
    run_name: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    metrics_every_steps: int = Field(default=100, ge=1)
    logging: DiagnosticLoggingSettings = Field(default_factory=DiagnosticLoggingSettings)


class RuntimeSettings(_Settings):
    device: Literal["auto", "cpu", "cuda"] = "auto"
    precision: Literal["fp32"] = "fp32"
    num_workers: int = Field(default=4, ge=0)
    prefetch_factor: int = Field(default=2, ge=1)
    persistent_workers: bool = True
    pin_memory: bool = True


class LinkPretrainSettings(_Settings):
    """Standalone profile, deliberately excluding diffusion and trajectory data."""

    experiment: ExperimentInfo
    paths: PathsSettings
    pretraining: PretrainingSettings = Field(default_factory=PretrainingSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)


@dataclass(frozen=True)
class ResolvedLinkPretrainConfig:
    """Validated pretraining settings and graph YAML metadata."""

    settings: LinkPretrainSettings
    config_path: Path
    project_root: Path
    graph_settings: dict[str, Any]


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    """Read a YAML mapping and reject empty or malformed roots."""
    try:
        with path.open(encoding="utf-8") as stream:
            content = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(content, dict):
        raise ValueError(f"Expected a YAML mapping at the root of {path}.")
    return content


def _absolute(path: Path, root: Path) -> Path:
    """Resolve a configuration path relative to the project root."""
    candidate = path.expanduser()
    return (candidate if candidate.is_absolute() else root / candidate).resolve()


def _validate_graph_yaml(graph: dict[str, Any], path: Path) -> None:
    """Check the graph YAML contract without constructing a PyG graph."""
    dataset = graph.get("dataset")
    model = graph.get("model")
    if not isinstance(dataset, dict) or not isinstance(graph.get("features"), dict):
        raise ValueError(f"Graph YAML requires dataset and features mappings: {path}")
    required_paths = ("root", "network_path")
    has_paths = all(
        isinstance(dataset.get(key), str) and dataset[key].strip()
        for key in required_paths
    )
    if not has_paths:
        raise ValueError(f"Graph YAML requires dataset.root and dataset.network_path: {path}")
    if not isinstance(model, dict) or not isinstance(model.get("gnn"), dict):
        raise ValueError(f"Graph YAML requires model.gnn: {path}")
    for name in ("node_features", "edge_features"):
        values = model.get(name, ["x"] if name == "node_features" else [])
        if not isinstance(values, list):
            raise ValueError(f"Graph YAML model.{name} must be a list of feature names: {path}")
        if any(not isinstance(item, str) or not item for item in values):
            raise ValueError(f"Graph YAML model.{name} must be a list of feature names: {path}")
        if name == "node_features" and not values:
            raise ValueError("Graph YAML model.node_features cannot be empty.")
        if len(set(values)) != len(values):
            raise ValueError(f"Graph YAML model.{name} contains duplicates.")


def load_link_pretrain_config(
    path: str | Path,
    project_root: str | Path | None = None
) -> ResolvedLinkPretrainConfig:
    """Load and validate pretraining settings without accessing the graph cache.

    Args:
        path: Path to the pretraining YAML. Relative paths may be resolved
            against ``project_root`` when supplied.
        project_root: Optional repository root. When omitted, discover it
            from the YAML's ancestor directories, falling back to the cwd.

    Returns:
        ``ResolvedLinkPretrainConfig`` with absolute configuration paths,
        validated settings, and the referenced graph YAML mapping.

    Raises:
        OSError: If a required YAML file cannot be read.
        ValueError: If YAML content or validated settings are invalid.
    """
    requested = Path(path).expanduser()
    explicit_root = Path(project_root).expanduser().resolve() if project_root is not None else None
    if explicit_root is not None and not requested.is_absolute() and not requested.is_file():
        requested = explicit_root / requested
    source_path = requested.resolve()

    # Config locations are not fixed. Infer the repository root from its layout
    # when possible, so moving YAML deeper does not change graph/data paths.
    if explicit_root is not None:
        root = explicit_root
    else:
        root = next(
            (parent for parent in source_path.parents
             if (parent / "configs" / "graph").is_dir()
             and (parent / "src" / "urban_syn").is_dir()),
            Path.cwd().resolve()
        )
    parsed = LinkPretrainSettings.model_validate(_read_yaml_mapping(source_path))
    resolved_paths = parsed.paths.model_copy(update={
        "graph_config": _absolute(parsed.paths.graph_config, root),
        "output_root": _absolute(parsed.paths.output_root, root)
    })
    resume_from = parsed.training.checkpoint.resume_from
    checkpoint = parsed.training.checkpoint.model_copy(update={
        "resume_from": _absolute(resume_from, root) if resume_from is not None else None
    })
    training = parsed.training.model_copy(update={"checkpoint": checkpoint})
    resolved = parsed.model_copy(update={"paths": resolved_paths, "training": training})
    graph = _read_yaml_mapping(resolved.paths.graph_config)
    _validate_graph_yaml(graph, resolved.paths.graph_config)
    return ResolvedLinkPretrainConfig(resolved, source_path, root, graph)
