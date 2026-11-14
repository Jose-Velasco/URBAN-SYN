"""Build experiment components from validated settings.

Model and graph construction reuse the project's existing modules. The
functions here have no training-loop or MLflow responsibilities, and import
PyTorch Geometric only when a graph/model is actually requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler, ReduceLROnPlateau

from urban_syn.diffusion.losses import (
    ConfiguredLoss,
    HardConnectivity,
    LossComposer,
    NoiseMSE,
    ReconstructionMSE
)

from urban_syn.diffusion.sampling.base import Sampler
from urban_syn.diffusion.sampling.ddpm import DDPMSampler

from .config import LossSettings, OptimizerSettings, ResolvedExperimentConfig, SchedulerSettings

if TYPE_CHECKING:
    from torch_geometric.data import Data


Scheduler = LRScheduler | ReduceLROnPlateau | None


@dataclass
class TrainingComponents:
    """Initialized components required by the shared training engine."""

    graph: Data
    model: nn.Module
    optimizer: Optimizer
    scheduler: Scheduler
    device: torch.device


def resolve_device(selection: str) -> torch.device:
    """Resolve the configured device and fail if requested CUDA is missing."""
    if selection == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if selection == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("runtime.device=cuda requires an available CUDA GPU.")
        return torch.device("cuda")
    if selection == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported runtime device: {selection!r}")


def _project_path(path: str | Path, project_root: Path) -> Path:
    """Resolve graph-YAML data paths using the experiment's project root."""
    candidate = Path(path).expanduser()
    return (candidate if candidate.is_absolute() else project_root / candidate).resolve()


def load_experiment_graph(config: ResolvedExperimentConfig, device: torch.device) -> Data:
    """Load the existing processed graph and select the configured features.

    This does not rebuild missing graph artifacts. It uses the same graph
    configuration and feature-selection helpers as the existing training code,
    while correctly resolving paths when the caller is outside project_root.
    """
    from torch_geometric.data import Data

    from urban_syn.graph.config import build_dataset_from_config, load_graph_config
    from urban_syn.graph.selection import select_graph_features
    from urban_syn.graph.validation import validate_graph_features

    graph_config = load_graph_config(config.settings.paths.graph_config)
    dataset_config = dict(graph_config["dataset"])
    for key in ("root", "network_path"):
        dataset_config[key] = str(_project_path(dataset_config[key], config.project_root))
    graph_config["dataset"] = dataset_config

    processed_dir = Path(dataset_config["root"]) / "processed"
    for filename in ("road_graph.pt", "preprocessor.joblib"):
        artifact = processed_dir / filename
        if not artifact.is_file():
            raise FileNotFoundError(
                f"Processed graph artifact not found: {artifact}. "
                "Prepare the road graph before starting training."
            )

    dataset = build_dataset_from_config(graph_config)
    graph = dataset[0]

    model_config = graph_config["model"]
    node_features = model_config.get("node_features", ["x"])
    edge_features = model_config.get("edge_features", [])
    validate_graph_features(graph, node_features=node_features, edge_features=edge_features)
    graph = select_graph_features(
        graph,
        node_features=node_features,
        edge_features=edge_features
    )

    # Keep just the tensors used by the joint model to avoid GPU copies of
    # unused graph metadata and raw feature attributes.
    selected = Data(
        x=graph.x,
        edge_index=graph.edge_index,
        edge_attr=getattr(graph, "edge_attr", None),
        edge_id=graph.edge_id
    )
    return selected.to(device)


def build_model(config: ResolvedExperimentConfig, graph: Data) -> nn.Module:
    """Construct the trainable joint GNN + diffusion model.

    Uses the GNN settings from the graph YAML and diffusion settings from the
    experiment YAML. No smoke-test configuration is reloaded here.
    """
    from urban_syn.diffusion.training import build_joint_model

    model = build_joint_model(graph, config.joint_settings)
    if not any(parameter.requires_grad for parameter in model.road_encoder.parameters()):
        raise ValueError("Joint training requires trainable road-encoder parameters.")
    return model


def build_optimizer(model: nn.Module, settings: OptimizerSettings) -> Optimizer:
    """Build an optimizer over all and only currently trainable parameters."""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("The model has no trainable parameters.")

    optimizer_classes = {"adam": Adam, "adamw": AdamW}
    optimizer_type = optimizer_classes.get(settings.name)
    if optimizer_type is None:
        raise ValueError(f"Unsupported optimizer: {settings.name!r}")

    return optimizer_type(
        parameters,
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay
    )


def build_scheduler(
    optimizer: Optimizer,
    settings: SchedulerSettings,
    epochs: int
) -> Scheduler:
    """Build an epoch-level scheduler, or None if disabled.

    CosineAnnealingLR advances once per epoch. ReduceLROnPlateau must be
    advanced with the monitored training or validation objective.
    """
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    if settings.name == "none":
        return None
    if settings.name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=settings.min_lr)
    if settings.name == "reduce_on_plateau":
        return ReduceLROnPlateau(optimizer, mode="min", patience=settings.patience_epochs)
    raise ValueError(f"Unsupported scheduler: {settings.name!r}")


def step_scheduler(scheduler: Scheduler, monitored_loss: float | None = None) -> None:
    """Advance an epoch-level scheduler using the correct call signature."""
    if scheduler is None:
        return
    if isinstance(scheduler, ReduceLROnPlateau):
        if monitored_loss is None:
            raise ValueError("ReduceLROnPlateau requires a monitored loss.")
        scheduler.step(monitored_loss)
        return
    scheduler.step()


def build_training_components(
    config: ResolvedExperimentConfig,
    graph: Data | None = None,
    model: nn.Module | None = None
) -> TrainingComponents:
    """Assemble model, graph, optimizer, and scheduler for a joint experiment.

    The optional graph/model inputs allow tests and future callers to inject
    prepared instances. Constructing components does not start training,
    create a checkpoint, or open an MLflow run.
    """
    device = resolve_device(config.settings.runtime.device)
    graph = load_experiment_graph(config, device) if graph is None else graph.to(device)
    model = build_model(config, graph) if model is None else model.to(device)

    optimizer = build_optimizer(model, config.settings.training.optimizer)
    scheduler = build_scheduler(
        optimizer,
        config.settings.training.scheduler,
        config.settings.training.epochs
    )
    return TrainingComponents(
        graph=graph,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )


def build_loss_composer(settings: LossSettings) -> LossComposer:
    """Construct the enabled training objectives from validated settings.

    Parameters
    ----------
    settings : LossSettings
        Selection, weight, and frequency of each available loss.

    Returns
    -------
    LossComposer
        Injected objective components with the requested composition policy.
    """
    registry = {
        "noise_mse": (NoiseMSE, True),
        "reconstruction_mse": (ReconstructionMSE, True),
        "hard_connectivity": (HardConnectivity, False)
    }
    selected = []

    for name, (loss_type, differentiable) in registry.items():
        choice = getattr(settings, name)
        if not choice.enabled:
            continue

        selected.append(ConfiguredLoss(
            name=name,
            function=loss_type(),
            weight=choice.weight,
            include_in_total=choice.include_in_total,
            every_steps=choice.every_steps,
            differentiable=differentiable
        ))

    return LossComposer(selected)


def build_sampler(name: str = "ddpm") -> Sampler:
    """Build a diffusion sampler from a validated algorithm name.

    Parameters
    ----------
    name : str, default="ddpm"
        Sampling algorithm selected by generation.sampler.name.

    Returns
    -------
    Sampler
        Configured sampler implementing the common sampling contract.

    Raises
    ------
    ValueError
        If the requested algorithm is not implemented.
    """
    if name == "ddpm":
        return DDPMSampler()
    raise ValueError(f"Unsupported diffusion sampler: {name!r}")
