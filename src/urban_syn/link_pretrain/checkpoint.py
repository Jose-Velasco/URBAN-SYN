"""Checkpoint storage for road-link pretraining and encoder reuse."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau

from .config import ResolvedLinkPretrainConfig
from urban_syn.experiment.tracking import TrackingReference


Scheduler = LRScheduler | ReduceLROnPlateau | None


def model_metadata(config: ResolvedLinkPretrainConfig, train_graph: Any) -> dict[str, Any]:
    """Describe the model and graph inputs needed to check checkpoint compatibility.

    Args:
        config: Resolved model, split, and sampling settings.
        train_graph: Training-only message graph with selected PyG features.

    Returns:
        Plain Python metadata describing the graph dimensions and model
        configuration. It intentionally does not persist graph splits or samples.
    """
    options = config.settings.pretraining
    edge_attr = getattr(train_graph, "edge_attr", None)
    return {
        "seed": config.settings.experiment.seed,
        "num_roads": int(train_graph.num_nodes),
        "message_edges": int(train_graph.edge_index.size(1)),
        "node_features": int(train_graph.x.size(1)),
        "edge_features": 0 if edge_attr is None else int(edge_attr.size(1)),
        "gnn": config.graph_settings["model"]["gnn"],
        "selected_node_features": config.graph_settings["model"].get("node_features", ["x"]),
        "selected_edge_features": config.graph_settings["model"].get("edge_features", []),
        "split": options.graph_split.model_dump(mode="json"),
        "negatives": options.negatives.model_dump(mode="json"),
        "decoder": options.decoder.model_dump(mode="json"),
        "objective": options.objective
    }


def _write_torch_file(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a checkpoint to avoid leaving a partially written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Scheduler,
    epoch: int,
    global_step: int,
    best_loss: float,
    best_epoch: int,
    epochs_without_improvement: int,
    metadata: dict[str, Any],
    tracking_reference: TrackingReference | None = None
) -> None:
    """Save model, optimizer, scheduler, and local PyTorch RNG state.

    Args:
        path: Destination ``.pt`` file, replaced atomically.
        model: Combined road encoder and directional link decoder.
        optimizer: Optimizer with its momentum/weight decay state.
        scheduler: Epoch-level scheduler or ``None``.
        epoch: Last completed one-based epoch.
        global_step: Number of successful optimizer updates so far.
        best_loss: Best fixed-validation objective so far.
        best_epoch: Epoch that produced ``best_loss``.
        epochs_without_improvement: Count used by early stopping.
        metadata: Model and graph metadata from ``model_metadata``.
        tracking_reference: Optional provider-neutral run identity for resuming
            MLflow history from the checkpoint itself.
    """
    payload = {
        "format_version": 1,
        "epoch": epoch,
        "global_step": global_step,
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "epochs_without_improvement": epochs_without_improvement,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "model_metadata": metadata,
        "tracking_reference": dict(tracking_reference) if tracking_reference is not None else None
    }
    _write_torch_file(path, payload)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    metadata: dict[str, Any],
    device: torch.device,
    optimizer: Optimizer | None = None,
    scheduler: Scheduler = None
) -> dict[str, Any]:
    """Load a compatible model and optionally restore training state.

    Args:
        path: Previously saved trusted checkpoint.
        model: Fresh encoder/decoder, with matching feature dimensions.
        metadata: Expected model/graph settings for this experiment.
        device: Device for model weights and optimizer state.
        optimizer: Supply to resume training; omit for held-out evaluation.
        scheduler: Scheduler to restore alongside an optimizer, if configured.

    Returns:
        Checkpoint payload including completed epoch, best loss, and step.

    Raises:
        ValueError: If checkpoint format, settings, or scheduler differ.
        OSError: If the checkpoint file is inaccessible.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported road-link checkpoint format.")
    if checkpoint.get("model_metadata") != metadata:
        raise ValueError("Checkpoint model, graph, or sampling settings do not match this run.")
    if optimizer is not None:
        saved_scheduler = checkpoint.get("scheduler_state_dict")
        if (saved_scheduler is None) != (scheduler is None):
            raise ValueError("Checkpoint and configuration use different schedulers.")

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint["cuda_rng_state"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
    return checkpoint


def save_encoder(path: Path, model: nn.Module, metadata: dict[str, Any]) -> None:
    """Export independently loadable GNN weights with architecture metadata.

    Args:
        path: Encoder-only ``.pt`` output file.
        model: Trained ``RoadLinkPredictionModel`` exposing ``.encoder``.
        metadata: Same graph and model settings stored in full checkpoints.
    """
    _write_torch_file(path, {
        "format_version": 1,
        "encoder_state_dict": model.encoder.state_dict(),
        "model_metadata": metadata
    })
