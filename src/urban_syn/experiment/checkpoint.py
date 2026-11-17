"""Save and resume a joint GNN/diffusion experiment at epoch boundaries.

Checkpoints contain only tensors and standard Python containers so they can
also be read by the existing generation loader with weights_only=True.
Only load checkpoints from trusted project runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch.optim import Optimizer

from .config import ResolvedExperimentConfig
from .factories import Scheduler


CHECKPOINT_VERSION = 2


def experiment_fingerprint(config: ResolvedExperimentConfig, graph: Data) -> str:
    """Fingerprint model/training choices and the graph's road-ID order.

    Parameters
    ----------
    config : ResolvedExperimentConfig
        Validated settings to check for compatible resume.
    graph : Data
        Selected PyG graph containing ordered road IDs, connectivity and features.

    Returns
    -------
    str
        SHA-256 digest of relevant experiment settings and road ordering.
    """
    settings = config.settings
    training_options = settings.training.model_dump(mode="json")
    # resume_from changes from null to a file path when restarting a run.
    # It is execution metadata, not a different training experiment.
    training_options["checkpoint"].pop("resume_from", None)
    relevant = {
        "gnn": config.joint_settings.gnn.model_dump(mode="json"),
        "diffusion": config.joint_settings.diffusion.model_dump(mode="json"),
        "training": training_options,
        "seed": settings.experiment.seed,
        "graph_config": str(settings.paths.graph_config),
        "train_sequences": str(settings.paths.train_sequences),
        "trip_id_map": str(settings.paths.trip_id_map)
    }
    # Resuming under another output directory is fine; the model and data
    # selection must still match. Other runtime/tracking values may change.
    digest = hashlib.sha256(json.dumps(relevant, sort_keys=True).encode("utf-8"))
    # Include graph connectivity and actual selected features so a rebuilt graph
    # with identical edge IDs but changed training inputs cannot silently resume.
    for attribute in ("edge_id", "edge_index", "x", "edge_attr"):
        value = getattr(graph, attribute, None)
        digest.update(attribute.encode("utf-8"))
        if value is None:
            digest.update(b"<missing>")
            continue
        value = value.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(value.numpy().tobytes())

    for path in (settings.paths.train_sequences, settings.paths.trip_id_map):
        if path is not None:
            status = path.stat()
            digest.update(f"{path}:{status.st_size}:{status.st_mtime_ns}".encode("utf-8"))
    return digest.hexdigest()


def capture_rng_state() -> dict[str, Any]:
    """Capture random state with safe-to-load standard Python containers.

    Returns
    -------
    dict[str, Any]
        Python, NumPy, PyTorch CPU, and available CUDA RNG states.
    """
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (
            numpy_state[0],
            numpy_state[1].tolist(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4]
        ),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore random streams saved at an epoch boundary.

    Parameters
    ----------
    state : dict[str, Any]
        State previously returned by capture_rng_state().

    Returns
    -------
    None
        Restores process-local random generators in place.
    """
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((
        numpy_state[0],
        np.asarray(numpy_state[1], dtype=np.uint32),
        numpy_state[2],
        numpy_state[3],
        numpy_state[4]
    ))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_training_checkpoint(
    path: str | Path,
    config: ResolvedExperimentConfig,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Scheduler,
    epoch: int,
    global_step: int,
    best_loss: float,
    best_epoch: int,
    fingerprint: str,
    train_fmm_ids: tuple[int, ...],
    validation_fmm_ids: tuple[int, ...],
    tracking_reference: dict[str, str | None] | None
) -> None:
    """Atomically save all state needed for epoch-boundary recovery.

    Parameters
    ----------
    path : str | Path
        Destination .pt file; temporary file is placed in same directory.
    config : ResolvedExperimentConfig
        Saved model and experiment settings for provenance.
    model : nn.Module
        Joint GNN/diffusion model.
    optimizer : Optimizer
        Current optimizer and momentum states.
    scheduler : Scheduler
        Configured epoch scheduler, or None.
    epoch : int
        Completed one-based epoch number.
    global_step : int
        Completed optimizer update count.
    best_loss : float
        Smallest monitored loss observed through this epoch.
    best_epoch : int
        Epoch associated with best_loss.
    fingerprint : str
        Model/config/road-order compatibility digest.
    train_fmm_ids : tuple[int, ...]
        Exact training FMM IDs in dataset row order.
    validation_fmm_ids : tuple[int, ...]
        Exact held-out validation IDs, empty if disabled.
    tracking_reference : dict[str, str | None] | None
        Provider-neutral run reference, or None when tracking is unavailable.

    Returns
    -------
    None
        Writes the checkpoint to disk and replaces the previous file atomically.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "epoch": epoch,
        "global_step": global_step,
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "experiment_fingerprint": fingerprint,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "rng_state": capture_rng_state(),
        "train_fmm_ids": list(train_fmm_ids),
        "validation_fmm_ids": list(validation_fmm_ids),
        "tracking_reference": dict(tracking_reference) if tracking_reference is not None else None,
        "config": config.joint_settings.model_dump(mode="json"),
        "experiment_config": config.settings.model_dump(mode="json")
    }
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Scheduler,
    device: torch.device,
    fingerprint: str,
    train_fmm_ids: tuple[int, ...],
    validation_fmm_ids: tuple[int, ...]
) -> dict[str, Any]:
    """Restore model and optimizer state from a compatible trusted checkpoint.

    Parameters
    ----------
    path : str | Path
        Checkpoint file created by save_training_checkpoint().
    model : nn.Module
        Freshly initialized joint model.
    optimizer : Optimizer
        Optimizer matching the saved parameter groups.
    scheduler : Scheduler
        Scheduler matching the original training configuration, or None.
    device : torch.device
        Device on which the restored optimizer/model state should reside.
    fingerprint : str
        Required matching config/model/graph digest.
    train_fmm_ids : tuple[int, ...]
        Expected training identity partition.
    validation_fmm_ids : tuple[int, ...]
        Expected validation identity partition.

    Returns
    -------
    dict[str, Any]
        Checkpoint metadata, including completed epoch and tracking reference.
    """
    # weights_only=True also allows generation to load this .pt safely.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    version = checkpoint.get("checkpoint_version")
    if version not in (1, CHECKPOINT_VERSION):
        raise ValueError("Unsupported or incomplete training checkpoint version.")

    if version == 1:
        # Allow resuming checkpoints made by the first Stage 3 implementation.
        # Only the saved tracking metadata format differs; model training state
        # is unchanged. New checkpoints write provider-neutral references.
        old_run_id = checkpoint.pop("mlflow_run_id", None)
        old_uri = checkpoint.pop("tracking_uri", None)
        checkpoint["tracking_reference"] = {
            "provider": "mlflow" if old_run_id is not None else "none",
            "run_id": old_run_id,
            "uri": old_uri
        }
    if checkpoint.get("experiment_fingerprint") != fingerprint:
        raise ValueError("Checkpoint settings or graph road ordering differ from this run.")
    if tuple(checkpoint["train_fmm_ids"]) != train_fmm_ids:
        raise ValueError("Training source FMM IDs differ from the saved run.")
    if tuple(checkpoint["validation_fmm_ids"]) != validation_fmm_ids:
        raise ValueError("Validation source FMM IDs differ from the saved run.")
    if (checkpoint.get("scheduler_state_dict") is None) != (scheduler is None):
        raise ValueError("Checkpoint and experiment use different scheduler modes.")

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    restore_rng_state(checkpoint["rng_state"])
    return checkpoint
