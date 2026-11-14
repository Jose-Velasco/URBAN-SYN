"""Shared YAML-driven training engine for joint GNN and diffusion experiments.

This module owns optimization, optional validation, checkpoint selection, and
tracking. Model computation and independently configured losses are injected
through the Stage 1/2 factory interfaces.
"""

from __future__ import annotations

import json
import logging
import math
import random
import uuid
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch_geometric.data import Data

from urban_syn.diffusion.losses import LossComposer

from .checkpoint import (
    experiment_fingerprint,
    load_training_checkpoint,
    save_training_checkpoint
)
from .config import ResolvedExperimentConfig
from .datasets import DatasetSplit, build_epoch_loader, load_training_dataset
from .diagnostics import EmbeddingProbe, gradient_norm
from .factories import TrainingComponents, build_loss_composer, build_training_components, step_scheduler
from .tracking import ExperimentTracker, TrackingRun


logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Initialize the Python, NumPy, and PyTorch random streams.

    Parameters
    ----------
    seed : int
        Nonnegative experiment seed.

    Returns
    -------
    None
        Sets the process-local random generators.
    """
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def training_log_file(path: Path) -> Iterator[None]:
    """Mirror training messages to stdout and the experiment log file.

    Parameters
    ----------
    path : Path
        File location for persistent training messages.

    Yields
    ------
    None
        Configured logger remains active during the context.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    disk = logging.FileHandler(path, encoding="utf-8")
    for handler in (stream, disk):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    original_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        yield
    finally:
        for handler in (stream, disk):
            logger.removeHandler(handler)
            handler.close()
        logger.setLevel(original_level)


def _run_directory(config: ResolvedExperimentConfig) -> Path:
    """Select a unique output directory or reuse an existing resumed run.

    Parameters
    ----------
    config : ResolvedExperimentConfig
        Output root, experiment name, and optional checkpoint resume path.

    Returns
    -------
    Path
        Directory for one experiment run, under the configured output root.
    """
    output_root = config.settings.paths.output_root.resolve()
    source = config.settings.training.checkpoint.resume_from
    if source is not None:
        checkpoint = Path(source).expanduser()
        checkpoint = (checkpoint if checkpoint.is_absolute() else config.project_root / checkpoint).resolve()
        run_dir = checkpoint.parent.parent
        if checkpoint.parent.name != "checkpoints" or not run_dir.is_relative_to(output_root):
            raise ValueError(
                "Resume checkpoints must be under <output_root>/<experiment>/<run>/checkpoints/."
            )
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint}")
        return run_dir

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
    return output_root / config.settings.experiment.name / run_id


def _save_run_files(config: ResolvedExperimentConfig, split: DatasetSplit, run_dir: Path) -> None:
    """Save reproducibility files without altering source training datasets.

    Parameters
    ----------
    config : ResolvedExperimentConfig
        Validated experiment and model architecture settings.
    split : DatasetSplit
        Exact training and optional validation membership.
    run_dir : Path
        Current experiment output directory.

    Returns
    -------
    None
        Writes JSON files containing run settings and selected FMM IDs.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    config_file = run_dir / "resolved_config.json"
    if not config_file.exists():
        config_file.write_text(json.dumps({
            "experiment": config.settings.model_dump(mode="json"),
            "model": config.joint_settings.model_dump(mode="json"),
            "project_root": str(config.project_root)
        }, indent=2), encoding="utf-8")

    split_file = run_dir / "data_split.json"
    if not split_file.exists():
        split_file.write_text(json.dumps({
            "train_fmm_ids": split.train_fmm_ids,
            "validation_fmm_ids": split.validation_fmm_ids
        }, indent=2), encoding="utf-8")


def _batch_loss_metrics(losses: Any, prefix: str) -> dict[str, float]:
    """Convert one loss-composer result into provider-neutral scalar metrics.

    Parameters
    ----------
    losses : LossResult
        Weighted objective and separately computed loss values.
    prefix : str
        Split name, either train or val.

    Returns
    -------
    dict[str, float]
        Named scalar metrics detached from the gradient graph.
    """
    metrics = {
        f"{prefix}/objective": float(losses.total.detach().item()),
        f"{prefix}/gradient_objective": float(losses.gradient_total.detach().item())
    }
    for name, value in losses.values.items():
        metrics[f"{prefix}/{name}"] = float(value.detach().item())
    return metrics


def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    """Average observed batch metrics, including intermittently logged terms.

    Parameters
    ----------
    metrics : list[dict[str, float]]
        One metric mapping per processed training or validation batch.

    Returns
    -------
    dict[str, float]
        Per-key average across batches in which the metric was calculated.
    """
    if not metrics:
        raise ValueError("A training or validation epoch processed no batches.")
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for batch in metrics:
        for name, value in batch.items():
            sums[name] = sums.get(name, 0.0) + value
            counts[name] = counts.get(name, 0) + 1
    return {key: sums[key] / counts[key] for key in sums}


def _assert_joint_gradients(model: nn.Module) -> None:
    """Verify that the first joint optimizer step updates both model branches.

    Parameters
    ----------
    model : nn.Module
        Joint road encoder and diffusion denoiser after backpropagation.

    Returns
    -------
    None
        Raises when an expected branch is disconnected from the objective.
    """
    for component in ("road_encoder", "diff_model"):
        branch = getattr(model, component, None)
        if branch is None:
            continue
        gradients = (
            parameter.grad for parameter in branch.parameters()
            if parameter.requires_grad and parameter.grad is not None
        )
        if not any(torch.isfinite(gradient).all().item() and gradient.abs().sum().item() > 0
                   for gradient in gradients):
            raise RuntimeError(f"Joint training produced no finite, nonzero {component} gradients.")


def run_epoch(
    model: nn.Module,
    graph: Data,
    batches: Iterable[tuple[Tensor, Tensor]],
    composer: LossComposer,
    optimizer: Optimizer | None,
    grad_clip_norm: float,
    global_step: int,
    max_batches: int | None,
    log_every_steps: int,
    run: TrackingRun | None = None,
    gnn_diagnostics_every_steps: int = 100
) -> tuple[dict[str, float], int]:
    """Execute one full training or validation epoch with shared model outputs.

    Parameters
    ----------
    model : nn.Module
        Trainable joint model exposing compute_training_outputs/losses.
    graph : Data
        Graph on the same device as the joint model.
    batches : Iterable[tuple[Tensor, Tensor]]
        Same-length road-ID and [0, 1] rate batches.
    composer : LossComposer
        Independently configured objective implementations.
    optimizer : Optimizer | None
        Optimizer when training; None for no-gradient validation.
    grad_clip_norm : float
        Maximum norm of training gradients.
    global_step : int
        Number of completed optimizer updates entering the epoch.
    max_batches : int | None
        Optional number of batches processed during training or validation.
    log_every_steps : int
        Interval for step-level tracking events.
    run : TrackingRun | None, default=None
        Optional tracker run for step-level metrics.
    gnn_diagnostics_every_steps : int, default=100
        Frequency of optional GNN diagnostics, in optimizer steps.

    Returns
    -------
    tuple[dict[str, float], int]
        Averaged batch metrics and updated optimizer-step counter.
    """
    training = optimizer is not None
    model.train(training)
    prefix = "train" if training else "val"
    recorded: list[dict[str, float]] = []

    for batch_index, (road_ids, rates) in enumerate(batches):
        if max_batches is not None and batch_index >= max_batches:
            break
        road_ids = road_ids.to(device=graph.x.device, dtype=torch.long, non_blocking=True)
        rates = rates.to(device=graph.x.device, dtype=graph.x.dtype, non_blocking=True)

        # Existing diffusion training expects [-1, 1] position rates.
        normalized_rates = rates.mul(2).sub(1)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            output = model.compute_training_outputs(graph, road_ids, normalized_rates)
            losses = model.compute_training_losses(
                output,
                composer,
                step=global_step if training else batch_index
            )
            if not torch.isfinite(losses.total).item():
                raise FloatingPointError(f"Non-finite {prefix} objective in batch {batch_index}.")
            diagnostic_metrics = {}
            if training:
                losses.total.backward()
                if global_step == 0:
                    _assert_joint_gradients(model)
                if (global_step + 1) % gnn_diagnostics_every_steps == 0 or global_step == 0:
                    diagnostic_metrics = {
                        "train/gnn_grad_norm": gradient_norm(model.road_encoder),
                        "train/denoiser_grad_norm": gradient_norm(model.diff_model)
                    }
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                if not torch.isfinite(grad_norm).item():
                    raise FloatingPointError("Non-finite gradient norm during joint training.")
                optimizer.step()
                global_step += 1

        metrics = _batch_loss_metrics(losses, prefix)
        if training:
            metrics[f"{prefix}/grad_norm"] = float(grad_norm.item())
            metrics.update(diagnostic_metrics)
        recorded.append(metrics)
        if training and run is not None and global_step % log_every_steps == 0:
            run.log_metrics({
                **metrics,
                "train/learning_rate": float(optimizer.param_groups[0]["lr"])
            }, step=global_step)

    return _mean_metrics(recorded), global_step


def _record_epoch_metrics(path: Path, epoch: int, step: int, metrics: dict[str, float]) -> None:
    """Append one self-describing JSON event, independent of the tracker.

    Parameters
    ----------
    path : Path
        Persistent line-oriented metrics file.
    epoch : int
        Completed training epoch.
    step : int
        Completed optimizer steps.
    metrics : dict[str, float]
        Train and optional validation values for this epoch.

    Returns
    -------
    None
        Appends a newline-delimited JSON metrics record.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"epoch": epoch, "global_step": step, **metrics}) + "\n")


def _checkpoint_monitor(config: ResolvedExperimentConfig) -> str:
    """Resolve training or validation objective used for checkpoint selection.

    Parameters
    ----------
    config : ResolvedExperimentConfig
        Validated checkpoint and validation options.

    Returns
    -------
    str
        Either train/objective or val/objective.
    """
    choice = config.settings.training.checkpoint.monitor
    if choice == "auto":
        return "val/objective" if config.settings.training.validation.enabled else "train/objective"
    return "val/objective" if choice == "val_loss" else "train/objective"


def _train_epochs(
    config: ResolvedExperimentConfig,
    components: TrainingComponents,
    dataset: Any,
    split: DatasetSplit,
    composer: LossComposer,
    fingerprint: str,
    run_dir: Path,
    restored: dict[str, Any] | None,
    run: TrackingRun
) -> int:
    """Perform epoch-level optimization, checkpoint selection, and artifact logging.

    Parameters
    ----------
    config : ResolvedExperimentConfig
        Validated experiment settings.
    components : TrainingComponents
        Prepared model, graph, optimizer, scheduler, and device.
    dataset : RoadSequenceDataset
        Training sequences used by the length-grouped data loaders.
    split : DatasetSplit
        Original-trip-aware training and optional validation indices.
    composer : LossComposer
        Injected loss implementations and weights.
    fingerprint : str
        Digest used to verify that checkpoints belong to this experiment.
    run_dir : Path
        Directory containing local experiment outputs.
    restored : dict[str, Any] | None
        Loaded checkpoint metadata, or None for new training.
    run : TrackingRun
        Provider-neutral active run for metrics, tags, and artifacts.

    Returns
    -------
    int
        Epoch number of the best checkpoint selected by the configured monitor.
    """
    settings = config.settings
    training = settings.training
    monitor = _checkpoint_monitor(config)
    checkpoint_dir = run_dir / "checkpoints"

    start_epoch = int(restored["epoch"]) + 1 if restored else 1
    global_step = int(restored["global_step"]) if restored else 0
    best_loss = float(restored["best_loss"]) if restored else math.inf
    best_epoch = int(restored["best_epoch"]) if restored else 0
    run.set_tags({"checkpoint_monitor": monitor})
    embedding_probe = EmbeddingProbe(
        settings.tracking.logging.gnn_embedding_probe_size,
        settings.experiment.seed
    )
    # Capture the initial/restored model as the comparison baseline, without
    # changing its training mode or consuming the training random stream.
    embedding_probe.observe(components.model, components.graph)

    for epoch in range(start_epoch, training.epochs + 1):
        train_batches = build_epoch_loader(
            dataset,
            split.train_indices,
            training.batch_size,
            seed=settings.experiment.seed + epoch,
            num_workers=settings.runtime.num_workers,
            shuffle=True,
            pin_memory=components.device.type == "cuda"
        )
        train_metrics, global_step = run_epoch(
            components.model,
            components.graph,
            train_batches,
            composer,
            components.optimizer,
            training.grad_clip_norm,
            global_step,
            training.max_batches,
            settings.tracking.logging.metrics_every_steps,
            run,
            gnn_diagnostics_every_steps=settings.tracking.logging.gnn_diagnostics_every_steps
        )
        metrics = dict(train_metrics)

        if training.validation.enabled:
            validation_batches = build_epoch_loader(
                dataset,
                split.validation_indices,
                training.batch_size,
                seed=settings.experiment.seed,
                num_workers=settings.runtime.num_workers,
                shuffle=False,
                pin_memory=components.device.type == "cuda"
            )
            val_metrics, _ = run_epoch(
                components.model,
                components.graph,
                validation_batches,
                composer,
                optimizer=None,
                grad_clip_norm=training.grad_clip_norm,
                global_step=global_step,
                max_batches=None,
                log_every_steps=settings.tracking.logging.metrics_every_steps
            )
            metrics.update(val_metrics)

        monitored_loss = metrics[monitor]
        step_scheduler(components.scheduler, monitored_loss)
        metrics["train/learning_rate"] = float(components.optimizer.param_groups[0]["lr"])

        improved = monitored_loss < best_loss
        if improved:
            best_loss = monitored_loss
            best_epoch = epoch

        # Observe only when at least one checkpoint will be saved. This adds
        # one no-gradient full-graph GNN forward per selected epoch, not per
        # training batch, and measures the actual post-optimization weights.
        checkpoint_due = (
            (improved and training.checkpoint.save_best)
            or training.checkpoint.save_last
            or epoch % training.checkpoint.every_epochs == 0
        )
        if checkpoint_due:
            metrics.update(embedding_probe.observe(components.model, components.graph))

        _record_epoch_metrics(run_dir / "metrics" / "history.jsonl", epoch, global_step, metrics)
        run.log_metrics({**metrics, "epoch": float(epoch)}, step=global_step)

        destinations = []
        if improved and training.checkpoint.save_best:
            destinations.append(checkpoint_dir / "best.pt")
        if training.checkpoint.save_last:
            destinations.append(checkpoint_dir / "last.pt")
        if epoch % training.checkpoint.every_epochs == 0:
            destinations.append(checkpoint_dir / f"epoch_{epoch:04d}.pt")

        for path in destinations:
            save_training_checkpoint(
                path,
                config,
                components.model,
                components.optimizer,
                components.scheduler,
                epoch,
                global_step,
                best_loss,
                best_epoch,
                fingerprint,
                split.train_fmm_ids,
                split.validation_fmm_ids,
                run.reference
            )

        logger.info(
            "epoch=%d/%d step=%d train=%.6f %s=%.6f best_epoch=%d",
            epoch,
            training.epochs,
            global_step,
            metrics["train/objective"],
            monitor,
            monitored_loss,
            best_epoch
        )

    completed_epoch = max(start_epoch - 1, training.epochs)
    run.set_tags({"best_epoch": str(best_epoch), "completed_epochs": str(completed_epoch)})
    for destination, path in (
        ("configuration", run_dir / "resolved_config.json"),
        ("configuration", run_dir / "data_split.json"),
        ("metrics", run_dir / "metrics" / "history.jsonl")
    ):
        if path.is_file():
            run.log_artifact(path, artifact_path=destination)
    if settings.tracking.artifacts.log_training_logs:
        run.log_artifact(run_dir / "logs" / "train.log", artifact_path="logs")
    for enabled, filename in (
        (settings.tracking.artifacts.log_best_checkpoint, "best.pt"),
        (settings.tracking.artifacts.log_last_checkpoint, "last.pt")
    ):
        checkpoint = checkpoint_dir / filename
        if enabled and checkpoint.is_file():
            run.log_artifact(checkpoint, artifact_path="checkpoints")
    return best_epoch


def run_joint_training(config: ResolvedExperimentConfig, tracker: ExperimentTracker) -> Path:
    """Run joint training using an injected, provider-neutral tracking backend.

    Parameters
    ----------
    config : ResolvedExperimentConfig
        Validated experiment settings and referenced graph/model options.
    tracker : ExperimentTracker
        Tracking adapter constructed by the pipeline runner or caller.

    Returns
    -------
    Path
        Run directory holding checkpoints, metrics, and training logs.
    """
    settings = config.settings
    training = settings.training
    run_dir = _run_directory(config)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    with training_log_file(run_dir / "logs" / "train.log"):
        logger.info("Experiment %s | output: %s", settings.experiment.name, run_dir)
        seed_everything(settings.experiment.seed)
        components = build_training_components(config)
        dataset, split = load_training_dataset(config, components.graph.edge_id)
        _save_run_files(config, split, run_dir)
        composer = build_loss_composer(training.losses)
        fingerprint = experiment_fingerprint(config, components.graph)
        monitor = _checkpoint_monitor(config)
        logger.info(
            "Loaded %d sequences (%d train, %d validation); monitor=%s",
            len(dataset),
            len(split.train_indices),
            len(split.validation_indices),
            monitor
        )

        restored = None
        if training.checkpoint.resume_from is not None:
            checkpoint = Path(training.checkpoint.resume_from).expanduser()
            if not checkpoint.is_absolute():
                checkpoint = config.project_root / checkpoint
            restored = load_training_checkpoint(
                checkpoint.resolve(),
                components.model,
                components.optimizer,
                components.scheduler,
                components.device,
                fingerprint,
                split.train_fmm_ids,
                split.validation_fmm_ids
            )
            logger.info(
                "Resuming after epoch %d at step %d",
                restored["epoch"],
                restored["global_step"]
            )

        previous = restored.get("tracking_reference") if restored else None
        with tracker.start_run(previous=previous) as run:
            best_epoch = _train_epochs(
                config,
                components,
                dataset,
                split,
                composer,
                fingerprint,
                run_dir,
                restored,
                run
            )

        logger.info("Training finished; best epoch: %d", best_epoch)
    return run_dir
