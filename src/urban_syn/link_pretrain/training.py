"""Training and fixed held-out evaluation for structural road-link pretraining."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint, model_metadata, save_checkpoint, save_encoder
from .config import ResolvedLinkPretrainConfig
from .data import LinkBatch, SamplingStats
from .evaluation import build_fixed_evaluation
from .loader import make_training_dataloader
from .metrics import append_metric_history, export_metric_history
from .loss import calculate_link_loss
from .model import RoadLinkPredictionModel, build_link_prediction_model
from .tracking import read_tracking_reference, write_tracking_reference
from urban_syn.experiment.diagnostics import (
    EmbeddingDiagnostics, gradient_norm,
    should_log_gradient_diagnostics, should_record_embeddings
)
from urban_syn.experiment.tracking import ExperimentTracker, TrackingReference, TrackingRun

if TYPE_CHECKING:
    from torch_geometric.data import Data


Scheduler = LRScheduler | ReduceLROnPlateau | None


class LinkRoadEmbeddingSource:
    """Expose padding-free pretraining embeddings to shared diagnostics."""

    def module(self, model: RoadLinkPredictionModel) -> torch.nn.Module:
        """Return the model's road GNN so its training mode can be restored."""
        return model.encoder

    def encode(self, model: RoadLinkPredictionModel, graph: Data) -> Tensor:
        """Return road embeddings ``[N, D]`` in zero-based PyG road order."""
        return model.encode(graph)


@dataclass(frozen=True)
class TrainingResult:
    """Best-validation selection and continuation information.

    Attributes:
        run_dir: Folder holding metrics, configuration, logs, and checkpoints.
        best_checkpoint: Selected best-validation model checkpoint.
        best_epoch: One-based epoch with the lowest validation loss.
        best_validation_loss: Mean objective on fixed validation pairs.
        global_step: Number of completed optimizer steps at the end of training.
    """

    run_dir: Path
    best_checkpoint: Path
    best_epoch: int
    best_validation_loss: float
    global_step: int


def _build_training_tools(
    config: ResolvedLinkPretrainConfig,
    model: RoadLinkPredictionModel
) -> tuple[Optimizer, Scheduler]:
    """Build the existing optimizer and scheduler for the complete link model.

    Args:
        config: Validated optimizer, scheduler, and epoch settings.
        model: Encoder and decoder whose parameters are jointly optimized.

    Returns:
        Configured optimizer and optional epoch-level scheduler.
    """
    from urban_syn.experiment.factories import build_optimizer, build_scheduler

    settings = config.settings.training
    optimizer = build_optimizer(model, settings.optimizer)
    scheduler = build_scheduler(optimizer, settings.scheduler, settings.epochs)
    return optimizer, scheduler


def _resolve_device(config: ResolvedLinkPretrainConfig) -> torch.device:
    """Use the existing project's device-selection rules."""
    from urban_syn.experiment.factories import resolve_device

    return resolve_device(config.settings.runtime.device)


@torch.no_grad()
def evaluate_link_metrics(
    model: RoadLinkPredictionModel,
    message_graph: Data,
    samples: LinkBatch,
    objective: str,
    chunk_size: int
) -> dict[str, float]:
    """Score fixed held-out pairs using one graph encoding and bounded chunks.

    Args:
        model: Encoder and directional link scorer in evaluation mode.
        message_graph: Device-ready training-only message graph.
        samples: Fixed positive links ``[P]`` and aligned negatives
            ``[P, amount]``; unchanged across validation epochs.
        objective: Configured BCE or BPR objective.
        chunk_size: Maximum positive links scored per decoder pass.

    Returns:
        Mean loss, pairwise ranking accuracy, Hits@1, and mean reciprocal
        rank. Ties count against the positive link, including in MRR.
        Loss is weighted by positives, so results do not depend on chunk size.

    Raises:
        ValueError: If no examples are available or chunk size is invalid.
        FloatingPointError: If the computed objective is non-finite.
    """
    if chunk_size < 1:
        raise ValueError("Evaluation chunk size must be positive.")
    positive_count = samples.positive_src.numel()
    if positive_count == 0:
        raise ValueError("Cannot evaluate a sample set with no positive links.")

    model.eval()
    embeddings = model.encode(message_graph)
    weighted_loss = 0.0
    correctly_ranked_pairs = 0
    top_ranked_positives = 0
    reciprocal_rank_sum = 0.0
    negative_count = 0

    for start in range(0, positive_count, chunk_size):
        stop = min(start + chunk_size, positive_count)
        # Reuse the same source-aligned scoring method as training.
        # Only sample indices are sliced; graph embeddings are reused.
        chunk_count = stop - start
        chunk = LinkBatch(
            positive_src=samples.positive_src[start:stop],
            positive_dst=samples.positive_dst[start:stop],
            negative_dst=samples.negative_dst[start:stop],
            hard_mask=samples.hard_mask[start:stop],
            stats=SamplingStats(
                requested_positives=chunk_count,
                retained_positives=chunk_count
            )
        )
        scores = model.score_batch(embeddings, chunk)
        loss = calculate_link_loss(scores.positive, scores.negative, objective)
        if not torch.isfinite(loss).item():
            raise FloatingPointError("Non-finite held-out link loss.")
        weighted_loss += loss.item() * (stop - start)

        positive = scores.positive.unsqueeze(1)
        correctly_ranked_pairs += int((positive > scores.negative).sum().item())
        negative_count += scores.negative.numel()
        # Any tied or higher-scoring negative pushes the true link lower.
        higher_or_tied = (scores.negative >= positive).sum(dim=1)
        top_ranked_positives += int((higher_or_tied == 0).sum().item())
        reciprocal_rank_sum += (1.0 / (higher_or_tied.float() + 1.0)).sum().item()

    return {
        "loss": weighted_loss / positive_count,
        "pairwise_accuracy": correctly_ranked_pairs / negative_count,
        "hits_at_1": top_ranked_positives / positive_count,
        "mrr": reciprocal_rank_sum / positive_count
    }


@torch.no_grad()
def evaluate_links(
    model: RoadLinkPredictionModel,
    message_graph: Data,
    samples: LinkBatch,
    objective: str,
    chunk_size: int
) -> float:
    """Return only fixed held-out loss for callers selecting an objective.

    Args:
        model: Evaluated road encoder and decoder.
        message_graph: Training-only message-passing graph.
        samples: Fixed positive and aligned negative links.
        objective: BCE or BPR.
        chunk_size: Maximum positive links per decoder pass.

    Returns:
        Mean evaluation loss, weighted over retained positive links.
    """
    return evaluate_link_metrics(
        model, message_graph, samples, objective, chunk_size
    )["loss"]



def _step_scheduler(scheduler: Scheduler, validation_loss: float) -> None:
    """Advance the existing epoch scheduler using fixed-validation loss."""
    from urban_syn.experiment.factories import step_scheduler

    step_scheduler(scheduler, validation_loss)


@dataclass
class _TrainingResources:
    """Prepared model and data reused across training epochs."""

    model: RoadLinkPredictionModel
    message_graph: Data
    optimizer: Optimizer
    scheduler: Scheduler
    loader: DataLoader
    fixed_samples: dict[str, LinkBatch]
    metadata: dict[str, Any]


@dataclass
class _TrainingProgress:
    """Mutable epoch-level state restored from a checkpoint when resuming."""

    start_epoch: int = 1
    global_step: int = 0
    best_loss: float = float("inf")
    best_epoch: int = 0
    stalled_epochs: int = 0


def _prepare_training(
    config: ResolvedLinkPretrainConfig,
    full_graph: Data,
    splits: tuple[Data, Data, Data],
    fixed_samples: dict[str, LinkBatch] | None,
    logger: logging.Logger
) -> _TrainingResources:
    """Create the GNN, optimizer, sampling loader, and fixed evaluation pairs.

    Args:
        config: Validated GNN and training settings.
        full_graph: Original CPU graph, used for negative exclusion only.
        splits: Training, validation, and test graph splits.
        fixed_samples: Existing in-memory held-out batches, if available.
        logger: Console/file logger for data preparation results.

    Returns:
        Ready-to-train components. Only the training split is transferred to
        the accelerator for message passing.
    """
    settings = config.settings
    train_graph = splits[0]
    device = _resolve_device(config)
    message_graph = train_graph.clone().to(device)
    model = build_link_prediction_model(train_graph, config).to(device)
    optimizer, scheduler = _build_training_tools(config, model)
    loader = make_training_dataloader(full_graph, train_graph, settings)
    if fixed_samples is None:
        fixed_samples = build_fixed_evaluation(
            full_graph, splits, settings.pretraining.negatives, settings.experiment.seed
        )
    logger.info(
        "Fixed evaluation: validation=%d/%d, test=%d/%d positives retained",
        fixed_samples["validation"].stats.retained_positives,
        fixed_samples["validation"].stats.requested_positives,
        fixed_samples["test"].stats.retained_positives,
        fixed_samples["test"].stats.requested_positives
    )
    return _TrainingResources(
        model, message_graph, optimizer, scheduler, loader, fixed_samples,
        model_metadata(config, train_graph)
    )


def _train_epoch(
    resources: _TrainingResources,
    config: ResolvedLinkPretrainConfig,
    epoch: int,
    global_step: int,
    run: TrackingRun,
    logger: logging.Logger,
    history_path: Path
) -> tuple[float, int, SamplingStats]:
    """Optimize one epoch on sampled supervised links.

    Args:
        resources: Model, optimizer, training graph, and CPU sampling loader.
        config: Objective, clipping threshold, and logging intervals.
        epoch: One-based epoch number.
        global_step: Completed optimizer updates before this epoch.
        run: Backend-independent tracking run for occasional step metrics.
        logger: Console/file logger for training progress.
        history_path: Append-only local event file for step-level diagnostics.

    Returns:
        Weighted mean loss, updated optimizer step, and aggregated sampling
        counters. Empty batches contribute counters but no optimizer steps.

    Raises:
        RuntimeError: If no positive links survive negative sampling.
        FloatingPointError: If a loss or gradient norm is non-finite.
    """
    settings = config.settings
    model = resources.model
    optimizer = resources.optimizer
    model.train()
    total_loss = 0.0
    total_positives = 0
    sampling_stats = SamplingStats()

    for batch_index, batch in enumerate(resources.loader, start=1):
        sampling_stats = sampling_stats + batch.stats
        if batch.positive_src.numel() > 0:
            optimizer.zero_grad(set_to_none=True)
            scores = model(resources.message_graph, batch)
            loss = calculate_link_loss(
                scores.positive, scores.negative, settings.pretraining.objective
            )
            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"Non-finite link loss at epoch {epoch}, batch {batch_index}."
                )
            loss.backward()
            next_step = global_step + 1
            if should_log_gradient_diagnostics(
                next_step, settings.tracking.logging.gnn_diagnostics_every_steps
            ):
                # Measure before clipping to detect unstable or vanishing gradients.
                gradient_metrics = {"train/gnn_grad_norm": gradient_norm(model.encoder)}
                run.log_metrics(gradient_metrics, next_step)
                append_metric_history(history_path, epoch, next_step, gradient_metrics)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), settings.training.grad_clip_norm
            )
            if not torch.isfinite(grad_norm).item():
                raise FloatingPointError(f"Non-finite gradient norm at epoch {epoch}.")
            optimizer.step()

            retained = batch.positive_src.numel()
            total_loss += float(loss.item()) * retained
            total_positives += retained
            global_step = next_step

            if global_step % settings.tracking.metrics_every_steps == 0:
                mean_loss = total_loss / total_positives
                running_metrics = {"train/batch_running_loss": mean_loss}
                run.log_metrics(running_metrics, global_step)
                append_metric_history(history_path, epoch, global_step, running_metrics)
                logger.info(
                    "Epoch %d | step %d | running train loss %.5f",
                    epoch, global_step, mean_loss
                )

        if settings.training.max_batches is not None and batch_index >= settings.training.max_batches:
            break

    if total_positives == 0:
        raise RuntimeError("No training links with valid negatives were available this epoch.")
    return total_loss / total_positives, global_step, sampling_stats


def _save_epoch_checkpoints(
    config: ResolvedLinkPretrainConfig,
    resources: _TrainingResources,
    progress: _TrainingProgress,
    checkpoint_dir: Path,
    epoch: int,
    new_best: bool,
    reference: TrackingReference
) -> None:
    """Write selected, latest, and optional numbered checkpoints as configured.

    Args:
        config: Checkpoint frequency and selection settings.
        resources: Model, optimizer, scheduler, and compatibility metadata.
        progress: Updated best loss, patience, and optimizer step.
        checkpoint_dir: Output folder for model states and encoder export.
        epoch: One-based completed epoch.
        new_best: Whether this epoch achieved the lowest validation loss.
        reference: Tracking run identity stored for portable resumption.
    """
    settings = config.settings.training.checkpoint
    targets: list[Path] = []
    if new_best and settings.save_best:
        targets.append(checkpoint_dir / "best.pt")
    if settings.save_last:
        targets.append(checkpoint_dir / "last.pt")
    if epoch % settings.every_epochs == 0:
        targets.append(checkpoint_dir / f"epoch_{epoch:04d}.pt")

    for destination in targets:
        save_checkpoint(
            destination, resources.model, resources.optimizer, resources.scheduler,
            epoch, progress.global_step, progress.best_loss, progress.best_epoch,
            progress.stalled_epochs, resources.metadata, reference
        )
    if new_best and settings.save_best:
        save_encoder(checkpoint_dir / "encoder.pt", resources.model, resources.metadata)


def _train_epochs(
    config: ResolvedLinkPretrainConfig,
    resources: _TrainingResources,
    progress: _TrainingProgress,
    run_dir: Path,
    run: TrackingRun,
    logger: logging.Logger
) -> _TrainingProgress:
    """Select the best checkpoint while logging validation and representation health.

    Args:
        config: Training schedule, loss objective, checkpoint, and diagnostics settings.
        resources: Ready-to-train model and batches.
        progress: Initial or restored best loss and optimizer step.
        run_dir: Folder for checkpoints and append-only local history.
        run: Provider-neutral tracking run.
        logger: Console/file logger for per-epoch summaries.

    Returns:
        Progress including the selected epoch and completed update count.
    """
    settings = config.settings
    checkpoint_dir = run_dir / "checkpoints"
    history_path = run_dir / "metrics.jsonl"
    diagnostic_settings = settings.tracking.logging
    diagnostics = None
    if diagnostic_settings.gnn_embedding_enabled:
        diagnostics = EmbeddingDiagnostics(
            sample_size=diagnostic_settings.gnn_embedding_sample_size,
            seed=settings.experiment.seed,
            source=LinkRoadEmbeddingSource()
        )
        # Record the random-initialization baseline for new runs. When resuming,
        # this observation is only the reference for later drift measurements.
        initial = diagnostics.observe(resources.model, resources.message_graph)
        sample_path = diagnostics.write_sample_manifest(
            run_dir / "road_embedding_sample.json", resources.message_graph
        )
        run.log_artifact(sample_path, "metrics")
        if progress.start_epoch == 1:
            run.log_metrics({**initial, "epoch": 0.0}, progress.global_step)
            append_metric_history(history_path, 0, progress.global_step, initial)

    for epoch in range(progress.start_epoch, settings.training.epochs + 1):
        started = time.perf_counter()
        train_loss, progress.global_step, sampling = _train_epoch(
            resources, config, epoch, progress.global_step, run, logger, history_path
        )
        validation = evaluate_link_metrics(
            resources.model, resources.message_graph, resources.fixed_samples["validation"],
            settings.pretraining.objective,
            settings.pretraining.negatives.evaluation_chunk_size
        )
        val_loss = validation["loss"]
        _step_scheduler(resources.scheduler, val_loss)

        # min_delta governs patience only; always preserve the actual lowest loss.
        new_best = val_loss < progress.best_loss
        enough_improvement = val_loss < (
            progress.best_loss - settings.training.early_stopping.min_delta
        )
        if new_best:
            progress.best_loss = val_loss
            progress.best_epoch = epoch
        if enough_improvement:
            progress.stalled_epochs = 0
        else:
            progress.stalled_epochs += 1

        learning_rate = float(resources.optimizer.param_groups[0]["lr"])
        duration = time.perf_counter() - started
        metrics = {
            "train/loss": train_loss,
            "validation/loss": val_loss,
            "validation/pairwise_accuracy": validation["pairwise_accuracy"],
            "validation/hits_at_1": validation["hits_at_1"],
            "validation/mrr": validation["mrr"],
            "validation/best_loss": progress.best_loss,
            "train/learning_rate": learning_rate,
            "train/epoch_seconds": duration,
            "sampling/retained_positives": float(sampling.retained_positives),
            "sampling/skipped": float(sampling.skipped_positives),
            "sampling/hard_fraction": sampling.achieved_hard_fraction,
            "sampling/repeated_negatives": float(sampling.repeated_negatives),
            "sampling/empty_hard_pools": float(sampling.empty_hard_pools)
        }
        if diagnostics is not None and should_record_embeddings(
            epoch, True, diagnostic_settings.gnn_embedding_every_epochs
        ):
            metrics.update(diagnostics.observe(resources.model, resources.message_graph))

        append_metric_history(history_path, epoch, progress.global_step, metrics)
        run.log_metrics(metrics, progress.global_step)
        logger.info(
            "Epoch %d/%d | train %.5f | val %.5f | best %.5f (epoch %d) | "
            "lr %.3g | steps %d | skipped %d | %.1fs",
            epoch, settings.training.epochs, train_loss, val_loss,
            progress.best_loss, progress.best_epoch, learning_rate,
            progress.global_step, sampling.skipped_positives, duration
        )
        _save_epoch_checkpoints(
            config, resources, progress, checkpoint_dir, epoch, new_best, run.reference
        )
        if settings.training.early_stopping.enabled:
            if progress.stalled_epochs >= settings.training.early_stopping.patience:
                logger.info(
                    "Early stopping after %d epochs without validation improvement.",
                    progress.stalled_epochs
                )
                break
    return progress


def _record_selected_embeddings(
    config: ResolvedLinkPretrainConfig,
    resources: _TrainingResources,
    checkpoint_path: Path,
    run_dir: Path,
    progress: _TrainingProgress,
    run: TrackingRun
) -> None:
    """Record embedding health using the best-validation encoder weights.

    Args:
        config: Diagnostic sampling settings and fixed experiment seed.
        resources: Model and training-only message graph for encoding.
        checkpoint_path: Selected best-validation checkpoint to inspect.
        run_dir: Local metrics history directory.
        progress: Best epoch and current training progress.
        run: Backend-independent tracker for selected-model metrics.

    The checkpoint is loaded for evaluation only, without altering optimizer
    state. No held-out edges are introduced into the training message graph.
    """
    logging_settings = config.settings.tracking.logging
    if not logging_settings.gnn_embedding_enabled:
        return

    selected_checkpoint = load_checkpoint(
        checkpoint_path, resources.model, resources.metadata,
        resources.message_graph.x.device
    )
    selected_step = int(selected_checkpoint["global_step"])
    diagnostics = EmbeddingDiagnostics(
        sample_size=logging_settings.gnn_embedding_sample_size,
        seed=config.settings.experiment.seed,
        source=LinkRoadEmbeddingSource()
    )
    measures = diagnostics.observe(resources.model, resources.message_graph)
    selected = {
        "selected/" + name.removeprefix("train/"): value
        for name, value in measures.items()
    }
    append_metric_history(
        run_dir / "metrics.jsonl", progress.best_epoch, selected_step, selected
    )
    run.log_metrics(selected, selected_step)


def train_links(
    config: ResolvedLinkPretrainConfig,
    full_graph: Data,
    splits: tuple[Data, Data, Data],
    run_dir: Path,
    logger: logging.Logger,
    tracker: ExperimentTracker,
    resume_from: Path | None = None,
    fixed_samples: dict[str, LinkBatch] | None = None
) -> TrainingResult:
    """Train with fixed validation, resumable state, and injected tracking.

    Args:
        config: GNN, negative sampling, optimizer, and training settings.
        full_graph: Original CPU graph used only for false-negative exclusion.
        splits: ``(train, validation, test)`` link splits; only train message
            edges are used in the GNN, including held-out evaluation.
        run_dir: Existing run folder for checkpoints, JSONL, and Parquet.
        logger: Scoped console/file logger.
        tracker: ``ExperimentTracker`` port; never imported from MLflow here.
        resume_from: Optional trusted epoch-boundary checkpoint.
        fixed_samples: Optional validation/test batches reused by ``all``.

    Returns:
        Location and results of the lowest-validation-loss checkpoint.

    Raises:
        RuntimeError: If no training positives or best checkpoint is available.
        ValueError: If the checkpoint has already reached the epoch limit.
    """
    settings = config.settings
    torch.manual_seed(settings.experiment.seed)
    resources = _prepare_training(config, full_graph, splits, fixed_samples, logger)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = checkpoint_dir / "best.pt"
    progress = _TrainingProgress()
    previous_reference = read_tracking_reference(run_dir)

    if resume_from is not None:
        restored = load_checkpoint(
            resume_from, resources.model, resources.metadata,
            resources.message_graph.x.device,
            optimizer=resources.optimizer, scheduler=resources.scheduler
        )
        progress = _TrainingProgress(
            start_epoch=int(restored["epoch"]) + 1,
            global_step=int(restored["global_step"]),
            best_loss=float(restored["best_loss"]),
            best_epoch=int(restored["best_epoch"]),
            stalled_epochs=int(restored["epochs_without_improvement"])
        )
        previous_reference = restored.get("tracking_reference") or previous_reference
        logger.info("Resumed from epoch %d, step %d", progress.start_epoch - 1, progress.global_step)
    if progress.start_epoch > settings.training.epochs:
        raise ValueError("Checkpoint already reached the configured training epoch limit.")

    started = time.perf_counter()
    with tracker.start_run(previous=previous_reference) as run:
        write_tracking_reference(run_dir, run.reference)
        training_failed = False
        try:
            progress = _train_epochs(config, resources, progress, run_dir, run, logger)
            if not best_checkpoint.is_file():
                raise RuntimeError("No best-validation checkpoint available. Enable save_best.")
            _record_selected_embeddings(
                config, resources, best_checkpoint, run_dir, progress, run
            )
            logger.info(
                "Training finished in %.1fs | selected epoch %d | best val %.5f | checkpoint %s",
                time.perf_counter() - started, progress.best_epoch,
                progress.best_loss, best_checkpoint
            )
            run.log_artifact(best_checkpoint, "checkpoints")
            encoder_checkpoint = checkpoint_dir / "encoder.pt"
            if encoder_checkpoint.is_file():
                run.log_artifact(encoder_checkpoint, "checkpoints")
        except Exception:
            training_failed = True
            logger.exception("Link training failed; attempting to upload the error log")
            log_path = run_dir / "logs" / "train.log"
            if log_path.is_file():
                try:
                    run.log_artifact(log_path, "errors")
                except Exception:
                    logger.exception("Could not upload training failure log")
            raise
        finally:
            history = run_dir / "metrics.jsonl"
            if history.is_file():
                try:
                    parquet = export_metric_history(history, run_dir / "metrics.parquet")
                    if parquet is not None:
                        run.log_artifact(parquet, "metrics")
                    run.log_artifact(history, "metrics")
                except Exception:
                    if training_failed:
                        logger.exception("Could not export training metrics after failure")
                    else:
                        raise

    return TrainingResult(
        run_dir, best_checkpoint, progress.best_epoch,
        progress.best_loss, progress.global_step
    )


def test_links(
    config: ResolvedLinkPretrainConfig,
    full_graph: Data,
    splits: tuple[Data, Data, Data],
    checkpoint_path: Path,
    run_dir: Path,
    logger: logging.Logger,
    tracker: ExperimentTracker,
    fixed_samples: dict[str, LinkBatch] | None = None
) -> dict[str, float | int]:
    """Evaluate fixed held-out test links using a selected checkpoint only.

    Args:
        config: Settings matching the saved model and graph split.
        full_graph: Original road graph, only for false-negative exclusions.
        splits: Directed graph splits, with training-only message edges.
        checkpoint_path: Best-validation checkpoint (or an explicit path).
        run_dir: Folder for test results and complete metrics history.
        logger: Existing console/file logger.
        tracker: Provider-independent experiment tracking adapter.
        fixed_samples: Optional held-out batches reused by a combined run.

    Returns:
        Test loss, ranking metrics, retained count, and checkpoint epoch.
        Test results are never used to choose model weights.
    """
    settings = config.settings
    device = _resolve_device(config)
    training_graph = splits[0]
    message_graph = training_graph.clone().to(device)
    model = build_link_prediction_model(training_graph, config).to(device)
    saved = load_checkpoint(
        checkpoint_path, model, model_metadata(config, training_graph), device
    )
    if fixed_samples is None:
        fixed_samples = build_fixed_evaluation(
            full_graph, splits, settings.pretraining.negatives, settings.experiment.seed
        )
    metrics = evaluate_link_metrics(
        model, message_graph, fixed_samples["test"], settings.pretraining.objective,
        settings.pretraining.negatives.evaluation_chunk_size
    )
    result = {
        "test_loss": metrics["loss"],
        "test_pairwise_accuracy": metrics["pairwise_accuracy"],
        "test_hits_at_1": metrics["hits_at_1"],
        "test_mrr": metrics["mrr"],
        "checkpoint_epoch": int(saved["epoch"]),
        "test_retained_positives": fixed_samples["test"].stats.retained_positives,
        "test_skipped_positives": fixed_samples["test"].stats.skipped_positives
    }
    output = run_dir / "test_metrics.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Held-out test loss %.5f | selected checkpoint epoch %d | retained %d links",
        metrics["loss"], saved["epoch"], result["test_retained_positives"]
    )
    history = run_dir / "metrics.jsonl"
    append_metric_history(
        history, int(saved["epoch"]), int(saved["global_step"]),
        {f"test/{name}": value for name, value in metrics.items()}
    )
    with tracker.start_run(
        previous=saved.get("tracking_reference") or read_tracking_reference(run_dir)
    ) as run:
        write_tracking_reference(run_dir, run.reference)
        run.log_metrics({f"test/{name}": value for name, value in metrics.items()}, int(saved["global_step"]))
        run.log_artifact(output, "metrics")
        parquet = export_metric_history(history, run_dir / "metrics.parquet")
        if parquet is not None:
            run.log_artifact(parquet, "metrics")
        run.log_artifact(history, "metrics")
    return result
