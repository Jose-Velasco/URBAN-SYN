"""Stage-oriented CLI for training and evaluating road-link prediction."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from .config import ResolvedLinkPretrainConfig, load_link_pretrain_config
from .graph import (
    graph_paths, load_pretraining_graph, split_directed_graph, summarize_split,
    validate_graph_cache, validate_road_graph
)
from .log_utils import run_logging
from .stages import TARGETS, stages_for_target
from urban_syn.pipeline.runner import PipelineRunner

if TYPE_CHECKING:
    from torch_geometric.data import Data
    from .data import LinkBatch


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI without importing heavy graph/training dependencies."""
    parser = argparse.ArgumentParser(description="Run URBAN-SYN road-link pretraining stages.")
    parser.add_argument("target", choices=TARGETS)
    parser.add_argument("--config", required=True, help="Standalone pretraining YAML")
    parser.add_argument("--project-root", default=None, help="Repository root (normally inferred)")
    parser.add_argument(
        "--run-dir", default=None,
        help="Prepared run directory, relative to the project root"
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Checkpoint to resume training or evaluate; paths relative to the project root"
    )
    parser.add_argument(
        "--result-file", default=None,
        help="Optional JSON result path for recording a successfully completed training run"
    )
    options = parser.add_mutually_exclusive_group()
    options.add_argument(
        "--dry-run", action="store_true", help="Show plan without loading or writing data"
    )
    options.add_argument(
        "--validate-only", action="store_true",
        help="Validate inputs without writing artifacts"
    )
    return parser


def _project_path(value: str, project_root: Path) -> Path:
    """Interpret requested paths independently of YAML directory nesting."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else project_root / path).resolve()


def _run_directory(config: ResolvedLinkPretrainConfig, requested: str | None) -> Path:
    """Choose a unique run for a new prepare or an explicit existing run."""
    if requested:
        return _project_path(requested, config.project_root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:12]
    return config.settings.paths.output_root / config.settings.experiment.name / f"{stamp}-{suffix}"


def _print_plan(
    config: ResolvedLinkPretrainConfig, args: argparse.Namespace, logger: logging.Logger
) -> None:
    """Log the resolved experiment plan without accessing graph data.

    Args:
        config: Validated experiment and graph configuration.
        args: Parsed CLI arguments, including optional paths.
        logger: Scoped console/file logger used to emit the plan.
    """
    settings = config.settings
    split = settings.pretraining.graph_split
    negatives = settings.pretraining.negatives
    network, caches = graph_paths(config)

    logger.info("Graph YAML: %s", settings.paths.graph_config)
    logger.info("Network: %s", network)
    logger.info("Cache: %s", caches[0].parent)
    architecture = config.graph_settings["model"]["gnn"].get("architecture", "gatv2")
    logger.info("GNN architecture: %s", architecture)
    train_fraction = 1 - split.val_fraction - split.test_fraction
    logger.info("Graph positive link split (train/val/test): %s/%s/%s",
                f"{train_fraction:.0%}", f"{split.val_fraction:.0%}", f"{split.test_fraction:.0%}")
    logger.info("Disjoint training supervision: %.0f%% of training positives",
                split.disjoint_train_ratio * 100)
    logger.info("Negatives: %s, K=%d, amount=%d, hard_fraction=%g",
                negatives.strategy, negatives.k_hops, negatives.amount, negatives.hard_fraction)
    logger.info("Objective: %s", settings.pretraining.objective)
    logger.info("Training: %d epochs, %d seed roads/batch",
                settings.training.epochs, settings.training.batch_size)
    logger.info("Scheduler: %s, early stopping: %s, patience=%d",
                settings.training.scheduler.name, settings.training.early_stopping.enabled,
                settings.training.early_stopping.patience)
    logger.info(
        "Runtime: device=%s, workers=%d",
        settings.runtime.device, settings.runtime.num_workers
    )
    logger.info("MLflow: %s", "enabled" if settings.tracking.enabled else "disabled")
    if args.run_dir:
        logger.info("Run directory: %s", _project_path(args.run_dir, config.project_root))
    if args.checkpoint:
        logger.info("Checkpoint: %s", _project_path(args.checkpoint, config.project_root))


def _validate(config: ResolvedLinkPretrainConfig, logger: logging.Logger) -> Data:
    """Load the cached road graph and check its structural invariants.

    Args:
        config: Validated configuration for graph loading.
        logger: Scoped logger for dimensions and selected feature counts.

    Returns:
        Validated CPU PyG ``Data`` graph, reused by split preparation.
    """
    validate_graph_cache(config)
    graph = load_pretraining_graph(config)
    validate_road_graph(graph)
    edge_attr = getattr(graph, "edge_attr", None)
    edge_width = None if edge_attr is None else edge_attr.size(1)
    logger.info("Graph validated: nodes=%s, edges=%s, node_features=%d, edge_features=%s",
                f"{graph.num_nodes:,}", f"{graph.num_edges:,}", graph.x.size(1), edge_width)
    return graph


def _validation_log_file(config: ResolvedLinkPretrainConfig, args: argparse.Namespace) -> Path:
    """Choose the standalone validation log path."""
    if args.run_dir:
        return _project_path(args.run_dir, config.project_root) / "logs" / "validation.log"
    output_root = config.settings.paths.output_root
    experiment_name = config.settings.experiment.name
    return output_root / experiment_name / "logs" / "validation.log"


def _snapshot_config(config: ResolvedLinkPretrainConfig, run_dir: Path) -> None:
    """Save experiment and graph configuration snapshots for the run.

    Args:
        config: Resolved settings and source YAML paths.
        run_dir: Run directory in which to create the config snapshots.
    """
    config_folder = run_dir / "config"
    config_folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config.config_path, config_folder / "experiment.yaml")
    shutil.copyfile(config.settings.paths.graph_config, config_folder / "graph.yaml")
    (run_dir / "resolved_config.json").write_text(
        json.dumps(config.settings.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8"
    )


def _prepare(
    config: ResolvedLinkPretrainConfig,
    graph: Data,
    splits: tuple[Data, Data, Data],
    run_dir: Path,
    logger: logging.Logger
) -> None:
    """Record the already prepared split without regenerating its edges.

    Args:
        config: Validated settings and experiment seed.
        graph: Previously validated original CPU road graph.
        splits: Prepared training, validation, and test graph references.
        run_dir: Directory where configuration snapshots are written.
        logger: Console/file logger for split statistics.
    """
    summary = summarize_split(graph, splits, config.settings.experiment.seed)
    _snapshot_config(config, run_dir)
    logger.info("Directed link split validated in memory (seed=%d)", summary.seed)
    logger.info(
        "Full=%d, message=%d, training labels=%d, validation labels=%d, test labels=%d",
        summary.full_positive_count,
        summary.message_edge_count,
        summary.train_positive_count,
        summary.val_positive_count,
        summary.test_positive_count
    )
    logger.info(
        "Eligible training seeds=%d, nodes isolated in message graph=%d",
        summary.eligible_train_seeds,
        summary.isolated_message_nodes
    )


def _existing_run_folder(checkpoint: Path) -> Path:
    """Locate the run directory containing a checkpoints subdirectory."""
    if checkpoint.parent.name != "checkpoints":
        raise ValueError("Expected checkpoint under a run's checkpoints/ directory; pass --run-dir.")
    return checkpoint.parent.parent


def _write_training_result(run_dir: Path, best_checkpoint: Path, destination: Path) -> None:
    """Write the exact completed pretraining run for an external job launcher.

    This file is produced only after all requested stages succeed. It allows
    Slurm to publish a checkpoint reference without guessing output timestamps.
    """
    encoder = run_dir / "checkpoints" / "encoder.pt"
    if not encoder.is_file() or not best_checkpoint.is_file():
        raise FileNotFoundError("Cannot publish pretraining result: best.pt or encoder.pt is missing.")
    payload = {
        "format_version": 1,
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_checkpoint),
        "encoder_checkpoint": str(encoder),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _execution_paths(
    config: ResolvedLinkPretrainConfig, args: argparse.Namespace
) -> tuple[Path | None, Path | None]:
    """Resolve an output folder and optional resume/evaluation checkpoint.

    Dry-run and input-only validation do not create or require run folders.
    A standalone test needs an explicit prior run or checkpoint; it never
    silently starts a new run with an untrained model.
    """
    if args.dry_run or args.validate_only or args.target in ("validate", "prepare-graph"):
        return None, None

    checkpoint_option = args.checkpoint
    if checkpoint_option is None and args.target in ("train", "all"):
        resume = config.settings.training.checkpoint.resume_from
        checkpoint_option = str(resume) if resume is not None else None
    checkpoint = None
    if checkpoint_option is not None:
        checkpoint = _project_path(checkpoint_option, config.project_root)

    if args.run_dir is not None:
        return _project_path(args.run_dir, config.project_root), checkpoint
    if checkpoint is not None:
        return _existing_run_folder(checkpoint), checkpoint
    if args.target == "test":
        raise ValueError("A standalone test requires --run-dir or --checkpoint.")
    return _run_directory(config, None), None


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the selected pretraining pipeline command.

    Args:
        argv: Optional process argument sequence. Defaults to command line.

    Returns:
        Exit code ``0`` for successful execution and ``1`` for handled
        configuration, graph, training, or tracking errors.
    """
    args = build_parser().parse_args(argv)
    try:
        config = load_link_pretrain_config(args.config, args.project_root)
        run_dir, checkpoint = _execution_paths(config, args)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        with run_logging() as logger:
            logger.error("Configuration error: %s", exc)
        return 1

    log_file = None
    if run_dir is not None:
        # Use a single persistent training log for both train and all.
        log_name = "train.log" if args.target in ("train", "all") else f"{args.target}.log"
        log_file = run_dir / "logs" / log_name
    elif args.target == "validate" and not (args.dry_run or args.validate_only):
        log_file = _validation_log_file(config, args)

    with run_logging(log_file) as logger:
        try:
            stages = stages_for_target(config, args.target)
            _print_plan(config, args, logger)

            selected_graph = None
            selected_splits = None
            fixed_samples = None
            trained = None

            def prepare_graph_handler(experiment: ResolvedLinkPretrainConfig) -> None:
                """Reuse the joint pipeline's graph builder before model loading."""
                from urban_syn.experiment.graph_preparation import prepare_experiment_graph

                prepare_experiment_graph(experiment)

            def validate_handler(experiment: ResolvedLinkPretrainConfig) -> None:
                """Validate and retain graph tensors for subsequent handlers."""
                nonlocal selected_graph
                selected_graph = _validate(experiment, logger)

            def prepare_handler(experiment: ResolvedLinkPretrainConfig) -> None:
                """Split graph once, then record settings and split summary."""
                nonlocal selected_splits
                if selected_graph is None or run_dir is None:
                    raise RuntimeError("Prepare requires a validated graph and run directory.")
                selected_splits = split_directed_graph(
                    selected_graph,
                    experiment.settings.pretraining.graph_split,
                    experiment.settings.experiment.seed
                )
                _prepare(experiment, selected_graph, selected_splits, run_dir, logger)

            def train_handler(experiment: ResolvedLinkPretrainConfig) -> None:
                """Train the encoder/decoder using one split and fixed evaluation."""
                nonlocal fixed_samples, trained
                from .evaluation import build_fixed_evaluation
                from .training import train_links
                from .tracking import build_pretraining_tracker

                if selected_graph is None or selected_splits is None or run_dir is None:
                    raise RuntimeError("Training requires graph preparation and a run directory.")
                fixed_samples = build_fixed_evaluation(
                    selected_graph, selected_splits,
                    experiment.settings.pretraining.negatives,
                    experiment.settings.experiment.seed
                )
                tracker = build_pretraining_tracker(experiment, run_dir)
                trained = train_links(
                    experiment, selected_graph, selected_splits, run_dir, logger,
                    tracker, resume_from=checkpoint, fixed_samples=fixed_samples
                )

            def test_handler(experiment: ResolvedLinkPretrainConfig) -> None:
                """Evaluate the selected best model; never retrain for test."""
                nonlocal selected_splits
                # Keep heavier training and tracking imports inside the handler
                # so dry runs and graph-only validation do not load them.
                from .tracking import build_pretraining_tracker
                from .training import test_links

                if selected_graph is None or run_dir is None:
                    raise RuntimeError("Testing requires a validated graph and a run directory.")
                if selected_splits is None:
                    selected_splits = split_directed_graph(
                        selected_graph,
                        experiment.settings.pretraining.graph_split,
                        experiment.settings.experiment.seed
                    )

                # A combined run must test its newly selected best checkpoint.
                # Standalone testing may override that path explicitly.
                if trained is not None:
                    selected_checkpoint = trained.best_checkpoint
                elif checkpoint is not None:
                    selected_checkpoint = checkpoint
                else:
                    selected_checkpoint = run_dir / "checkpoints" / "best.pt"

                tracker = build_pretraining_tracker(experiment, run_dir)
                test_links(
                    experiment, selected_graph, selected_splits,
                    selected_checkpoint, run_dir, logger, tracker,
                    fixed_samples=fixed_samples
                )

            handlers = {
                "prepare-graph": prepare_graph_handler,
                "validate": validate_handler,
                "prepare": prepare_handler,
                "train": train_handler,
                "test": test_handler
            }
            PipelineRunner(config, handlers=handlers, emit=logger.info).run(
                stages, dry_run=args.dry_run, validate_only=args.validate_only
            )
            # Only announce a producing run after every requested stage has
            # completed; a failed held-out test must not publish its checkpoint.
            if trained is not None and args.result_file is not None:
                result_file = _project_path(args.result_file, config.project_root)
                _write_training_result(run_dir, trained.best_checkpoint, result_file)
            if args.dry_run:
                logger.info("Dry run finished. No graph was loaded or files written.")
            elif args.validate_only or args.target == "validate":
                logger.info("Input validation finished; no split or training was written.")
            return 0
        except Exception:
            logger.exception("Link pretraining stage failed")
            return 1
