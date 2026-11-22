"""GNN performance benchmark on the training message graph.

Profiles either road-GNN computation or CPU DataLoader link sampling.
Neither mode measures a complete pretraining step or link-prediction quality.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from torch.utils.data import DataLoader

from .config import load_link_pretrain_config
from .graph import load_pretraining_graph, prepare_link_split
from .loader import make_training_dataloader
from .log_utils import run_logging


def benchmark_sampling(
    loader: DataLoader,
    warmup: int,
    steps: int,
    logger: logging.Logger
) -> None:
    """Measure worker-produced link batches without invoking the road encoder.

    Args:
        loader: Shuffled CPU link-sampling DataLoader created from the
            training split and experiment settings.
        warmup: Batches consumed before timing, including worker startup.
        steps: Maximum number of batches to measure after warmup.
        logger: Scoped console/file logger for throughput and sample counts.

    Returns:
        None. Writes timing and sample-coverage measurements to ``logger``.

    Raises:
        ValueError: If warmup or steps is invalid, or no measured batch exists.
    """
    if warmup < 0 or steps < 1:
        raise ValueError("Sampling benchmark expects warmup >= 0 and steps >= 1.")

    iterator = iter(loader)
    for _ in range(warmup):
        try:
            next(iterator)
        except StopIteration as exc:
            raise ValueError("Sampling dataset is too small for the requested warmup.") from exc

    # Time retrieval from the iterator, including delays if workers cannot
    # fill the prefetch queue quickly enough for the training process.
    start = time.perf_counter()
    measured = 0
    requested = 0
    retained = 0
    hard = 0
    uniform = 0
    skipped = 0
    for _ in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        measured += 1
        requested += batch.stats.requested_positives
        retained += batch.stats.retained_positives
        skipped += batch.stats.skipped_positives
        hard += batch.stats.drawn_hard
        uniform += batch.stats.drawn_uniform

    elapsed = time.perf_counter() - start
    if measured == 0:
        raise ValueError("No link batches were available for the sampling benchmark.")

    logger.info("Sampling DataLoader: %d measured batches in %.3f s", measured, elapsed)
    logger.info(
        "Sampled positives: %d retained / %d requested (%d skipped)",
        retained, requested, skipped
    )
    logger.info("Average link-batch retrieval: %.3f s/batch", elapsed / measured)
    logger.info("Sampling throughput: %.1f retained positives/s", retained / elapsed)
    if hard + uniform > 0:
        logger.info("Achieved hard-negative fraction: %.1f%%", 100 * hard / (hard + uniform))
    else:
        logger.warning("No negatives were available in the measured batches.")


def run_benchmark(argv: list[str] | None = None) -> int:
    """Profile GNN computation or worker-side sampling on the training graph.

    Args:
        argv: Optional CLI arguments, including configuration path, device,
            number of warmup iterations, and measured iterations.

    Returns:
        Exit code ``0`` for a completed benchmark or ``1`` for handled errors.
        Logs timings and, depending on the selected mode, memory usage
        or sampled-link throughput. Neither mode reports link accuracy.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark the road GNN on training message edges."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--run-dir", required=True,
        help="Directory for benchmark logs (not split artifacts)"
    )
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Measured iterations (default: 3 for GNN, 20 for sampling)"
    )
    parser.add_argument(
        "--sampling-only",
        action="store_true",
        help="Benchmark CPU DataLoader link sampling instead of GNN computation"
    )
    args = parser.parse_args(argv)
    if args.steps is None:
        if args.sampling_only:
            args.steps = 20
        else:
            args.steps = 3

    config = load_link_pretrain_config(args.config, args.project_root)
    path = Path(args.run_dir).expanduser()
    if path.is_absolute():
        run_dir = path.resolve()
    else:
        run_dir = (config.project_root / path).resolve()

    if args.sampling_only:
        log_name = "sampling_benchmark.log"
    else:
        log_name = "gnn_benchmark.log"

    with run_logging(run_dir / "logs" / log_name) as logger:
        benchmark_log = logging.getLogger("urban_syn.gnn.benchmark")
        level, propagate = benchmark_log.level, benchmark_log.propagate
        for handler in logger.handlers:
            benchmark_log.addHandler(handler)
        benchmark_log.setLevel(logging.INFO)
        benchmark_log.propagate = False
        try:
            graph = load_pretraining_graph(config)
            splits, summary = prepare_link_split(
                graph, config.settings.pretraining.graph_split,
                config.settings.experiment.seed
            )
            train = splits[0]
            if args.sampling_only:
                loader = make_training_dataloader(graph, train, config.settings)
                logger.info(
                    "CPU sampling: %d eligible roads, batch size=%d, workers=%d",
                    len(loader.dataset),
                    config.settings.training.batch_size,
                    loader.num_workers
                )
                if loader.num_workers > 0:
                    queued_batches = loader.num_workers * loader.prefetch_factor
                    if args.steps <= queued_batches:
                        # A short measurement may only consume batches already
                        # waiting in the queue and overstate sustained speed.
                        logger.warning(
                            "Measure more than %d batches to exceed the initial prefetch queue.",
                            queued_batches
                        )
                benchmark_sampling(loader, args.warmup, args.steps, logger)
                logger.info("Sampling benchmark completed.")
                return 0

            from urban_syn.experiment.factories import resolve_device
            from urban_syn.gnn.benchmark import benchmark
            from urban_syn.gnn.config import parse_gnn_config
            from urban_syn.gnn.encoder import RoadGNNEncoder
            from urban_syn.gnn.factory import make_conv_factory

            settings = parse_gnn_config(config.graph_settings["model"]["gnn"])
            encoder = RoadGNNEncoder.from_graph(train, settings, make_conv_factory(settings))
            device = resolve_device(args.device)
            logger.info("Training-only message graph: %d nodes, %d directed edges",
                        train.num_nodes, train.num_edges)
            logger.info("Directed split seed=%d", summary.seed)
            logger.info("GNN: %s, selected device: %s", settings.architecture, device)
            benchmark(
                encoder,
                train.x,
                train.edge_index,
                getattr(train, "edge_attr", None),
                device=device,
                warmup=args.warmup,
                steps=args.steps,
                learning_rate=config.settings.training.optimizer.learning_rate
            )
            logger.info("GNN benchmark completed. Review recorded timing and memory.")
            return 0
        except (OSError, ValueError, RuntimeError, ImportError):
            logger.exception("Road-GNN or sampling benchmark failed")
            return 1
        finally:
            for handler in logger.handlers:
                benchmark_log.removeHandler(handler)
            benchmark_log.setLevel(level)
            benchmark_log.propagate = propagate


if __name__ == "__main__":
    raise SystemExit(run_benchmark())
