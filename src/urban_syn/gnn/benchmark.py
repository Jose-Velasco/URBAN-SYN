"""Benchmark standalone road GNN forwards and training steps on a cached graph.

Run from the repository root:
    uv run python -m urban_syn.gnn.benchmark --config configs/graph/nyc_graph.yaml
"""
import argparse
import logging
from pathlib import Path
from time import perf_counter

import torch
from torch import Tensor
from torch_geometric.nn import summary
from urban_syn.graph.config import build_dataset_from_config, load_graph_config
from urban_syn.graph.selection import select_graph_features
from urban_syn.graph.utils import graph_info
from urban_syn.graph.validation import validate_graph_features

from .encoder import RoadGNNEncoder
from .config import parse_gnn_config
from .factory import make_conv_factory

logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    """Parse benchmark settings without changing the cached graph."""
    parser = argparse.ArgumentParser(description="Benchmark the road GNN on a cached PyG graph.")
    
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    
    return parser.parse_args()

def _synchronize(device: torch.device) -> None:
    """Wait for asynchronous CUDA operations before timing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _device(value: str) -> torch.device:
    """Choose a device or report unavailable CUDA."""
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    return torch.device("cuda" if value == "cuda" or (value == "auto" and torch.cuda.is_available()) else "cpu")


def _inputs_from_config(config: dict) -> tuple[RoadGNNEncoder, Tensor, Tensor, Tensor | None]:
    """Load cached graph, select YAML feature groups, and construct the encoder."""
    dataset = build_dataset_from_config(config)
    graph = dataset[0]

    logger.info("\n%s", graph_info(graph))

    model_options = config.get("model", {})
    if not isinstance(model_options, dict) or "gnn" not in model_options:
        raise ValueError("YAML requires model.gnn for the encoder benchmark.")

    node_names = model_options.get("node_features", ["x"])
    edge_names = model_options.get("edge_features", [])
    validate_graph_features(graph, node_features=node_names, edge_features=edge_names)
    
    selected = select_graph_features(
        graph,
        node_features=node_names,
        edge_features=edge_names,
    )
    
    settings = parse_gnn_config(model_options["gnn"])
    encoder = RoadGNNEncoder.from_graph(selected, settings, make_conv_factory(settings))
    
    logger.info("Graph: %s road nodes, %s physical edges", f"{selected.num_nodes:,}", f"{selected.num_edges:,}")
    logger.info("Model: %s, input_dim=%d, edge_dim=%s, output_dim=%d",
                encoder.settings.architecture, encoder.input_dim, encoder.edge_dim,
                encoder.settings.output_dim)
    logger.info("Trainable parameters: %s", f"{sum(p.numel() for p in encoder.parameters()):,}")
    
    return encoder, selected.x, selected.edge_index, getattr(selected, "edge_attr", None)


def benchmark(
    encoder: RoadGNNEncoder,
    x: Tensor,
    edge_index: Tensor,
    edge_attr: Tensor | None,
    device: torch.device,
    warmup: int,
    steps: int,
    learning_rate: float,
) -> None:
    """Time inference and complete optimizer steps, logging peak GPU memory."""
    if warmup < 0 or steps < 1:
        raise ValueError("--warmup must be >= 0 and --steps must be >= 1.")

    encoder = encoder.to(device)
    # Move only model inputs. Graph metadata, coordinates, and IDs stay on CPU.
    x = x.to(device)
    
    edge_index = edge_index.to(device)
    edge_attr = None if edge_attr is None else edge_attr.to(device)

    # Inspect model layers and tensor shapes without tracking gradients.
    encoder.eval()
    with torch.no_grad():
        logger.info("\n%s", summary(encoder, x, edge_index, edge_attr))
    
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=learning_rate)

    def run_step(training: bool) -> float:
        """Run one measured forward or training step."""
        _synchronize(device)
        started = perf_counter()

        if training:
            encoder.train()
            optimizer.zero_grad(set_to_none=True)
            embeddings = encoder(x, edge_index, edge_attr)
            
            # Synthetic differentiable loss for timing only, not model evaluation.
            embeddings.square().mean().backward()
            optimizer.step()
        else:
            encoder.eval()
            with torch.no_grad():
                encoder(x, edge_index, edge_attr)

        _synchronize(device)
        return perf_counter() - started

    for training, label in ((False, "Inference"), (True, "Training step")):
        for _ in range(warmup):
            run_step(training)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        times = [run_step(training) for _ in range(steps)]
        logger.info("%s: average %.3f s (min %.3f s, max %.3f s)",
                    label, sum(times) / steps, min(times), max(times))

        if device.type == "cuda":
            mib = 1024 * 1024
            logger.info("Peak CUDA memory: allocated %.1f MiB, reserved %.1f MiB",
                        torch.cuda.max_memory_allocated(device) / mib,
                        torch.cuda.max_memory_reserved(device) / mib)


def main() -> None:
    """Run the cached-graph benchmark from CLI arguments."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()

    config = load_graph_config(args.config)

    encoder, x, edge_index, edge_attr = _inputs_from_config(config)
    device = _device(args.device)
    logger.info("Benchmark device: %s", device)

    try:
        benchmark(encoder, x, edge_index, edge_attr, device=device,
                  warmup=args.warmup, steps=args.steps, learning_rate=args.learning_rate)
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA out of memory. Try smaller GNN settings or run on HPC.")
        raise


if __name__ == "__main__":
    main()
