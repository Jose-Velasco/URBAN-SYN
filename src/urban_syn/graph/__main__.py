"""Build and validate a YAML-configured road graph from the command line."""
import argparse
import logging
from pathlib import Path

import torch
from torch_geometric.data import Data

from .config import build_dataset_from_config, load_graph_config
from .selection import select_node_features
from .utils import graph_info

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Read graph CLI configuration path and cache-rebuild flag.

    Returns
    -------
    argparse.Namespace
        Parsed command-line options.
    """
    parser = argparse.ArgumentParser(description="Build a canonical PyG road graph.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--force-reload", action="store_true")
    return parser.parse_args()


def validate_graph(graph: Data) -> None:
    """Check road IDs, topology, coordinates, and numeric features.

    Parameters
    ----------
    graph : Data
        Cached PyG road graph.

    Raises
    ------
    ValueError
        If essential graph attributes are invalid.
    """
    n = graph.num_nodes

    if graph.x is None or graph.x.ndim != 2 or graph.x.size(0) != n:
        raise ValueError("Node-feature matrix has invalid shape.")
    if graph.edge_id.numel() != n:
        raise ValueError("Canonical edge IDs do not match node count.")
    if graph.pos.shape != (n, 2):
        raise ValueError("GPS midpoint positions must have shape [N, 2].")
    if not torch.isfinite(graph.x).all() or not torch.isfinite(graph.pos).all():
        raise ValueError("Node features or GPS positions are not finite.")
    if graph.edge_index.ndim != 2 or graph.edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, E].")
    if graph.edge_index.numel() and (
        graph.edge_index.min() < 0 or graph.edge_index.max() >= n
    ):
        raise ValueError("Found out-of-range graph edge indices.")


def main() -> None:
    """Prepare the graph and optionally inspect selected model-input features."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    config = load_graph_config(args.config)
    dataset = build_dataset_from_config(config, force_reload=args.force_reload)
    
    graph = dataset[0]
    validate_graph(graph)
    
    logger.info("=== GRAPH SUMMARY ===")
    logger.info("Road nodes: %s", f"{graph.num_nodes:,}")
    logger.info("Directed transitions: %s", f"{graph.num_edges:,}")
    logger.info("Base features: %s", tuple(graph.x.shape))
    logger.info("GPS positions: %s (lon, lat)", tuple(graph.pos.shape))
    logger.info("Feature names: %s", dataset.preprocessor.get_feature_names_out().tolist())
    logger.info("Has isolated nodes: %s", graph.has_isolated_nodes())

    model = config.get("model", {})
    if model:
        chosen = model.get("node_features", ["x"])
        if not isinstance(chosen, list) or not all(isinstance(s, str) for s in chosen):
            raise ValueError("model.node_features must be a list of attribute names.")
        
        selected = select_node_features(graph, chosen)
        logger.info("Selected node feature GNN inputs: %s -> %s", chosen, tuple(selected.x.shape))
        logger.info("=== Selected node feature Summary ===")
        logger.info("\n%s", graph_info(selected))

    logger.info("Graph validation passed.")

if __name__ == "__main__":
    main()
