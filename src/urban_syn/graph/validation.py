"""Validate graph features selected for GNN model inputs."""
from collections.abc import Sequence

import torch
from torch import Tensor
from torch_geometric.data import Data


def _validate_feature_group(
    graph: Data,
    names: Sequence[str],
    expected_rows: int,
    kind: str,
) -> None:
    """Check the shape and numeric validity of selected graph features.

    Parameters
    ----------
    graph : Data
        PyG road graph containing the feature attributes.
    names : Sequence[str]
        Names of the features selected for the GNN.
    expected_rows : int
        Number of road nodes or physical transitions.
    kind : str
        Feature group label used in validation messages.

    Raises
    ------
    ValueError
        If a selected feature is missing or invalid.
    """
    for name in names:
        features = getattr(graph, name, None)

        if not isinstance(features, Tensor):
            raise ValueError(f"{kind} feature '{name}' is missing or not a tensor.")

        # Each feature group must contribute one row per node or physical edge.
        if features.ndim != 2 or features.size(0) != expected_rows:
            raise ValueError(
                f"{kind} feature '{name}' must have shape "
                f"[{expected_rows}, F], got {tuple(features.shape)}."
            )

        if features.size(1) == 0 or not torch.isfinite(features).all():
            raise ValueError(f"{kind} feature '{name}' is empty or non-finite.")


def validate_graph_features(
    graph: Data,
    node_features: Sequence[str] = ("x",),
    edge_features: Sequence[str] = (),
) -> None:
    """Validate canonical road IDs and selected GNN input features.

    Parameters
    ----------
    graph : Data
        PyG road graph.
    node_features : Sequence[str]
        Node feature attributes selected for training.
    edge_features : Sequence[str]
        Edge feature attributes selected for training.

    Raises
    ------
    ValueError
        If road IDs are duplicated or selected features are invalid.
    """
    if not isinstance(graph.edge_id, Tensor):
        raise ValueError("Canonical road IDs are missing or not a tensor.")

    if graph.edge_id.numel() != graph.num_nodes:
        raise ValueError("Canonical road IDs must match the number of nodes.")

    # Each PyG node represents exactly one canonical road segment.
    if torch.unique(graph.edge_id).numel() != graph.num_nodes:
        raise ValueError("Canonical road IDs must be unique.")

    _validate_feature_group(graph, node_features, graph.num_nodes, "Node")
    _validate_feature_group(graph, edge_features, graph.num_edges, "Edge")
