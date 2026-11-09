"""Select cached node feature groups for model input without altering the dataset."""
from collections.abc import Sequence

import torch
from torch_geometric.data import Data


def select_node_features(
    graph: Data,
    feature_names: Sequence[str],
    drop_selected: bool = True,
) -> Data:
    """Clone a PyG graph and combine selected node groups into ``x``.

    Parameters
    ----------
    graph : Data
        Cached road graph containing node-aligned feature groups.
    feature_names : Sequence[str]
        Attributes to concatenate in model-input order.
    drop_selected : bool, default=True
        Remove selected auxiliary groups from the resulting graph.

    Returns
    -------
    Data
        Independent cloned graph with selected features in ``x``.

    Raises
    ------
    ValueError
        If a requested feature group is missing or has incorrect shape.
    """
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise ValueError("Select one or more unique node feature groups.")
    
    selected = []
    for name in feature_names:
        value = getattr(graph, name, None)
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.size(0) != graph.num_nodes:
            raise ValueError(f"Missing or invalid node feature group: {name}")
        selected.append(value)
    
    result = graph.clone()
    result.x = selected[0].clone() if len(selected) == 1 else torch.cat(selected, dim=-1)
    
    if drop_selected:
        for name in feature_names:
            if name != "x":
                delattr(result, name)
    
    return result

def select_edge_features(
    graph: Data,
    feature_names: Sequence[str],
    drop_selected: bool = True,
) -> Data:
    """Clone a graph and combine selected physical-edge groups into ``edge_attr``.

    Parameters
    ----------
    graph : Data
        Road graph with features aligned to edge_index.
    feature_names : Sequence[str]
        Edge feature attributes in model-input order. Empty disables edge input.
    drop_selected : bool, default=True
        Remove auxiliary groups after selection.

    Returns
    -------
    Data
        Cloned graph with the selected edge features.

    Raises
    ------
    ValueError
        If an edge feature is missing or incorrectly shaped.
    """
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("Select unique edge feature groups.")

    groups = []
    for name in feature_names:
        value = getattr(graph, name, None)
        
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.size(0) != graph.num_edges:
            raise ValueError(f"Missing or invalid edge feature group: {name}")
        groups.append(value)

    result = graph.clone()
    result.edge_attr = torch.cat(groups, dim=-1) if groups else None
    
    if drop_selected:
        for name in feature_names:
            if name != "edge_attr":
                delattr(result, name)
    
    return result

def select_graph_features(
    graph: Data,
    node_features: Sequence[str],
    edge_features: Sequence[str] = (),
    drop_selected: bool = True,
) -> Data:
    """Select node and edge feature groups with a single graph clone.

    Parameters
    ----------
    graph : Data
        Road graph containing precomputed feature groups.
    node_features : Sequence[str]
        Node features to concatenate into x.
    edge_features : Sequence[str]
        Edge features to concatenate into edge_attr.
    drop_selected : bool, default=True
        Remove selected auxiliary attributes from the result.

    Returns
    -------
    Data
        Cloned graph with selected GNN inputs.
    """
    if not node_features or len(node_features) != len(set(node_features)):
        raise ValueError("Select one or more unique node feature groups.")

    if len(edge_features) != len(set(edge_features)):
        raise ValueError("Select unique edge feature groups.")

    # Validate feature groups before cloning the potentially large graph.
    def collect(names: Sequence[str], rows: int) -> list[torch.Tensor]:
        groups = []
        for name in names:
            value = getattr(graph, name, None)
            
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 2
                or value.size(0) != rows
            ):
                raise ValueError(f"Missing or invalid feature group: {name}")
            
            groups.append(value)
        return groups

    nodes = collect(node_features, graph.num_nodes)
    edges = collect(edge_features, graph.num_edges)

    selected = graph.clone()

    selected.x = torch.cat(nodes, dim=-1)
    selected.edge_attr = torch.cat(edges, dim=-1) if edges else None

    if drop_selected:
        for name in node_features:
            if name != "x":
                delattr(selected, name)

        for name in edge_features:
            if name != "edge_attr":
                delattr(selected, name)

    return selected