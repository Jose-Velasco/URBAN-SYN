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
