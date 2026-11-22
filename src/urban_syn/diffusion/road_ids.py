"""Convert between FMM road IDs and Diff-RNTraj model IDs.

Important: Convert valid FMM road IDs first, then pad the resulting model-ID sequences with zeros.
This prevents confusion between the valid FMM edge_id=0 and diffusion padding ID 0.
"""
import torch
from torch import Tensor

def edge_ids_to_model_ids(
    edge_ids: Tensor,
    graph_edge_ids: Tensor,
) -> Tensor:
    """Convert FMM road IDs into one-based diffusion model IDs.

    Parameters
    ----------
    edge_ids : Tensor
        Road IDs from FMM, without padding.
    graph_edge_ids : Tensor
        Road IDs in PyG node order.

    Returns
    -------
    Tensor
        Diffusion model IDs, preserving the input shape. with ids starting at 1 since id 0 is reserved for padding in the diffusion model

    Raises
    ------
    ValueError
        If a road ID is missing or the graph contains duplicate IDs.
    """
    if graph_edge_ids.ndim != 1 or graph_edge_ids.numel() == 0:
        raise ValueError("graph_edge_ids must be a nonempty 1D tensor.")

    # Road IDs and PyG node indices are not necessarily identical.
    sorted_ids, node_indices = graph_edge_ids.to(edge_ids.device).long().sort()

    if torch.unique_consecutive(sorted_ids).numel() != sorted_ids.numel():
        raise ValueError("graph_edge_ids contains duplicates.")

    requested = edge_ids.long().reshape(-1)
    positions = torch.searchsorted(sorted_ids, requested)
    # to prevent index out of range
    safe_positions = positions.clamp(max=sorted_ids.numel() - 1)

    # searchsorted returns insertion positions, not necessarily exact matches.
    # basically handles validating road_ids are the same in both edge_ids and graph_edge_ids.
    # Make sure greater or equal to sorted_ids.numel() mean there exists an element that both dont agree since it would be inserted at end
    # make sure both have same road_ids in edge_ids and graph_edge_ids.
    matched = (positions < sorted_ids.numel()) & (
        sorted_ids[safe_positions] == requested
    )

    if not matched.all():
        raise ValueError("Some road IDs are missing from the PyG graph.")

    # Add one because diffusion model ID zero is reserved for padding, thus ids start at 1 instead of 0.
    return (node_indices[safe_positions] + 1).reshape(edge_ids.shape)


def model_ids_to_edge_ids(
    model_ids: Tensor,
    graph_edge_ids: Tensor,
) -> Tensor:
    """Convert diffusion model IDs back to road IDs.

    graph_edge_ids: Tensor holds the original shared edge_id that are used to map back to he road network

    Padding ID zero is returned as -1.
    """
    if graph_edge_ids.ndim != 1:
        raise ValueError("graph_edge_ids must be a 1D tensor.")

    if (model_ids < 0).any() or (model_ids > graph_edge_ids.numel()).any():
        raise ValueError("Diffusion model ID is out of range.")

    # results is the output that will hold the edge_ids without the added 0 padding
    result = torch.full_like(model_ids, -1, dtype=torch.long)
    # excludes the 0 padding index since its id is 0
    valid = model_ids > 0

    # minus 1 since each id in graph_edge_ids was increased by 1 in in edge_ids_to_model_ids
    # this selects the road ids of the ones model_ids (basically model_ids are the ideas we want the original edge_ids)
    # this could be all or a subset of all actual edge_ids if model_ids has all the model_ids
    result[valid] = graph_edge_ids.to(model_ids.device).long()[
        model_ids[valid].long() - 1
    ]

    return result

def encode_transition_keys(
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
    road_id_base: int,
) -> torch.Tensor:
    """Encode directed pairs of model road IDs as unique integer keys.

    Parameters
    ----------
    source_ids, target_ids : Tensor
        Source and destination model IDs with matching shapes.
    road_id_base : int
        Number of model road IDs, including padding.

    Returns
    -------
    Tensor
        Encoded transition keys with the same input shape.
    """
    return source_ids.long() * road_id_base + target_ids.long()
