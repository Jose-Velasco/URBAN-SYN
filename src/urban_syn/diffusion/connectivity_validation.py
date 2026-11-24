"""Validate the trajectory shapes and road IDs used by connectivity losses."""

from __future__ import annotations

import torch
from torch import Tensor


def valid_consecutive_road_pairs(
    predicted_roads: Tensor,
    model_road_ids: Tensor | None,
    road_id_base: int
) -> Tensor:
    """Return a mask for consecutive non-padding positions in a trajectory batch.

    Args:
        predicted_roads: Predicted clean road vectors with shape ``[B, T, D]``.
        model_road_ids: Source trajectory road IDs with shape ``[B, T]``.
            Model ID ``0`` indicates padding; valid roads use IDs ``1..N``.
        road_id_base: Number of model road IDs, including the padding ID.

    Returns:
        Boolean tensor of shape ``[B, T - 1]``. A pair is selected only
        when both source positions correspond to valid roads.

    Raises:
        ValueError: If a shape, device, dtype, or road ID is invalid.
    """
    if predicted_roads.ndim != 3 or predicted_roads.size(-1) < 1:
        raise ValueError("Predicted road vectors must have shape [B, T, D] with D >= 1.")
    if model_road_ids is None:
        raise ValueError("Connectivity guidance requires batch['road_ids'].")
    if not isinstance(model_road_ids, Tensor):
        raise ValueError("batch['road_ids'] must be a tensor.")
    if model_road_ids.ndim != 2 or model_road_ids.shape != predicted_roads.shape[:2]:
        raise ValueError("Road IDs must have shape [B, T] matching predicted road vectors.")
    if model_road_ids.device != predicted_roads.device:
        raise ValueError("Road IDs and predicted road vectors must be on the same device.")
    if model_road_ids.dtype not in (
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
    ):
        raise ValueError("Road IDs must use an integer tensor dtype.")
    if road_id_base < 2:
        raise ValueError("road_id_base must include padding and at least one road.")
    if ((model_road_ids < 0) | (model_road_ids >= road_id_base)).any():
        raise ValueError("Road IDs must be in the valid model vocabulary, including padding 0.")

    valid = model_road_ids.ne(0)
    return valid[:, :-1] & valid[:, 1:]


def validate_negative_embedding_table(
    predicted_roads: Tensor,
    embedding_table: Tensor,
    road_id_base: int
) -> None:
    """Check the padded clean GNN embedding table used for BPR negatives.

    Args:
        predicted_roads: Road vectors with shape ``[B, T, D]``.
        embedding_table: Model-ID-indexed GNN vectors of shape ``[N + 1, D]``.
        road_id_base: Total vocabulary size, including padding road ID zero.

    Raises:
        ValueError: If the table disagrees with the model vocabulary or width.
    """
    if (embedding_table.ndim != 2
            or embedding_table.shape != (road_id_base, predicted_roads.size(-1))):
        raise ValueError("BPR negative road embeddings must have shape [road_id_base, D].")
    if embedding_table.device != predicted_roads.device:
        raise ValueError("BPR negative embeddings and predictions must share a device.")
