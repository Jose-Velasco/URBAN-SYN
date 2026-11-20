"""Input checks shared by directional link scoring and link losses."""

from __future__ import annotations

import torch
from torch import Tensor


def validate_decoder_inputs(
    embeddings: Tensor,
    sources: Tensor,
    destinations: Tensor,
    embedding_dim: int
) -> None:
    """Validate the embeddings and road-index tensors used by the decoder.

    Args:
        embeddings: Floating-point road features with shape ``[N, embedding_dim]``.
        sources: Zero-based road indices with shape ``[P]`` or ``[P, amount]``.
        destinations: Destination indices with exactly the same shape as sources.
        embedding_dim: Width expected by the decoder.

    Raises:
        ValueError: If shapes, index dtypes, or devices are incompatible.

    Road-index bounds are checked by PyTorch during tensor indexing.
    """
    if embeddings.ndim != 2 or embeddings.size(1) != embedding_dim:
        raise ValueError(f"Road embeddings must have shape [N, {embedding_dim}].")

    if sources.shape != destinations.shape or sources.ndim not in (1, 2):
        raise ValueError("Source and destination indices must have matching 1D or 2D shapes.")

    if sources.dtype != torch.long or destinations.dtype != torch.long:
        raise ValueError("Source and destination indices must use torch.long.")

    if sources.device != embeddings.device or destinations.device != embeddings.device:
        raise ValueError("Road embeddings and indices must be on the same device.")


def validate_link_logits(positive_logits: Tensor, negative_logits: Tensor) -> None:
    """Validate source-aligned positive and negative link logits.

    Args:
        positive_logits: Raw floating-point scores of shape ``[P]`` with ``P > 0``.
        negative_logits: Scores of shape ``[P, amount]`` with ``amount > 0``.
            Row ``i`` contains negatives paired with positive ``i``.

    Raises:
        ValueError: If either tensor is empty, incorrectly shaped, has an
            incompatible dtype, or is on a different device.
    """
    if positive_logits.ndim != 1 or positive_logits.numel() == 0:
        raise ValueError("Positive logits must have shape [P] with P > 0.")

    if negative_logits.ndim != 2 or negative_logits.shape[0] != positive_logits.numel():
        raise ValueError("Negative logits must have shape [P, amount].")

    if negative_logits.shape[1] == 0:
        raise ValueError("Every positive link must have at least one negative.")

    if not positive_logits.is_floating_point() or not negative_logits.is_floating_point():
        raise ValueError("Link logits must be floating-point tensors.")

    if positive_logits.dtype != negative_logits.dtype:
        raise ValueError("Positive and negative logits must have the same dtype.")

    if positive_logits.device != negative_logits.device:
        raise ValueError("Positive and negative logits must use the same device.")
