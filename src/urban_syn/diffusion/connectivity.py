"""Differentiable BCE and BPR connectivity guidance for predicted road vectors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

from .connectivity_validation import (
    valid_consecutive_road_pairs,
    validate_negative_embedding_table
)
from .losses import LossContext

if TYPE_CHECKING:
    from urban_syn.link_pretrain.model import DirectionalLinkDecoder


class ConnectivityBCE:
    """Encourage predicted consecutive road vectors to score as connected."""

    def __init__(self, scorer: DirectionalLinkDecoder) -> None:
        """Use a pretrained frozen scorer with differentiable input scores.

        Args:
            scorer: Frozen directional MLP with ``score_pairs`` support.
        """
        self.scorer = scorer

    def __call__(self, context: LossContext) -> Tensor:
        """Compute positive-label BCE over consecutive nonpadding positions.

        Args:
            context: Diffusion predictions and batch road IDs.

        Returns:
            Differentiable scalar. Empty valid-pair batches yield connected zero.
        """
        predicted_roads = context.diffusion.reconstructed[..., :-1]
        model_road_ids = context.batch.get("road_ids")
        mask = valid_consecutive_road_pairs(
            predicted_roads, model_road_ids, context.road_id_base
        )
        if not mask.any():
            return predicted_roads.sum() * 0.0
        source = predicted_roads[:, :-1, :][mask]
        destination = predicted_roads[:, 1:, :][mask]
        return F.softplus(-self.scorer.score_pairs(source, destination)).mean()


class ConnectivityBPR:
    """Rank predicted consecutive road pairs above sampled graph nonedges.

    The scorer's parameters stay frozen. The generated pair is the ranking
    positive, while each negative destination uses a GNN embedding selected
    by the graph-aware sampler from the ground-truth source road.
    """

    def __init__(self, scorer: DirectionalLinkDecoder) -> None:
        """Configure pretrained connectivity ranking guidance.

        Args:
            scorer: Frozen BPR-pretrained directional connectivity scorer.
        """
        self.scorer = scorer

    def __call__(self, context: LossContext) -> Tensor:
        """Compute pairwise BPR against each retained graph-negative candidate.

        Args:
            context: Predicted clean vectors, GNN road embedding table,
                source road IDs, and prepared negative IDs and masks.

        Returns:
            Scalar mean of ``softplus(negative_logit - generated_pair_logit)``.
            If no valid negative pair is available, returns connected zero.
        """
        predicted_roads = context.diffusion.reconstructed[..., :-1]
        model_road_ids = context.batch.get("road_ids")
        mask = valid_consecutive_road_pairs(
            predicted_roads, model_road_ids, context.road_id_base
        )
        validate_negative_embedding_table(
            predicted_roads, context.road_embeddings, context.road_id_base
        )
        if not mask.any():
            return predicted_roads.sum() * 0.0

        keep = context.batch.get("negative_retained_mask")
        negative_ids = context.batch.get("negative_model_ids")
        if not isinstance(keep, Tensor) or not isinstance(negative_ids, Tensor):
            raise ValueError("BPR requires negative samples prepared by the DataLoader.")
        source = predicted_roads[:, :-1, :][mask]
        if keep.dtype != torch.bool or keep.shape != (source.shape[0],):
            raise ValueError("BPR retained mask must match the valid source positions.")
        if negative_ids.dtype != torch.long or negative_ids.ndim != 2:
            raise ValueError("BPR negative IDs must be a two-dimensional long tensor.")
        if negative_ids.shape[1] < 1:
            raise ValueError("BPR requires at least one negative per retained source.")
        source = source[keep]
        if source.shape[0] != negative_ids.shape[0]:
            raise ValueError("BPR negative IDs must match the retained source positions.")
        if source.shape[0] == 0:
            return predicted_roads.sum() * 0.0
        destination = predicted_roads[:, 1:, :][mask][keep]
        negative_embeddings = context.road_embeddings[negative_ids]

        positive_logits = self.scorer.score_pairs(source, destination).unsqueeze(1)
        source_repeated = source.unsqueeze(1).expand_as(negative_embeddings)
        negative_logits = self.scorer.score_pairs(source_repeated, negative_embeddings)
        return F.softplus(negative_logits - positive_logits).mean()
