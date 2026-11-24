"""Differentiable BCE and BPR connectivity guidance for predicted road vectors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn.functional as F
from torch import Tensor

from .connectivity_validation import (
    valid_consecutive_road_pairs,
    validate_negative_embedding_table
)
from .losses import LossContext

if TYPE_CHECKING:
    from urban_syn.experiment.connectivity_sampling import DiffusionNegativeSampler
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

    def __init__(
        self,
        scorer: DirectionalLinkDecoder,
        negatives: DiffusionNegativeSampler
    ) -> None:
        """Configure pretrained connectivity ranking guidance.

        Args:
            scorer: Frozen BPR-pretrained directional connectivity scorer.
            negatives: Sampler of directed nonedges aligned to source positions.
        """
        self.scorer = scorer
        self.negatives = negatives

    def __call__(self, context: LossContext) -> Tensor:
        """Compute pairwise BPR against each retained graph-negative candidate.

        Args:
            context: Predicted clean vectors, GNN road embedding table,
                source road IDs, and deterministic sampling-step metadata.

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

        step = context.batch.get("sample_step", 0)
        if not isinstance(step, int) or step < 0:
            raise ValueError("BPR sampling requires a nonnegative integer sample_step.")

        model_road_ids = model_road_ids.long()
        source_ids = model_road_ids[:, :-1][mask]
        sampled = self.negatives.sample(source_ids, step)
        if not sampled.retained_mask.any():
            return predicted_roads.sum() * 0.0

        keep = sampled.retained_mask.to(device=predicted_roads.device)
        source = predicted_roads[:, :-1, :][mask][keep]
        destination = predicted_roads[:, 1:, :][mask][keep]
        negative_ids = sampled.negative_model_ids.to(device=predicted_roads.device)
        negative_embeddings = context.road_embeddings[negative_ids]

        positive_logits = self.scorer.score_pairs(source, destination).unsqueeze(1)
        source_repeated = source.unsqueeze(1).expand_as(negative_embeddings)
        negative_logits = self.scorer.score_pairs(source_repeated, negative_embeddings)
        return F.softplus(negative_logits - positive_logits).mean()
