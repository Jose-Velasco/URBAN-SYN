"""Adapt graph-valid negative sampling to consecutive diffusion road predictions.

The structural sampler receives real outgoing graph links as its input contract.
Those links are sampling anchors only. The guided positive pair is always the
consecutive pair of predicted clean embeddings from the diffusion model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from urban_syn.link_pretrain.config import NegativeSamplingSettings
from urban_syn.link_pretrain.graph import FullPositiveIndex
from urban_syn.link_pretrain.sampling import (
    NoValidNegativesError,
    TrainingPositiveSelector,
    build_negative_sampler
)


@dataclass(frozen=True)
class DiffusionNegativeBatch:
    """Negatives matched to the retained source positions of one diffusion batch.

    Attributes:
        retained_mask: Boolean CPU vector selecting source positions with valid
            sampled negatives. Its length equals the requested source count.
        negative_model_ids: Long CPU tensor of shape [retained, amount]. Values
            are one-based model road IDs, never zero/padding.
    """

    retained_mask: Tensor
    negative_model_ids: Tensor


class DiffusionNegativeSampler:
    """Reuse pretraining K-hop/mixed samplers for trajectory-source road IDs.

    The source IDs are taken from ground-truth diffusion tokens, but no claim
    is made that the next observed opath token is a direct graph neighbor.
    Repeated source roads share one sampled negative row within a batch.
    """

    def __init__(
        self,
        graph_edge_index: Tensor,
        num_nodes: int,
        settings: NegativeSamplingSettings,
        seed: int
    ) -> None:
        """Build read-only CPU sampling helpers from the current joint graph.

        Args:
            graph_edge_index: Full directed edge index in zero-based node order.
            num_nodes: Number of road nodes excluding the model padding ID.
            settings: Existing K-hop or mixed negative-sampling settings.
            seed: Nonnegative experiment seed used for deterministic requests.
        """
        if seed < 0:
            raise ValueError("Negative-sampling seed cannot be negative.")
        edges = graph_edge_index.detach().to(device="cpu", dtype=torch.long)
        full_positives = FullPositiveIndex(edges, num_nodes)
        self.positive_selector = TrainingPositiveSelector(edges, num_nodes)
        self.negative_sampler = build_negative_sampler(
            edges, full_positives, num_nodes, settings
        )
        self.num_nodes = num_nodes
        self.amount = settings.amount
        self.seed = seed
        self.eligible = torch.zeros(num_nodes, dtype=torch.bool)
        self.eligible[self.positive_selector.eligible_seeds] = True

    def sample(self, model_source_ids: Tensor, step: int) -> DiffusionNegativeBatch:
        """Draw valid negative road IDs aligned to requested source positions.

        Args:
            model_source_ids: One-dimensional tensor of nonpadding model IDs,
                on either CPU or the training device.
            step: Nonnegative optimizer step, or validation batch index. Used
                with the experiment seed for reproducible sampling and resume.

        Returns:
            A CPU mask of retained source positions and corresponding
            one-based negative model IDs. Sources with no outgoing graph edge
            or no available negatives are omitted without relabeling them.
        """
        if model_source_ids.ndim != 1 or model_source_ids.dtype != torch.long:
            raise ValueError("Sampling requires a one-dimensional long source tensor.")
        if step < 0:
            raise ValueError("Negative-sampling step cannot be negative.")
        source_nodes = model_source_ids.detach().to(device="cpu") - 1
        if bool(((source_nodes < 0) | (source_nodes >= self.num_nodes)).any()):
            raise ValueError("Sampling sources must be nonpadding model road IDs.")

        empty = DiffusionNegativeBatch(
            torch.zeros(source_nodes.numel(), dtype=torch.bool),
            torch.empty((0, self.amount), dtype=torch.long)
        )
        if source_nodes.numel() == 0:
            return empty

        unique_sources = torch.unique(source_nodes, sorted=True)
        eligible_sources = unique_sources[self.eligible[unique_sources]]
        if eligible_sources.numel() == 0:
            return empty

        generator = torch.Generator(device="cpu").manual_seed(self.seed + step)
        sampled_positives = self.positive_selector.select(eligible_sources, generator)
        try:
            sampled = self.negative_sampler.sample(
                eligible_sources, sampled_positives, generator
            )
        except NoValidNegativesError:
            return empty

        # Existing samplers retain source order but may skip individual source
        # roads with no negatives. Since sources are unique and sorted, a
        # vectorized search maps all original positions back without ambiguity.
        retained_sources = sampled.positive_src
        positions = torch.searchsorted(retained_sources, source_nodes)
        safe_positions = positions.clamp(max=retained_sources.numel() - 1)
        retained_mask = (positions < retained_sources.numel()) & (
            retained_sources[safe_positions] == source_nodes
        )
        negative_ids = sampled.negative_dst[safe_positions[retained_mask]] + 1
        return DiffusionNegativeBatch(retained_mask, negative_ids)
