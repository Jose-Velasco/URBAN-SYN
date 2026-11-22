"""In-memory held-out link examples for fixed validation and final testing.

Each training run creates these once from its seeded directed split and keeps
these batches unchanged through all validation epochs. Nothing is persisted
outside normal model checkpoints.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .config import NegativeSamplingSettings
from .data import LinkBatch, SamplingStats
from .graph import FullPositiveIndex
from .sampling import NegativeSampler, NoValidNegativesError, build_negative_sampler

if TYPE_CHECKING:
    from torch_geometric.data import Data


def _combine_batches(batches: list[LinkBatch], requested: int) -> LinkBatch:
    """Concatenate held-out sample chunks while preserving coverage counts.

    Args:
        batches: Nonempty sequence of source-aligned evaluation batches.
        requested: Total positive links before skips across all chunks.

    Returns:
        Combined ``LinkBatch`` with concatenated tensors and aggregated stats.

    Raises:
        ValueError: If no chunk contains valid evaluation samples.
    """
    if not batches:
        raise ValueError("No held-out examples have usable structural negatives.")
    stats = SamplingStats(
        requested_positives=requested,
        retained_positives=sum(batch.stats.retained_positives for batch in batches),
        skipped_positives=sum(batch.stats.skipped_positives for batch in batches),
        requested_hard=sum(batch.stats.requested_hard for batch in batches),
        drawn_hard=sum(batch.stats.drawn_hard for batch in batches),
        drawn_uniform=sum(batch.stats.drawn_uniform for batch in batches),
        repeated_negatives=sum(batch.stats.repeated_negatives for batch in batches),
        empty_hard_pools=sum(batch.stats.empty_hard_pools for batch in batches)
    )
    return LinkBatch(
        positive_src=torch.cat([batch.positive_src for batch in batches]),
        positive_dst=torch.cat([batch.positive_dst for batch in batches]),
        negative_dst=torch.cat([batch.negative_dst for batch in batches]),
        hard_mask=torch.cat([batch.hard_mask for batch in batches]),
        stats=stats
    )


def _fixed_phase_samples(
    positive_edges: Tensor, sampler: NegativeSampler, settings: NegativeSamplingSettings, seed: int
) -> LinkBatch:
    """Draw fixed negative examples for one held-out link partition.

    Args:
        positive_edges: CPU directed positive links of shape ``[2, E]``.
        sampler: Configured sampler using training-only message edges and
            full-graph positive exclusion.
        settings: Negative amount, hard-candidate strategy, hard fraction,
            and evaluation sampling chunk size.
        seed: Private random seed for this evaluation partition.

    Returns:
        ``LinkBatch`` combining fixed examples and coverage statistics.
        Negatives are sampled once and not resampled during validation.

    Raises:
        ValueError: If no held-out positives can receive valid negatives.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batches: list[LinkBatch] = []
    skipped_hard = 0
    for start in range(0, positive_edges.size(1), settings.evaluation_chunk_size):
        stop = start + settings.evaluation_chunk_size
        pairs = positive_edges[:, start:stop]
        try:
            batches.append(sampler.sample(pairs[0], pairs[1], generator))
        except NoValidNegativesError:
            # Count the requested hard negatives for skipped examples too.
            fraction = 1.0 if settings.strategy == "khop" else settings.hard_fraction
            skipped_hard += round(pairs.size(1) * settings.amount * fraction)
            continue
    if not batches:
        raise ValueError("No held-out examples have usable structural negatives.")
    combined = _combine_batches(batches, positive_edges.size(1))
    # Chunks with no usable negatives raise an error instead of producing a
    # batch. Include those skipped positives in the final coverage totals.
    missing = positive_edges.size(1) - (
        combined.stats.retained_positives + combined.stats.skipped_positives
    )
    if missing:
        combined = replace(combined, stats=replace(
            combined.stats,
            skipped_positives=combined.stats.skipped_positives + missing,
            requested_hard=combined.stats.requested_hard + skipped_hard,
            empty_hard_pools=combined.stats.empty_hard_pools + (
                missing if settings.strategy == "khop" else 0
            )
        ))
    return combined


def build_fixed_evaluation(
    graph: Data, splits: tuple[Data, Data, Data], settings: NegativeSamplingSettings, seed: int
) -> dict[str, LinkBatch]:
    """Build fixed validation and test link batches for one training run.

    Args:
        graph: Full unsplit PyG road graph, used only for excluding known
            positive links when sampling negatives.
        splits: ``(train, val, test)`` PyG graphs; only training ``edge_index``
            is used to discover directed K-hop candidates.
        settings: Negative-sampling strategy, ratios, and evaluation chunk size.
        seed: Base seed used to generate separate fixed phase samples.

    Returns:
        Dictionary with ``validation`` and ``test`` ``LinkBatch`` values.
        Hold the returned batches unchanged across validation epochs and
        final testing. No separate evaluation files are written.
    """
    full_positives = FullPositiveIndex(graph.edge_index, graph.num_nodes)
    sampler = build_negative_sampler(
        splits[0].edge_index, full_positives, graph.num_nodes, settings
    )
    seed_generator = torch.Generator(device="cpu").manual_seed(seed)
    # Draw two independent phase seeds from a local generator so evaluation
    # is reproducible without resetting PyTorch's process-wide RNG.
    validation_seed, test_seed = torch.randint(
        low=0,
        high=torch.iinfo(torch.int64).max,
        size=(2,),
        generator=seed_generator,
        dtype=torch.int64
    ).tolist()
    return {
        "validation": _fixed_phase_samples(
            splits[1].edge_label_index, sampler, settings, validation_seed
        ),
        "test": _fixed_phase_samples(
            splits[2].edge_label_index, sampler, settings, test_seed
        )
    }
