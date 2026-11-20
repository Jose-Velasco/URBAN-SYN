"""Small tensor contracts for source-aligned structural link samples.

PyTorch DataLoader is responsible for batching and multiprocessing. This
module defines aligned sample tensors without a competing batching framework.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor


@dataclass(frozen=True)
class SamplingStats:
    """Counts describing negative-pool coverage and sampling outcomes."""

    requested_positives: int = 0
    retained_positives: int = 0
    skipped_positives: int = 0
    requested_hard: int = 0
    drawn_hard: int = 0
    drawn_uniform: int = 0
    repeated_negatives: int = 0
    empty_hard_pools: int = 0

    def __add__(self, other: SamplingStats) -> SamplingStats:
        """Combine the count totals from two sampled link batches.

        Args:
            other: Counts from another batch or collection of batches.

        Returns:
            New immutable statistics with all eight counters summed.
        """
        if not isinstance(other, SamplingStats):
            return NotImplemented
        return SamplingStats(
            requested_positives=self.requested_positives + other.requested_positives,
            retained_positives=self.retained_positives + other.retained_positives,
            skipped_positives=self.skipped_positives + other.skipped_positives,
            requested_hard=self.requested_hard + other.requested_hard,
            drawn_hard=self.drawn_hard + other.drawn_hard,
            drawn_uniform=self.drawn_uniform + other.drawn_uniform,
            repeated_negatives=self.repeated_negatives + other.repeated_negatives,
            empty_hard_pools=self.empty_hard_pools + other.empty_hard_pools
        )

    @property
    def achieved_hard_fraction(self) -> float:
        """Fraction of drawn negatives obtained from local K-hop candidates."""
        total = self.drawn_hard + self.drawn_uniform
        return self.drawn_hard / total if total else 0.0


@dataclass(frozen=True)
class LinkBatch:
    """A positive-link batch with its aligned negative road destinations.

    Attributes:
        positive_src: CPU long tensor of source roads, shape ``[P]``.
        positive_dst: CPU long tensor of positive destinations, shape ``[P]``.
        negative_dst: CPU long tensor of negatives, shape ``[P, amount]``.
        hard_mask: Boolean tensor of the same shape as ``negative_dst``.
            True means a K-hop candidate, not a supervision label.
        stats: Sampling coverage and fallback counts for this batch.

    All road indices are zero-based PyG node indices. Row ``i`` of each
    negative tensor belongs to positive pair ``i``.
    """

    positive_src: Tensor
    positive_dst: Tensor
    negative_dst: Tensor
    hard_mask: Tensor
    stats: SamplingStats

    def __post_init__(self) -> None:
        """Check the inexpensive shape/dtype invariants at construction."""
        count = self.positive_src.numel()
        if self.positive_src.ndim != 1 or self.positive_dst.shape != (count,):
            raise ValueError("Positive links must be aligned one-dimensional tensors.")
        if self.negative_dst.ndim != 2 or self.negative_dst.shape[0] != count:
            raise ValueError("Negative destinations must have shape [positives, amount].")
        if self.negative_dst.shape != self.hard_mask.shape:
            raise ValueError("hard_mask must align with negative destinations.")
        if any(value.dtype != torch.long for value in
               (self.positive_src, self.positive_dst, self.negative_dst)):
            raise ValueError("Link IDs must use torch.long (zero-based PyG indices).")
        if self.hard_mask.dtype != torch.bool:
            raise ValueError("hard_mask must be a boolean tensor.")
        if any(value.device.type != "cpu" for value in
               (self.positive_src, self.positive_dst, self.negative_dst, self.hard_mask)):
            raise ValueError("LinkBatch sampling tensors must remain on CPU.")
        if self.stats.retained_positives != count:
            raise ValueError("Sampling stats disagree with retained positive rows.")

    def pin_memory(self) -> LinkBatch:
        """Pin sample tensors for nonblocking CPU-to-GPU transfer.

        Returns:
            New ``LinkBatch`` with the same zero-based road indices and
            sampling statistics, stored in page-locked CPU memory. PyTorch
            calls this method when DataLoader ``pin_memory=True`` and CUDA
            pinning is available. It does not transfer tensors to a GPU.
        """
        return replace(
            self,
            positive_src=self.positive_src.pin_memory(),
            positive_dst=self.positive_dst.pin_memory(),
            negative_dst=self.negative_dst.pin_memory(),
            hard_mask=self.hard_mask.pin_memory()
        )
