"""Interchangeable smoothing strategies for physical road-transition counts."""

import math
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor


class TransitionSmoother(Protocol):
    """Define how counts become probabilities on directed physical edges."""

    def smooth(
        self,
        counts: Tensor,
        edge_index: Tensor,
        num_nodes: int,
    ) -> Tensor:
        """Compute probabilities aligned with physical edges.

        Parameters
        ----------
        counts : Tensor
            Nonnegative transition counts of shape [E].
        edge_index : Tensor
            Physical graph adjacency of shape [2, E].
        num_nodes : int
            Number of road-segment nodes.

        Returns
        -------
        Tensor
            Transition probabilities of shape [E].
        """
        ...


@dataclass(frozen=True)
class AdditiveSmoothing:
    """Apply symmetric Dirichlet (add-alpha) smoothing over outgoing roads."""

    alpha: float = 1.0

    def __post_init__(self) -> None:
        """Validate the additive pseudocount.

        Raises
        ------
        ValueError
            If alpha is not positive and finite.
        """
        if not isinstance(self.alpha, (int, float)) or not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be positive and finite.")

    def smooth(
        self,
        counts: Tensor,
        edge_index: Tensor,
        num_nodes: int,
    ) -> Tensor:
        """Normalize observed counts over physically available outgoing edges.

        Parameters
        ----------
        counts : Tensor
            Nonnegative physical-edge counts of shape [E].
        edge_index : Tensor
            Directed physical adjacency of shape [2, E].
        num_nodes : int
            Total number of nodes, including isolated roads.

        Returns
        -------
        Tensor
            Add-alpha conditional probabilities of shape [E].

        Raises
        ------
        ValueError
            If the input tensors do not match or contain invalid counts.
        """
        if (
            counts.ndim != 1
            or edge_index.ndim != 2
            or edge_index.size(0) != 2
            or counts.numel() != edge_index.size(1)
        ):
            raise ValueError("counts must be [E] and edge_index must be [2, E].")
        
        if not torch.isfinite(counts).all() or (counts < 0).any():
            raise ValueError("Transition counts must be nonnegative and finite.")

        # Each edge is one possible next road for its source; no new edges are added.
        source = edge_index[0].to(device=counts.device)
        totals = counts.new_zeros(num_nodes)
        totals.index_add_(0, source, counts)
        
        out_degree = torch.bincount(source, minlength=num_nodes).to(counts.dtype)
        denominator = totals[source] + self.alpha * out_degree[source]
        
        return (counts + self.alpha) / denominator
