"""Read-only gradient and fixed-road embedding diagnostics.

The probe owns deterministic road selection and one evaluation-mode graph pass.
Injected metrics compute scalar measurements from the same selected embeddings.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import Data


_METRIC_PREFIX = "train/gnn_embedding_"


def gradient_norm(module: nn.Module) -> float:
    """Return the Euclidean norm of existing parameter gradients, or zero."""
    norms = [
        parameter.grad.detach().float().norm(2)
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if not norms:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(norms)).item())


def should_log_gradient_diagnostics(step: int, every_steps: int) -> bool:
    """Select the first optimizer step and subsequent scheduled steps.

    A frequency of zero disables these diagnostics entirely.
    ``step`` is the next completed optimizer-step number (1-based).
    """
    return every_steps > 0 and (step == 1 or step % every_steps == 0)


def should_probe_epoch(epoch: int, enabled: bool, every_epochs: int) -> bool:
    """Decide whether a completed epoch needs a full-graph probe."""
    return enabled and epoch > 0 and every_epochs > 0 and epoch % every_epochs == 0


class EmbeddingMetric(Protocol):
    """Stateless strategy for computing metrics from fixed road embeddings."""

    def compute(self, current: Tensor, previous: Tensor | None) -> dict[str, float]:
        """Return scalar metrics; ``previous`` is the preceding observation."""
        ...


class NormStatistics:
    """Mean and population standard deviation of per-road embedding norms."""

    def compute(self, current: Tensor, previous: Tensor | None) -> dict[str, float]:
        """Measure representation magnitudes at the current observation."""
        norms = torch.linalg.vector_norm(current, dim=-1)
        return {
            f"{_METRIC_PREFIX}norm_mean": float(norms.mean().item()),
            f"{_METRIC_PREFIX}norm_std": float(norms.std(unbiased=False).item()),
        }


class CosineDrift:
    """Mean directional change in each road embedding since last observation."""

    def compute(self, current: Tensor, previous: Tensor | None) -> dict[str, float]:
        """Return no metric for the first observation without a reference."""
        if previous is None:
            return {}
        current_norms = torch.linalg.vector_norm(current, dim=-1)
        previous_norms = torch.linalg.vector_norm(previous, dim=-1)
        cosine = F.cosine_similarity(current, previous, dim=-1, eps=1e-8)
        both_zero = (current_norms <= 1e-8) & (previous_norms <= 1e-8)
        cosine = torch.where(both_zero, torch.ones_like(cosine), cosine)
        return {
            f"{_METRIC_PREFIX}cosine_drift": float((1 - cosine).clamp_min(0).mean().item())
        }


class L2Drift:
    """Mean Euclidean movement per road since the previous observation."""

    def compute(self, current: Tensor, previous: Tensor | None) -> dict[str, float]:
        """Return no drift until an earlier observation exists."""
        if previous is None:
            return {}
        distances = torch.linalg.vector_norm(current - previous, dim=-1)
        return {f"{_METRIC_PREFIX}l2_drift": float(distances.mean().item())}


class PairwiseSimilarity:
    """Cosine similarity between distinct, sampled non-padding road embeddings.

    Only the upper triangle is used, excluding self-similarity. The high
    similarity fraction counts pairs whose cosine similarity exceeds 0.99.
    """

    def __init__(self, high_threshold: float = 0.99):
        """Set the high-similarity threshold for the pairwise diagnostic."""
        if not -1.0 <= high_threshold <= 1.0:
            raise ValueError("high_threshold must lie between -1 and 1.")
        self.high_threshold = high_threshold

    def compute(self, current: Tensor, previous: Tensor | None) -> dict[str, float]:
        """Measure similarities across different sampled road segments."""
        if current.shape[0] < 2:
            return {}
        # Normalization removes the effect of absolute embedding scale.
        normalized = F.normalize(current, dim=-1, eps=1e-8)
        similarity = normalized @ normalized.T
        row, column = torch.triu_indices(current.shape[0], current.shape[0], offset=1)
        pairs = similarity[row, column]
        return {
            "train/gnn_pairwise_cosine_mean": float(pairs.mean().item()),
            "train/gnn_pairwise_cosine_high_fraction": float(
                (pairs > self.high_threshold).float().mean().item()
            ),
        }


def default_embedding_metrics() -> tuple[EmbeddingMetric, ...]:
    """Provide a default suite while allowing different strategies to be injected."""
    return (NormStatistics(), CosineDrift(), L2Drift(), PairwiseSimilarity())


class EmbeddingProbe:
    """Compute injected diagnostics using a deterministic subset of road IDs.

    A single full-graph GNN pass supplies all metric strategies. The result is
    retained on CPU, never as a gradient-bearing tensor. The model's prior
    training/evaluation state is restored, including when a metric fails.
    """

    def __init__(
        self,
        sample_size: int,
        seed: int,
        metrics: Sequence[EmbeddingMetric] | None = None,
    ):
        """Configure the fixed subset, seed, and optional metric strategies."""
        if sample_size < 1:
            raise ValueError("sample_size must be positive.")
        self.sample_size = sample_size
        self.seed = seed
        self.metrics = tuple(metrics) if metrics is not None else default_embedding_metrics()
        self._indices: Tensor | None = None
        self._previous: Tensor | None = None
        self._num_roads: int | None = None

    def _select_roads(self, embeddings: Tensor) -> Tensor:
        """Choose road rows once, reserving row zero for padding."""
        if embeddings.ndim != 2 or embeddings.shape[0] < 2:
            raise ValueError("Expected road embeddings [N + 1, D], N >= 1.")
        num_roads = embeddings.shape[0] - 1
        if self._indices is None:
            generator = torch.Generator(device="cpu").manual_seed(self.seed)
            self._indices = torch.randperm(num_roads, generator=generator)[:self.sample_size] + 1
            self._num_roads = num_roads
        elif num_roads != self._num_roads:
            raise ValueError("The road vocabulary size changed between observations.")

        selected = embeddings.index_select(0, self._indices.to(embeddings.device))
        return selected.detach().float().cpu().clone()

    @torch.no_grad()
    def observe(self, model: nn.Module, graph: Data) -> dict[str, float]:
        """Take one eval-mode snapshot and run all injected metric strategies."""
        encoder = model.road_encoder
        was_training = encoder.training
        encoder.eval()
        try:
            # Message passing needs the entire graph before subset extraction.
            current = self._select_roads(model.encode_roads(graph))
            result: dict[str, float] = {}
            for metric in self.metrics:
                values = metric.compute(current, self._previous)
                overlap = result.keys() & values.keys()
                if overlap:
                    raise ValueError(f"Embedding diagnostics returned duplicate keys: {sorted(overlap)}")
                result.update(values)
            self._previous = current
            return result
        finally:
            encoder.train(was_training)
