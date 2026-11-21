"""Read-only gradient and fixed-road embedding diagnostics.

A fixed road sample is shared across training approaches. Injected embedding
sources provide unpadded embeddings through model-specific interfaces.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import Data


_METRIC_PREFIX = "train/gnn_embedding_"
_VICREG_VARIANCE_EPSILON = 1e-4
_VICREG_STD_THRESHOLD = 1.0


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


def should_record_embeddings(epoch: int, enabled: bool, every_epochs: int) -> bool:
    """Return whether a completed epoch needs an embedding snapshot."""
    return enabled and epoch > 0 and every_epochs > 0 and epoch % every_epochs == 0


def should_probe_epoch(epoch: int, enabled: bool, every_epochs: int) -> bool:
    """Preserve the scheduling entry point used by the existing joint trainer."""
    return should_record_embeddings(epoch, enabled, every_epochs)


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


class FeatureVariance:
    """Mean variance of each embedding coordinate across sampled roads."""

    def compute(self, current: Tensor, previous: Tensor | None) -> dict[str, float]:
        """Return average population variance; near-zero values can signal collapse."""
        variance = current.var(dim=0, unbiased=False)
        return {"train/gnn_embedding_feature_variance_mean": float(variance.mean().item())}


def _check_embedding_matrix(embeddings: Tensor) -> None:
    """Reject malformed inputs instead of emitting invalid MLflow metrics."""
    if embeddings.ndim != 2 or min(embeddings.shape) < 1:
        raise ValueError("Expected a nonempty embedding matrix [roads, dimensions].")
    if not bool(torch.isfinite(embeddings).all()):
        raise ValueError("Embedding diagnostics require finite values.")


class EffectiveRank:
    """Uncentered RankMe effective rank from singular-value entropy.

    Reference: Garrido et al., ICML 2023, https://proceedings.mlr.press/v202/garrido23a.html.
    Uses the raw embedding matrix, *without* covariance centering. The paper's
    normalized singular-value entropy is evaluated with an exact zero guard
    instead of adding a perturbation to nonzero probabilities. An all-zero
    matrix has no singular-value distribution and is reported as rank zero.
    """

    def compute(self, current: Tensor, previous: Tensor | None) -> dict[str, float]:
        """Return exp(entropy(singular_values / sum(singular_values)))."""
        _check_embedding_matrix(current)
        # SVD of only the sampled roads; float64 avoids precision loss for tiny spectra.
        singular_values = torch.linalg.svdvals(current.to(dtype=torch.float64))
        total = singular_values.sum()
        if total == 0:
            rank = 0.0
        else:
            probabilities = singular_values / total
            positive = probabilities[probabilities > 0]
            rank = float(torch.exp(-(positive * positive.log()).sum()).item())
        return {f"{_METRIC_PREFIX}rankme_effective_rank": rank}


class VICRegVariance:
    """Read-only per-coordinate standard deviations from VICReg.

    Reference: Bardes et al., ICLR 2022, official facebookresearch/vicreg
    main_vicreg.py VICReg.forward(): sqrt(z.var(dim=0) + 1e-4).
    The default torch.var uses *sample* (N-1) variance. This reports the
    std statistics and VICReg's fraction below one, NOT its training penalty.
    """

    def compute(self, current: Tensor, previous: Tensor | None) -> dict[str, float]:
        """Summarize VICReg standard deviations, with no gradient or loss."""
        _check_embedding_matrix(current)
        if current.shape[0] < 2:
            # Sample variance is undefined for a single road; do not log NaNs.
            return {}
        std = torch.sqrt(
            current.to(dtype=torch.float64).var(dim=0, correction=1)
            + _VICREG_VARIANCE_EPSILON
        )
        return {
            f"{_METRIC_PREFIX}vicreg_std_mean": float(std.mean().item()),
            f"{_METRIC_PREFIX}vicreg_std_min": float(std.min().item()),
            f"{_METRIC_PREFIX}vicreg_low_variance_fraction": float((std < _VICREG_STD_THRESHOLD).double().mean().item()),
        }


class RoadEmbeddingSource(Protocol):
    """Adapt different models to an ordered, padding-free road embedding table.

    Returned row ``i`` must represent the same zero-based PyG road index in
    all training pipelines. Each source chooses its own message graph.
    """

    def module(self, model: nn.Module) -> nn.Module:
        """Return the road encoder whose training mode must be restored."""
        ...

    def encode(self, model: nn.Module, graph: Data) -> Tensor:
        """Return ``[N, D]`` embeddings ordered by zero-based road index."""
        ...


class JointRoadEmbeddingSource:
    """Read unpadded road embeddings from the joint diffusion model."""

    def module(self, model: nn.Module) -> nn.Module:
        """Return the joint model's GNN module."""
        return model.road_encoder

    def encode(self, model: nn.Module, graph: Data) -> Tensor:
        """Return road embeddings in the graph's PyG node order."""
        return model.encode_road_nodes(graph)


def default_embedding_metrics() -> tuple[EmbeddingMetric, ...]:
    """Provide a default suite while allowing different strategies to be injected."""
    return (
        NormStatistics(), CosineDrift(), L2Drift(), PairwiseSimilarity(),
        FeatureVariance(), EffectiveRank(), VICRegVariance()
    )


class EmbeddingDiagnostics:
    """Compute metrics on the same seeded road indices across model variants.

    Each observation encodes the graph supplied by the caller. The source
    removes any padding row and returns roads in zero-based index order.
    Embeddings are detached to CPU, and encoder training mode is restored
    even when a metric fails.
    """

    def __init__(
        self,
        sample_size: int,
        seed: int,
        metrics: Sequence[EmbeddingMetric] | None = None,
        source: RoadEmbeddingSource | None = None
    ):
        """Configure a reproducible road sample and its metric strategies.

        Args:
            sample_size: Maximum number of physical roads to compare.
            seed: Seed used for choosing fixed zero-based road indices.
            metrics: Optional metric strategies; defaults to norm, drift,
                similarity, feature variance, and collapse measurements.
            source: Model adapter producing padding-free ``[N, D]`` embeddings.
                Defaults to the joint model's encode_road_nodes interface.
        """
        if sample_size < 1:
            raise ValueError("sample_size must be positive.")
        self.sample_size = sample_size
        self.seed = seed
        self.metrics = tuple(metrics) if metrics is not None else default_embedding_metrics()
        self.source = source if source is not None else JointRoadEmbeddingSource()
        self._indices: Tensor | None = None
        self._previous: Tensor | None = None
        self._num_roads: int | None = None
        self._sampled_road_ids: Tensor | None = None

    def _select_roads(self, embeddings: Tensor) -> Tensor:
        """Select physical road rows once from an unpadded ``[N, D]`` tensor."""
        if embeddings.ndim != 2 or embeddings.shape[0] < 1:
            raise ValueError("Expected padding-free road embeddings [N, D], N >= 1.")
        num_roads = embeddings.shape[0]
        if self._indices is None:
            generator = torch.Generator(device="cpu").manual_seed(self.seed)
            self._indices = torch.randperm(num_roads, generator=generator)[:self.sample_size]
            self._num_roads = num_roads
        elif num_roads != self._num_roads:
            raise ValueError("The road vocabulary size changed between observations.")

        selected = embeddings.index_select(0, self._indices.to(embeddings.device))
        return selected.detach().float().cpu().clone()

    def _verify_road_ids(self, graph: Data) -> None:
        """Ensure repeated observations still refer to the same physical roads."""
        road_ids = getattr(graph, "edge_id", None)
        if road_ids is None:
            return  # Older synthetic test graphs may not contain physical IDs.
        if self._indices is None or self._num_roads is None:
            raise RuntimeError("Observe embeddings before verifying sampled roads.")
        if road_ids.ndim != 1 or road_ids.numel() != self._num_roads:
            raise ValueError("Graph edge_id must align with the physical road rows.")
        selected = road_ids.index_select(0, self._indices.to(road_ids.device)).long().cpu()
        if self._sampled_road_ids is None:
            self._sampled_road_ids = selected
        elif not torch.equal(selected, self._sampled_road_ids):
            raise ValueError("Physical road IDs changed between embedding observations.")

    def write_sample_manifest(self, path: Path, graph: Data) -> Path:
        """Persist fixed road indices and IDs so independent runs can be compared.

        Must be called after the first observation. A missing physical ID column
        is an error rather than silently claiming the samples are comparable.
        """
        if self._indices is None:
            raise RuntimeError("Observe embeddings before exporting the sample.")
        self._verify_road_ids(graph)
        if self._sampled_road_ids is None:
            raise ValueError("Road graph has no physical edge_id for sample verification.")
        manifest: dict[str, Any] = {
            "seed": self.seed,
            "requested_sample_size": self.sample_size,
            "num_roads": self._num_roads,
            "node_indices": self._indices.tolist(),
            "road_edge_ids": self._sampled_road_ids.tolist(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
        return path

    @torch.no_grad()
    def observe(self, model: nn.Module, graph: Data) -> dict[str, float]:
        """Compute current embedding diagnostics from one eval-mode GNN pass.

        Args:
            model: Model understood by the injected road embedding source.
            graph: Device-ready message-passing graph for this experiment.

        Returns:
            Named scalar embedding metrics, including drift when there is a
            previous snapshot. The snapshot is held on CPU without gradients.
        """
        encoder = self.source.module(model)
        was_training = encoder.training
        encoder.eval()
        try:
            # Message passing needs the entire graph before subset extraction.
            current = self._select_roads(self.source.encode(model, graph))
            self._verify_road_ids(graph)
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


# Preserve existing imports in the joint training loop.
EmbeddingProbe = EmbeddingDiagnostics
