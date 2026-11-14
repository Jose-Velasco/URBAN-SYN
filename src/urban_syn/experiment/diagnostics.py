"""GNN gradient and deterministic road-embedding diagnostics.

Gradient norms are measured from existing backpropagation results. Embedding
measurements use a fixed road subset from a full-graph encoder pass in eval
mode, so training dropout does not inflate observed representation drift.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import Data


def gradient_norm(module: nn.Module) -> float:
    """Measure the Euclidean gradient norm of one model branch.

    Parameters
    ----------
    module : nn.Module
        Trainable submodule after backward() and before gradient clipping.

    Returns
    -------
    float
        Norm of the available parameter gradients, or zero if none exist.
    """
    norms = [
        parameter.grad.detach().float().norm(2)
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    if not norms:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(norms)).item())


class EmbeddingProbe:
    """Compare fixed road representations at successive model snapshots.

    The same non-padding road indices are sampled deterministically. Every
    observation computes embeddings from the full graph with the road encoder
    in eval mode, then restores its previous train/eval state. A baseline
    observation can be taken immediately before training or after a resume.
    """

    def __init__(self, sample_size: int, seed: int):
        """Initialize the fixed road selection and comparison state.

        Parameters
        ----------
        sample_size : int
            Maximum number of non-padding road embeddings to compare.
        seed : int
            Seed for road selection, independent of the training random state.
        """
        if sample_size < 1:
            raise ValueError("sample_size must be positive.")
        self.sample_size = sample_size
        self.seed = seed
        self._indices: Tensor | None = None
        self._previous: Tensor | None = None
        self._num_roads: int | None = None

    @torch.no_grad()
    def observe(self, model: nn.Module, graph: Data) -> dict[str, float]:
        """Measure norm statistics and drift using evaluation-mode embeddings.

        Parameters
        ----------
        model : nn.Module
            Joint model exposing road_encoder and encode_roads(graph).
        graph : Data
            Complete road graph on the same device as the model.

        Returns
        -------
        dict[str, float]
            Fixed-probe mean/standard deviation of embedding norms and mean
            cosine drift since the prior snapshot (if one exists).
        """
        encoder = model.road_encoder
        was_training = encoder.training
        encoder.eval()
        try:
            # Full-graph message passing is required even though only a small
            # subset of the resulting road embeddings is retained.
            embeddings = model.encode_roads(graph)
            if embeddings.ndim != 2 or embeddings.shape[0] < 2:
                raise ValueError("Expected road embeddings [N + 1, D], N >= 1.")

            num_roads = embeddings.shape[0] - 1
            if self._indices is None:
                generator = torch.Generator(device="cpu").manual_seed(self.seed)
                self._indices = (
                    torch.randperm(num_roads, generator=generator)[:self.sample_size] + 1
                )
                self._num_roads = num_roads
            elif num_roads != self._num_roads:
                raise ValueError("The road vocabulary size changed between observations.")

            # Copy only the selected rows to CPU; never retain the full-graph
            # output or its computation graph between measurements.
            probe = embeddings.index_select(0, self._indices.to(embeddings.device))
            probe = probe.detach().float().cpu().clone()
            norms = torch.linalg.vector_norm(probe, dim=-1)
            result = {
                "train/gnn_embedding_norm_mean": float(norms.mean().item()),
                "train/gnn_embedding_norm_std": float(norms.std(unbiased=False).item())
            }

            if self._previous is not None:
                previous_norms = torch.linalg.vector_norm(self._previous, dim=-1)
                cosine = F.cosine_similarity(probe, self._previous, dim=-1, eps=1e-8)
                both_zero = (norms <= 1e-8) & (previous_norms <= 1e-8)
                cosine = torch.where(both_zero, torch.ones_like(cosine), cosine)
                drift = (1 - cosine).clamp_min(0)
                result["train/gnn_embedding_cosine_drift"] = float(drift.mean().item())

            self._previous = probe
            return result
        finally:
            encoder.train(was_training)
