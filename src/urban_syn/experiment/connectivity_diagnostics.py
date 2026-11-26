"""Fixed road-link diagnostics for the pretrained connectivity scorer.

Monitoring roads come from the selected trajectory training or validation split.
Graph edges supply valid structural positives; sparse consecutive observations
are never assumed to be direct graph transitions.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor, nn

from urban_syn.diffusion.data import RoadSequenceDataset
from urban_syn.link_pretrain.data import LinkBatch
from urban_syn.link_pretrain.evaluation import evaluate_link_scores
from urban_syn.link_pretrain.graph import FullPositiveIndex
from urban_syn.link_pretrain.sampling import (
    NoValidNegativesError,
    TrainingPositiveSelector,
    build_negative_sampler
)

from .config import ScorerDiagnosticsSettings
from .datasets import DatasetSplit


def sample_monitor_roads(
    dataset: RoadSequenceDataset,
    split: DatasetSplit,
    validation_enabled: bool,
    sample_size: int,
    seed: int
) -> Tensor:
    """Select fixed zero-based road indices from one trajectory partition.

    Args:
        dataset: Prepared trajectories returning one-based model road IDs.
        split: Existing group-safe training and validation row indices.
        validation_enabled: Use validation rows when true, training otherwise.
        sample_size: Maximum number of unique roads to collect.
        seed: Seed controlling the randomized sequence traversal.

    Returns:
        Sorted CPU long road indices, excluding padding.
    """
    indices = split.validation_indices if validation_enabled else split.train_indices
    if not indices:
        raise ValueError("Cannot monitor an empty trajectory partition.")
    if sample_size < 1:
        raise ValueError("Monitoring sample size must be positive.")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(len(indices), generator=generator)
    roads: set[int] = set()
    for position in order.tolist():
        model_ids, _ = dataset[indices[position]]
        for road_id in model_ids.reshape(-1).tolist():
            if road_id > 0:
                roads.add(int(road_id) - 1)
                if len(roads) == sample_size:
                    break
        if len(roads) == sample_size:
            break
    if not roads:
        raise ValueError("No nonpadding roads found in the monitoring partition.")
    return torch.tensor(sorted(roads), dtype=torch.long)


class ScorerDiagnostics:
    """Evaluate a fixed, graph-validated sample without re-encoding the GNN."""

    def __init__(
        self,
        graph_edges: Tensor,
        num_nodes: int,
        monitor_roads: Tensor,
        settings: ScorerDiagnosticsSettings,
        seed: int,
        objective: str,
        split_name: str
    ) -> None:
        """Select reproducible graph-positive and negative examples.

        Args:
            graph_edges: Full directed road graph edges in zero-based PyG indices.
            num_nodes: Total road nodes without the diffusion padding row.
            monitor_roads: Candidate seeds from the chosen trajectory split.
            settings: Validated diagnostic cadence and sampling settings.
            seed: Fixed seed for seed, positive, and negative selection.
            objective: Original pretrained scorer objective, ``bce`` or ``bpr``.
            split_name: ``train`` or ``val`` metric namespace.
        """
        if objective not in ("bce", "bpr"):
            raise ValueError("Scorer diagnostics require a BCE/BPR-pretrained scorer.")
        if split_name not in ("train", "val"):
            raise ValueError("Monitoring split must be train or val.")

        edges = graph_edges.detach().to(device="cpu", dtype=torch.long)
        selector = TrainingPositiveSelector(edges, num_nodes)
        candidates = torch.unique(monitor_roads.detach().cpu().long(), sorted=True)
        if candidates.numel() == 0 or bool(((candidates < 0) | (candidates >= num_nodes)).any()):
            raise ValueError("Monitoring roads must belong to the loaded graph.")
        eligible = torch.zeros(num_nodes, dtype=torch.bool)
        eligible[selector.eligible_seeds] = True
        candidates = candidates[eligible[candidates]]
        if candidates.numel() == 0:
            raise ValueError("Selected roads have no outgoing graph links.")

        generator = torch.Generator(device="cpu").manual_seed(seed)
        positions = torch.randperm(candidates.numel(), generator=generator)
        sources = candidates[positions[:settings.sample_size]]
        positives = selector.select(sources, generator)
        full_index = FullPositiveIndex(edges, num_nodes)
        sampler = build_negative_sampler(edges, full_index, num_nodes, settings.negatives)
        try:
            self.samples: LinkBatch = sampler.sample(sources, positives, generator)
        except NoValidNegativesError as exc:
            raise ValueError("No valid structural negatives for scorer monitoring.") from exc

        self.settings = settings
        self.objective = objective
        self.split_name = split_name
        self.seed = seed
        self.requested = sources.numel()

    @torch.no_grad()
    def observe(self, embeddings: Tensor, scorer: nn.Module) -> dict[str, float]:
        """Score fixed graph links with the current, unpadded road embeddings.

        Args:
            embeddings: Evaluation-mode GNN road table ``[N, D]``.
            scorer: Frozen directional link scorer in evaluation mode.

        Returns:
            Graph BCE/BPR loss, pairwise accuracy, Hits@1, MRR, and coverage.
        """
        metrics = evaluate_link_scores(
            embeddings, scorer, self.samples, self.objective, self.settings.chunk_size
        )
        prefix = f"{self.split_name}/scorer/"
        return {
            **{f"{prefix}{key}": value for key, value in metrics.items()},
            f"{prefix}retained_pairs": float(self.samples.stats.retained_positives),
            f"{prefix}sample_coverage": self.samples.stats.retained_positives / self.requested
        }

    def write_sample_manifest(self, path: Path, graph_road_ids: Tensor) -> Path:
        """Save the sampled graph node indices and mapped physical road IDs."""
        sampled = self.samples
        physical = graph_road_ids.detach().cpu().long()
        payload = {
            "seed": self.seed,
            "split": self.split_name,
            "objective": self.objective,
            "requested": self.requested,
            "retained": sampled.stats.retained_positives,
            "positive_src": sampled.positive_src.tolist(),
            "positive_dst": sampled.positive_dst.tolist(),
            "negative_dst": sampled.negative_dst.tolist(),
            "positive_src_edge_id": physical[sampled.positive_src].tolist(),
            "positive_dst_edge_id": physical[sampled.positive_dst].tolist(),
            "negative_dst_edge_id": physical[sampled.negative_dst].tolist()
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path
