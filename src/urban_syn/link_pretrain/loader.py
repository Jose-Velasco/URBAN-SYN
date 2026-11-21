"""CPU link sampling with standard PyTorch DataLoader workers.

The training graph is used only for positive supervision and K-hop lookup.
The full road graph is used only to reject false negative destinations.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, get_worker_info

from .config import LinkPretrainSettings, NegativeSamplingSettings
from .data import LinkBatch, SamplingStats
from .graph import FullPositiveIndex
from .sampling import (
    NegativeSampler,
    NoValidNegativesError,
    TrainingPositiveSelector,
    build_negative_sampler
)

if TYPE_CHECKING:
    from torch_geometric.data import Data


class SeedRoadDataset(Dataset[int]):
    """Iterate over eligible road indices without duplicating positive edges."""

    def __init__(self, seeds: Tensor) -> None:
        """Store the sources that have outgoing supervised training links.

        Args:
            seeds: Unique, one-dimensional CPU long tensor of zero-based
                road indices. One dataset item represents one source road.

        Raises:
            ValueError: If the seed list is empty, duplicated, or not a
                one-dimensional CPU long tensor.
        """
        if seeds.dtype != torch.long or seeds.ndim != 1 or seeds.device.type != "cpu":
            raise ValueError("Eligible road indices must be a one-dimensional CPU long tensor.")
        if seeds.numel() == 0 or torch.unique(seeds).numel() != seeds.numel():
            raise ValueError("Eligible road indices must be nonempty and unique.")
        self.seeds = seeds

    def __len__(self) -> int:
        """Return the number of eligible source roads."""
        return self.seeds.numel()

    def __getitem__(self, index: int) -> int:
        """Return the zero-based road index at the requested dataset position."""
        return int(self.seeds[index])


class LinkBatchCollator:
    """Sample one positive and aligned negatives for each DataLoader batch.

    Sampling runs inside each worker when ``num_workers > 0``. Each worker
    owns a local PyTorch random generator; no process-global RNG is changed.
    """

    def __init__(
        self,
        selector: TrainingPositiveSelector,
        sampler: NegativeSampler,
        settings: NegativeSamplingSettings,
        seed: int
    ) -> None:
        """Configure sampling using CPU-only, read-only graph references.

        Args:
            selector: Selects a training-label positive for each source road.
            sampler: Draws validated negative destinations for those links.
            settings: Negative amount, strategy, and hard-negative fraction.
            seed: Experiment seed for sampling when no workers are used.
        """
        self.selector = selector
        self.sampler = sampler
        self.settings = settings
        self.seed = seed
        self._generator: torch.Generator | None = None
        self._generator_seed: int | None = None

    def _get_generator(self) -> torch.Generator:
        """Create or reuse a worker-local generator for random link choices."""
        worker = get_worker_info()
        if worker is None:
            worker_seed = self.seed
        else:
            # DataLoader assigns a different, reproducible seed to each
            # worker. Persistent workers retain this generator across epochs.
            worker_seed = worker.seed

        if self._generator is None or self._generator_seed != worker_seed:
            self._generator = torch.Generator(device="cpu").manual_seed(worker_seed)
            self._generator_seed = worker_seed
        return self._generator

    def _empty_batch(self, requested: int) -> LinkBatch:
        """Represent a batch with no usable negatives without failing an epoch.

        Args:
            requested: Number of source roads whose link samples were tried.

        Returns:
            Empty ``LinkBatch`` with shape ``[0, amount]`` for negatives
            and skip counters covering every requested source.
        """
        amount = self.settings.amount
        if self.settings.strategy == "khop":
            requested_hard = requested * amount
            empty_hard_pools = requested
        else:
            requested_hard = round(requested * amount * self.settings.hard_fraction)
            if requested_hard > 0:
                # A mixed batch with no valid negatives also has no usable
                # hard candidates, since all hard candidates are valid nonedges.
                empty_hard_pools = requested
            else:
                empty_hard_pools = 0

        return LinkBatch(
            positive_src=torch.empty(0, dtype=torch.long),
            positive_dst=torch.empty(0, dtype=torch.long),
            negative_dst=torch.empty((0, amount), dtype=torch.long),
            hard_mask=torch.empty((0, amount), dtype=torch.bool),
            stats=SamplingStats(
                requested_positives=requested,
                retained_positives=0,
                skipped_positives=requested,
                requested_hard=requested_hard,
                drawn_hard=0,
                drawn_uniform=0,
                repeated_negatives=0,
                empty_hard_pools=empty_hard_pools
            )
        )

    def __call__(self, seed_roads: Sequence[int]) -> LinkBatch:
        """Create a source-aligned sampled batch from DataLoader dataset items.

        Args:
            seed_roads: Selected zero-based training road indices, one per
                dataset item. Sources are sampled without replacement during
                an epoch, but positive choices vary from epoch to epoch.

        Returns:
            CPU ``LinkBatch`` with ``P <= len(seed_roads)`` positives and
            ``[P, amount]`` negatives. A fully skipped batch has ``P=0``
            and must be ignored by the training loss, but its stats remain.

        Raises:
            ValueError: If a supplied road is not eligible for supervised
                training, or if sampling produces invalid directed links.
        """
        sources = torch.tensor(seed_roads, dtype=torch.long)
        generator = self._get_generator()
        destinations = self.selector.select(sources, generator)

        try:
            return self.sampler.sample(sources, destinations, generator)
        except NoValidNegativesError:
            # Some roads have no legal negatives. A completely skipped batch
            # is valid coverage information, not a worker-process failure.
            return self._empty_batch(requested=len(seed_roads))


def make_training_dataloader(
    full_graph: Data,
    train_graph: Data,
    settings: LinkPretrainSettings
) -> DataLoader[LinkBatch]:
    """Create a shuffled DataLoader that samples links on CPU workers.

    Args:
        full_graph: Unsplit CPU road graph. Its edges are used exclusively
            to reject known direct links from negative samples.
        train_graph: CPU PyG training split. ``edge_label_index`` supplies
            supervised positives, while ``edge_index`` supplies the only
            topology for directed K-hop candidate lookup.
        settings: Validated experiment settings covering random seed,
            batch size, negative sampling, and DataLoader worker options.

    Returns:
        PyTorch ``DataLoader`` yielding ``LinkBatch`` objects. Dataset items
        are unique eligible source roads, shuffled at every epoch. Negative
        sampling and positive selection happen during worker collation.
        Use the same loader for successive epochs to retain worker RNG state.

    Raises:
        ValueError: If either graph uses incompatible node indices, a CUDA
            edge tensor, or has no eligible training-supervision sources.
    """
    if full_graph.num_nodes != train_graph.num_nodes:
        raise ValueError("Training and full road graphs must have the same node count.")

    edges = full_graph.edge_index
    message_edges = train_graph.edge_index
    training_labels = train_graph.edge_label_index
    if any(edge.device.type != "cpu" for edge in (edges, message_edges, training_labels)):
        raise ValueError("Link sampling requires CPU edge tensors.")

    full_positives = FullPositiveIndex(edges, full_graph.num_nodes)
    selector = TrainingPositiveSelector(training_labels, full_graph.num_nodes)
    sampler = build_negative_sampler(
        message_edges,
        full_positives,
        full_graph.num_nodes,
        settings.pretraining.negatives
    )
    dataset = SeedRoadDataset(selector.eligible_seeds)
    collator = LinkBatchCollator(
        selector,
        sampler,
        settings.pretraining.negatives,
        settings.experiment.seed
    )

    # DataLoader uses this generator for shuffling and deriving worker seeds.
    # Keep it local so construction never modifies torch's global RNG.
    shuffle_generator = torch.Generator(device="cpu")
    shuffle_generator.manual_seed(settings.experiment.seed)
    workers = settings.runtime.num_workers

    # PyTorch rejects prefetch/persistent options when num_workers=0.
    # Disabling them here keeps single-process debugging straightforward.
    loader_options = {
        "dataset": dataset,
        "batch_size": settings.training.batch_size,
        "shuffle": True,
        "drop_last": False,
        "collate_fn": collator,
        "num_workers": workers,
        "pin_memory": settings.runtime.pin_memory,
        "generator": shuffle_generator
    }
    if workers > 0:
        loader_options["prefetch_factor"] = settings.runtime.prefetch_factor
        loader_options["persistent_workers"] = settings.runtime.persistent_workers

    return DataLoader(**loader_options)
