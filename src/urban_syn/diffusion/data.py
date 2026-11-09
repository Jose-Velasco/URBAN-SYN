"""Load prepared FMM road sequences for joint GNN + diffusion training.

The Parquet dataset keeps shared road ``edge_id`` values. This module maps
those IDs into the current PyG graph's model-ID space when loading data.
Model road ID 0 remains reserved for padding, but batches here are grouped
by sequence length and therefore require no padding.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from collections import defaultdict
from math import ceil
from pathlib import Path
import torch
from torch import Tensor
import pandas as pd
from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
from .road_ids import edge_ids_to_model_ids

logger = logging.getLogger(__name__)


def _read_sequences(
    path: Path,
    min_points: int,
) -> tuple[list[int], list[Tensor], list[Tensor], int]:
    """Read valid prepared sequences and report the count excluded by length."""
    frame = pd.read_parquet(
        path,
        columns=["fmm_id", "edge_ids", "rates", "num_points"],
    )
    fmm_ids: list[int] = []
    # a list of list of road_ids
    edge_sequences: list[Tensor] = []
    # a list of list of rates for each corresponding road_id 
    rate_sequences: list[Tensor] = []
    filtered_count = 0

    for fmm_id, edge_ids, rates, num_points in frame.itertuples(index=False, name=None):
        length = int(num_points)
        if length < min_points:
            filtered_count += 1
            continue

        # edges = torch.as_tensor(edge_ids, dtype=torch.long)
        # positions = torch.as_tensor(rates, dtype=torch.float32)

        # Parquet may return read-only NumPy arrays. Create writable copies before
        # converting to tensors to avoid sharing potentially immutable memory.
        edges = torch.from_numpy(np.array(edge_ids, dtype=np.int64, copy=True))
        positions = torch.from_numpy(np.array(rates, dtype=np.float32, copy=True))

        _validate_sequence(fmm_id, edges, positions, length)

        fmm_ids.append(int(fmm_id))
        edge_sequences.append(edges)
        rate_sequences.append(positions)

    return fmm_ids, edge_sequences, rate_sequences, filtered_count


def _validate_sequence(
    fmm_id: int,
    edge_ids: Tensor,
    rates: Tensor,
    expected_length: int,
) -> None:
    """Catch malformed sequences without cluttering the loading loop."""
    if (
        edge_ids.ndim != 1
        or rates.ndim != 1
        or len(edge_ids) != expected_length
        or len(rates) != expected_length
        or expected_length < 1
    ):
        raise ValueError(f"FMM trajectory {fmm_id} has inconsistent sequence lengths.")
    if (edge_ids < 0).any():
        raise ValueError(f"FMM trajectory {fmm_id} has a negative road edge_id.")
    if not torch.isfinite(rates).all() or ((rates < 0) | (rates > 1)).any():
        raise ValueError(f"FMM trajectory {fmm_id} has rates outside [0, 1].")


class RoadSequenceDataset(Dataset[tuple[Tensor, Tensor]]):
    """In-memory prepared trajectories mapped into one PyG graph's ID order.

    Args:
        parquet_path: Train or test prepared road-sequence Parquet file.
        graph_edge_ids: ``graph.edge_id`` in PyG node order.
        min_points: Minimum number of matched observations to retain.
            Defaults to 1, matching preprocessing without silent filtering.

    Each item is ``(model_road_ids, rates)``, both shaped ``[T]``. Rates
    remain in [0, 1]; the existing training loop normalizes them to [-1, 1].
    """

    def __init__(
        self,
        parquet_path: str | Path,
        graph_edge_ids: Tensor,
        min_points: int = 1
        ):
        if min_points < 1:
            raise ValueError("min_points must be at least 1.")

        path = Path(parquet_path)
        self.fmm_ids, raw_ids, self._rates, self.filtered_count = _read_sequences(
            path, min_points
        )
        if not raw_ids:
            raise ValueError(f"No trajectories remain after loading {path}.")

        # Convert the flattened IDs only once. Calling the shared mapping helper
        # inside __getitem__ would repeatedly sort the entire road vocabulary.
        graph_ids = torch.as_tensor(graph_edge_ids, dtype=torch.long).detach().cpu()

        # Combine all trajectories' road IDs into one tensor so they can
        # be converted to model IDs in a single operation.
        flattened_ids = torch.cat(raw_ids)
        try:
            model_ids = edge_ids_to_model_ids(flattened_ids, graph_ids)
        except (ValueError, KeyError, IndexError) as exc:
            raise ValueError(
                "Prepared sequences contain road IDs missing from the loaded graph. "
                "Check that FMM and PyG use the same shared road network."
            ) from exc
        
        model_ids = torch.as_tensor(model_ids, dtype=torch.long).cpu()
        
        if model_ids.shape != flattened_ids.shape or (
            (model_ids < 1) | (model_ids > graph_ids.numel())
        ).any():
            raise ValueError("Road-ID conversion returned invalid model IDs.")

        # Check the inverse lookup too, so unknown edge IDs cannot silently
        # map to some other valid road via a searchsorted insertion position.
        if not torch.equal(graph_ids[model_ids - 1], flattened_ids):
            raise ValueError(
                "Prepared sequences contain road IDs missing from the loaded graph. "
                "Check that FMM and PyG use the same shared road network."
            )

        # raw_ids is a list of list of road ids of a trajectory, thus its length is how long the trajectory was
        self.lengths = [len(sequence) for sequence in raw_ids]
        
        # Split the flattened road-ID tensor into individual trajectories
        # using their original sequence lengths. basically self.lengths tells which road ids belong to a
        # single trajectory in the 1D version model_ids so we are again grouping back to lists of trajectories.
        self._model_ids = list(model_ids.split(self.lengths))
        
        logger.info(
            f"Loaded {len(self)} trajectories from {path} "
            f"({self.filtered_count} excluded by min_points={min_points})."
        )

    def __len__(self) -> int:
        """Return the number of GPS trajectories available in the dataset."""
        return len(self._model_ids)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Return one trajectory's road IDs and position rates.

        Road IDs identify the map-matched road segments, converted to model IDs.
        Each rate represents the observation's relative position along its road.
        Both sequences have the same length and are aligned by observation.
        """
        return self._model_ids[index], self._rates[index]


class LengthBatchSampler(Sampler[list[int]]):
    """Group indices by trajectory length and optionally shuffle each epoch.

    Batches are homogeneous in length, so PyTorch's default collate function
    stacks them into tensors shaped [B, T] without any padding.
    """

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 42
        ):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        self.batch_size = batch_size
        
        # if self.shuffle=True then trajectories in a group are shuffled and the groups ordering are shuffled too
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = torch.Generator().manual_seed(seed)
        
        self.groups: dict[int, list[int]] = defaultdict(list)
        for index, length in enumerate(lengths):
            self.groups[length].append(index)

    def __iter__(self) -> Iterator[list[int]]:
        batches: list[list[int]] = []
        for length in sorted(self.groups):
            # get a length grouped set of trajectories all having the same length
            indices = self.groups[length].copy()
            if self.shuffle:
                # indices of the order to shuffle the trajectories in the group
                permutation = torch.randperm(len(indices), generator=self.generator)
                # shuffle the trajectories in the group
                indices = [indices[i] for i in permutation.tolist()]

            # make batches of trajectories in the current length gruop
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        # shuffle the length groups if self.shuffle=True then trajectories ina group are shuffled and the groups ordering are shuffled too
        if self.shuffle:
            permutation = torch.randperm(len(batches), generator=self.generator)
            batches = [batches[i] for i in permutation.tolist()]

        yield from batches

    def __len__(self) -> int:
        sizes = (len(indices) for indices in self.groups.values())
        if self.drop_last:
            return sum(size // self.batch_size for size in sizes)
        return sum(ceil(size / self.batch_size) for size in sizes)


def make_length_dataloader(
    dataset: RoadSequenceDataset,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Create same-length batches of model IDs and raw position rates."""
    batch_sampler = LengthBatchSampler(
        dataset.lengths,
        batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
