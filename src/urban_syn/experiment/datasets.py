"""Prepare length-grouped diffusion loaders with optional trip-safe validation.

The existing RoadSequenceDataset performs road-ID conversion and input checks.
This module only selects rows, groups original trips, and creates loaders.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

from urban_syn.diffusion.data import LengthBatchSampler, RoadSequenceDataset

from .config import ResolvedExperimentConfig


@dataclass(frozen=True)
class DatasetSplit:
    """Training and optional validation indices over one prepared dataset.

    Attributes
    ----------
    train_indices : tuple[int, ...]
        Indices used for optimizer updates.
    validation_indices : tuple[int, ...]
        Held-out training-piece indices, empty when validation is disabled.
    train_fmm_ids : tuple[int, ...]
        FMM IDs corresponding to train_indices.
    validation_fmm_ids : tuple[int, ...]
        FMM IDs corresponding to validation_indices.
    """

    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    train_fmm_ids: tuple[int, ...]
    validation_fmm_ids: tuple[int, ...]


def split_prepared_sequences(
    fmm_ids: list[int],
    mapping_csv: Path | None,
    fraction: float,
    seed: int,
    enabled: bool
) -> DatasetSplit:
    """Split the training pieces by original (uid, tid) when requested.

    Parameters
    ----------
    fmm_ids : list[int]
        Ordered IDs corresponding to RoadSequenceDataset rows.
    mapping_csv : Path | None
        FMM-to-original-trip CSV, required only when validation is enabled.
    fraction : float
        Fraction of distinct original trips assigned to validation.
    seed : int
        Seed for reproducible original-trip selection.
    enabled : bool
        Whether validation examples should be held out.

    Returns
    -------
    DatasetSplit
        Nonoverlapping train/validation indices and source IDs.

    Raises
    ------
    ValueError
        If the mapping is malformed or too few distinct trips exist.
    """
    if len(fmm_ids) != len(set(fmm_ids)):
        raise ValueError("Prepared training FMM IDs must be unique.")

    indices = tuple(range(len(fmm_ids)))
    if not enabled:
        return DatasetSplit(indices, (), tuple(fmm_ids), ())

    if mapping_csv is None:
        raise ValueError("A trip_id_map CSV is required when validation is enabled.")
    if not 0 < fraction < 1:
        raise ValueError("Validation fraction must be between zero and one.")

    mapping = pd.read_csv(mapping_csv)
    columns = ("fmm_id", "uid", "tid")
    missing = set(columns) - set(mapping.columns)
    if missing:
        raise ValueError(f"Trip mapping is missing columns: {sorted(missing)}")
    mapping = mapping.loc[:, list(columns)].copy()
    if mapping.isna().any().any():
        raise ValueError("FMM trip mapping contains null IDs or trip identities.")
    try:
        mapping["fmm_id"] = pd.to_numeric(mapping["fmm_id"], errors="raise").astype("int64")
    except (ValueError, TypeError) as exc:
        raise ValueError("Trip mapping fmm_id values must be integers.") from exc
    if mapping["fmm_id"].duplicated().any():
        raise ValueError("Trip mapping contains duplicate fmm_id rows.")

    # Reindex in the precise sequence-dataset order; a missing join is unsafe.
    indexed = mapping.set_index("fmm_id").reindex(fmm_ids)
    if indexed[["uid", "tid"]].isna().any().any():
        raise ValueError("Some prepared training FMM IDs are missing from the trip mapping.")

    groups = pd.MultiIndex.from_frame(indexed[["uid", "tid"]])
    group_codes, unique_groups = pd.factorize(groups, sort=False)
    group_count = len(unique_groups)
    if group_count < 2:
        raise ValueError("Validation requires at least two distinct original trips.")

    validation_group_count = max(1, min(group_count - 1, int(np.ceil(group_count * fraction))))
    generator = np.random.default_rng(seed)
    selected_groups = set(generator.permutation(group_count)[:validation_group_count].tolist())

    train = tuple(i for i, code in enumerate(group_codes) if int(code) not in selected_groups)
    validation = tuple(i for i, code in enumerate(group_codes) if int(code) in selected_groups)
    if not train or not validation:
        raise ValueError("Validation produced an empty training or validation partition.")

    return DatasetSplit(
        train_indices=train,
        validation_indices=validation,
        train_fmm_ids=tuple(fmm_ids[i] for i in train),
        validation_fmm_ids=tuple(fmm_ids[i] for i in validation)
    )


def build_epoch_loader(
    dataset: RoadSequenceDataset,
    indices: tuple[int, ...],
    batch_size: int,
    seed: int,
    num_workers: int,
    shuffle: bool,
    pin_memory: bool
) -> DataLoader:
    """Build an epoch-reproducible, same-length trajectory data loader.

    Parameters
    ----------
    dataset : RoadSequenceDataset
        Source training dataset with one-based model road IDs.
    indices : tuple[int, ...]
        Source rows selected for this data partition.
    batch_size : int
        Number of equally long trajectories per mini-batch.
    seed : int
        Per-epoch shuffle seed. Do not share a mutable sampler RNG across epochs.
    num_workers : int
        DataLoader worker count.
    shuffle : bool
        Shuffle both within length groups and the resulting batch list.
    pin_memory : bool
        Whether to pin host tensors for device transfer.

    Returns
    -------
    DataLoader
        Loader yielding (road_ids, raw_rates) tensors of shape [B, T].
    """
    if not indices:
        raise ValueError("Cannot create a DataLoader from an empty partition.")

    selected = Subset(dataset, list(indices))
    lengths = [dataset.lengths[i] for i in indices]
    sampler = LengthBatchSampler(
        lengths=lengths,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        seed=seed
    )
    return DataLoader(
        selected,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


def load_training_dataset(config: ResolvedExperimentConfig, graph_edge_ids) -> tuple[RoadSequenceDataset, DatasetSplit]:
    """Load prepared train-only trajectories and split them if requested.

    Parameters
    ----------
    config : ResolvedExperimentConfig
        Experiment paths, random seed, and validation settings.
    graph_edge_ids : Tensor
        Road IDs in the selected PyG graph node ordering.

    Returns
    -------
    tuple[RoadSequenceDataset, DatasetSplit]
        Loaded road sequences and indices for the current experiment.
    """
    dataset = RoadSequenceDataset(config.settings.paths.train_sequences, graph_edge_ids)
    settings = config.settings.training.validation
    selection = split_prepared_sequences(
        dataset.fmm_ids,
        config.settings.paths.trip_id_map,
        settings.fraction,
        config.settings.experiment.seed,
        settings.enabled
    )
    return dataset, selection
