"""Trajectory-indexed windows and artifact validation for weighted pretraining."""

import csv
import math
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Dataset


def split_trajectory_indices(parents, fraction=0.2, shuffle=True, seed=42):
    """Shared parent split for fitting gravity statistics and training."""
    if not np.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError("validation_split must be finite and in [0, 1)")
    order = list(dict.fromkeys(parents))
    if shuffle:
        np.random.RandomState(seed).shuffle(order)
    held_out = set(order[:int(np.floor(fraction * len(order)))])
    return ([i for i, parent in enumerate(parents) if parent not in held_out],
            [i for i, parent in enumerate(parents) if parent in held_out])


class GravityTrajectoryDataset(Dataset):
    """One index per CharDataset trajectory; one random local window per access.

    CharDataset already brackets each trajectory with EOS. Only real next-token
    predictions (including the terminal EOS) contribute to the loss. Padding
    inputs use EOS and padding targets use the model's existing ignore index.
    PyTorch's worker RNG seeds also govern random offsets in DataLoader workers.
    """

    def __init__(self, dataset):
        self.base_dataset = dataset
        self.trajs = dataset.trajs
        self.block_size = dataset.get_block_size()
        if not isinstance(self.block_size, int) or self.block_size <= 0:
            raise ValueError("Gravity pretraining requires a positive integer block size")
        self.eos_id = dataset.stoi[dataset.EOS_TOKEN]
        self.trajectories = [
            torch.tensor([dataset.stoi[token] for token in trajectory], dtype=torch.long)
            for trajectory in self.trajs
        ]

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, index):
        trajectory = self.trajectories[index]
        predictions = len(trajectory) - 1
        max_offset = max(0, predictions - self.block_size)
        offset = torch.randint(max_offset + 1, (1,)).item() if max_offset else 0
        length = min(self.block_size, predictions - offset)
        x = torch.full((self.block_size,), self.eos_id, dtype=torch.long)
        y = torch.full((self.block_size,), -1, dtype=torch.long)
        x[:length] = trajectory[offset:offset + length]
        y[:length] = trajectory[offset + 1:offset + length + 1]
        return x, y

    def get_block_size(self):
        return self.block_size

    def get_vocab_size(self):
        return self.base_dataset.get_vocab_size()


class DeterministicTrajectoryWindows(Dataset):
    """Cover every held-out next-token target once, including terminal EOS."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.windows = [(index, offset)
                        for index in sorted(indices)
                        for offset in range(0, len(dataset.trajectories[index]) - 1,
                                            dataset.block_size)]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        trajectory_index, offset = self.windows[index]
        dataset = self.dataset
        sequence = dataset.trajectories[trajectory_index]
        length = min(dataset.block_size, len(sequence) - 1 - offset)
        x = torch.full((dataset.block_size,), dataset.eos_id, dtype=torch.long)
        y = torch.full((dataset.block_size,), -1, dtype=torch.long)
        x[:length] = sequence[offset:offset + length]
        y[:length] = sequence[offset + 1:offset + length + 1]
        return x, y


def _read_rows(path, required):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if not required <= set(columns) or len(columns) != len(set(columns)):
            raise ValueError(f"{path}: expected unique CSV columns including {sorted(required)}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{path}: malformed CSV row")
    return rows, set(columns)


def _validate_trajectories(rows, dataset, path):
    if len(rows) != len(dataset):
        raise ValueError(
            f"{path}: trajectory count mismatch: CSV has {len(rows)}, text has {len(dataset)}"
        )
    for index, (row, trajectory) in enumerate(zip(rows, dataset.trajs)):
        if row["rid_list"].strip().split(",") != trajectory[1:-1]:
            raise ValueError(
                f"{path}: trajectory row {index} differs from the text dataset; "
                "road sequences and row order must match exactly. Regenerate the gravity CSV."
            )


def load_gravity_data(dataset, gravity_path, trajectory_path=None):
    """Validate row alignment, raw weights, and optional original parent IDs.

    The original trajectory CSV, when present, anchors trajectory/parent identity
    as well as road sequences. Without it, row-by-row road matching against the
    text is still mandatory; parent IDs can come from the gravity CSV itself.
    """
    gravity_path = Path(gravity_path)
    if not gravity_path.is_file():
        name = gravity_path.parent.name
        if name.endswith("-Taxi"):
            name = name[:-5]
        command = (
            "python -c 'from preprocessing.gravity_based_sampling import GravityBasedSampler; "
            f'GravityBasedSampler(dataset="{name}").process()' + "'"
        )
        raise FileNotFoundError(
            f"Missing gravity artifact: {gravity_path}. Generate it from the matching "
            f"training CSV before gravity-weighted pretraining, from the repository root:\n{command}"
        )
    rows, columns = _read_rows(gravity_path, {"rid_list", "gravity"})
    _validate_trajectories(rows, dataset, gravity_path)
    source_rows, source_columns = rows, columns
    if trajectory_path is not None and Path(trajectory_path).is_file():
        source_rows, source_columns = _read_rows(Path(trajectory_path), {"rid_list"})
        _validate_trajectories(source_rows, dataset, trajectory_path)
        for column in ("traj_id", "parent_traj_id"):
            if column in source_columns and column in columns:
                if [row[column] for row in rows] != [row[column] for row in source_rows]:
                    raise ValueError(f"{gravity_path}: {column} order/identity mismatch with {trajectory_path}")
            elif column == "traj_id" and column in source_columns:
                raise ValueError(f"{gravity_path}: missing traj_id column needed to verify source row order")
    for records, record_columns, path in (
        (rows, columns, gravity_path), (source_rows, source_columns, trajectory_path)
    ):
        if "traj_id" in record_columns:
            ids = [row["traj_id"] for row in records]
            if any(not value.strip() for value in ids) or len(set(ids)) != len(ids):
                raise ValueError(f"{path}: trajectory IDs must be nonempty and unique")
    parent_rows = source_rows if "parent_traj_id" in source_columns else rows
    parents = None
    if "parent_traj_id" in source_columns or "parent_traj_id" in columns:
        parents = [row["parent_traj_id"] for row in parent_rows]
        if any(not parent.strip() or parent != parent.strip() for parent in parents):
            raise ValueError(f"{gravity_path}: parent_traj_id must be nonempty for every row")
    weights = []
    for index, row in enumerate(rows):
        try:
            weight = float(row["gravity"])
        except ValueError as exc:
            raise ValueError(f"{gravity_path}: invalid gravity value at trajectory row {index}") from exc
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(
                f"{gravity_path}: gravity must be finite and nonnegative at trajectory row {index}"
            )
        weights.append(weight)
    return torch.tensor(weights, dtype=torch.double), parents
