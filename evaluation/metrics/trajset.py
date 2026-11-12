"""TrajSet: a set of trajectories as flat numpy arrays, which is what every metric works on."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from evaluation.geo import haversine_m


@dataclass
class TrajSet:
    """Trajectories stored as one long array of points plus where each trajectory starts.

    Trajectory k has the points lat[offsets[k]:offsets[k + 1]] (and the same for lng).
    """

    ids: np.ndarray      # (n,) traj_id of each trajectory
    refs: np.ndarray     # (n,) ref_id of each trajectory (None/NaN when there is none)
    offsets: np.ndarray  # (n + 1,) start of each trajectory in lat/lng, then the total
    lat: np.ndarray      # (points,)
    lng: np.ndarray      # (points,)

    @classmethod
    def from_table(cls, table: pd.DataFrame) -> "TrajSet":
        """Build a TrajSet from a GPS table, keeping the trajectories in order of first appearance."""
        codes, ids = pd.factorize(table["traj_id"], sort=False)
        order = np.lexsort((table["seq"].to_numpy(), codes))
        counts = np.bincount(codes, minlength=len(ids))
        offsets = np.concatenate([[0], np.cumsum(counts)])
        refs = table["ref_id"].to_numpy(dtype=object)[order][offsets[:-1]]
        return cls(
            ids=np.asarray(ids, dtype=object),
            refs=np.array([None if pd.isna(r) else str(r) for r in refs], dtype=object),
            offsets=offsets,
            lat=table["lat"].to_numpy(dtype=float)[order],
            lng=table["lng"].to_numpy(dtype=float)[order],
        )

    def __len__(self) -> int:
        """Number of trajectories."""
        return len(self.ids)

    @property
    def counts(self) -> np.ndarray:
        """Number of points in each trajectory."""
        return np.diff(self.offsets)

    def owner(self) -> np.ndarray:
        """For every point, the index of the trajectory it belongs to."""
        return np.repeat(np.arange(len(self)), self.counts)

    def take(self, index) -> "TrajSet":
        """Return the trajectories at `index` (in that order) as a new TrajSet."""
        index = np.asarray(index, dtype=int)
        counts = self.counts[index]
        new_starts = np.cumsum(counts) - counts
        points = np.repeat(self.offsets[:-1][index] - new_starts, counts) + np.arange(counts.sum())
        return TrajSet(self.ids[index], self.refs[index], np.concatenate([[0], np.cumsum(counts)]),
                       self.lat[points], self.lng[points])

    def points(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the lat and lng arrays of trajectory k."""
        a, b = self.offsets[k], self.offsets[k + 1]
        return self.lat[a:b], self.lng[a:b]

    def first_points(self) -> tuple[np.ndarray, np.ndarray]:
        """lat and lng of each trajectory's first point."""
        return self.lat[self.offsets[:-1]], self.lng[self.offsets[:-1]]

    def last_points(self) -> tuple[np.ndarray, np.ndarray]:
        """lat and lng of each trajectory's last point."""
        return self.lat[self.offsets[1:] - 1], self.lng[self.offsets[1:] - 1]

    def lengths_m(self) -> np.ndarray:
        """Total path length L(T) of each trajectory in metres (sum of Haversine steps)."""
        steps = haversine_m(self.lat[:-1], self.lng[:-1], self.lat[1:], self.lng[1:])
        steps[self.offsets[1:-1] - 1] = 0.0  # no step from one trajectory's last point to the next one's first
        cumulative = np.concatenate([[0.0], np.cumsum(steps)])
        return cumulative[self.offsets[1:] - 1] - cumulative[self.offsets[:-1]]

    def radius_m(self) -> np.ndarray:
        """Radius r(T) of each trajectory in metres: mean Haversine distance to the arithmetic center.

        This is the TS-TrajGen definition the spec uses, not the RMS radius of gyration.
        """
        owner = self.owner()
        center_lat = np.bincount(owner, self.lat) / self.counts
        center_lng = np.bincount(owner, self.lng) / self.counts
        distance = haversine_m(self.lat, self.lng, center_lat[owner], center_lng[owner])
        return np.bincount(owner, distance) / self.counts


def sample_index(n: int, size: int | None, seed: int) -> np.ndarray:
    """Pick `size` of n items without replacement (all of them, in order, when size is None or >= n)."""
    if size is None or size >= n:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=size, replace=False))
