"""The shared GPS grid used by the geographic, OD and Jaccard metrics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OUTSIDE = -1  # cell number of every point outside the grid


@dataclass(frozen=True)
class Grid:
    """A regular lat/lng grid. Cells are numbered row * n_lng + column, from the south-west corner."""

    cell_deg: float
    lng_min: float
    lat_min: float
    n_lng: int
    n_lat: int

    @classmethod
    def from_settings(cls, cell_deg: float, lng_bounds, lat_bounds) -> "Grid":
        """Build the grid for one cell size over the spec's bounds.

        The number of cells per side is int(range / cell) + 2, which gives the spec's 87 x 90 grid for 0.005 degrees and
        the same bounds for every cell size.
        """
        n_lng = int((lng_bounds[1] - lng_bounds[0]) / cell_deg) + 2
        n_lat = int((lat_bounds[1] - lat_bounds[0]) / cell_deg) + 2
        return cls(float(cell_deg), float(lng_bounds[0]), float(lat_bounds[0]), n_lng, n_lat)

    @property
    def size(self) -> int:
        """Number of cells inside the grid."""
        return self.n_lng * self.n_lat

    def cells(self, lat, lng) -> np.ndarray:
        """Cell number of each point (floor assignment), OUTSIDE (-1) for points outside the grid.

        Points outside are not moved into the border cells; every grid metric counts them in one extra "outside" cell,
        so their share is kept and can be reported.
        """
        col = np.floor((np.asarray(lng, dtype=float) - self.lng_min) / self.cell_deg).astype(np.int64)
        row = np.floor((np.asarray(lat, dtype=float) - self.lat_min) / self.cell_deg).astype(np.int64)
        inside = (col >= 0) & (col < self.n_lng) & (row >= 0) & (row < self.n_lat)
        return np.where(inside, row * self.n_lng + col, OUTSIDE)
