"""Protocols for interchangeable road-graph construction components."""
from pathlib import Path
from typing import Protocol

import geopandas as gpd
import pandas as pd
import torch
from sklearn.pipeline import Pipeline
from torch_geometric.data import Data

class RoadFeatureEncoder(Protocol):
    """Define the interface for encoding road-node features."""

    @property
    def fitted_preprocessor(self) -> Pipeline | None:
        """Return the fitted preprocessing pipeline, when available.

        Returns
        -------
        Pipeline | None
            Fitted feature preprocessor or None.
        """
        ...

    def encode(self, roads: gpd.GeoDataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode road-node features and geographic positions.

        Parameters
        ----------
        roads : gpd.GeoDataFrame
            Ordered canonical road segments.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Node feature tensor [N, F] and lon/lat positions [N, 2].
        """
        ...


class GraphBuilder(Protocol):
    """Define the interface for constructing PyG road-network graphs."""

    @property
    def fitted_preprocessor(self) -> Pipeline | None:
        """Return fitted preprocessing for dataset caching.

        Returns
        -------
        Pipeline | None
            Feature preprocessor or None before fitting.
        """
        ...

    def build(self, roads: pd.DataFrame) -> Data:
        """Construct a road graph from canonical road data.

        Parameters
        ----------
        roads : pd.DataFrame
            Canonical road network with IDs, endpoints, and attributes.

        Returns
        -------
        Data
            Directed road-as-node PyG graph.
        """
        ...

    def build_from_file(self, path: str | Path, layer: str = "roads") -> Data:
        """Load a road network and construct its graph.

        Parameters
        ----------
        path : str | Path
            Road-network file path.
        layer : str, default="roads"
            Source road layer.

        Returns
        -------
        Data
            Directed PyG road graph.
        """
        ...
