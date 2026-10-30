"""Construct a directed road-as-node graph from a canonical road table."""
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import Pipeline
from torch_geometric.data import Data

from .protocols import RoadFeatureEncoder

logger = logging.getLogger(__name__)


def prepare_roads(roads: pd.DataFrame) -> pd.DataFrame:
    """Validate canonical road identifiers and define PyG node ordering.

    Parameters
    ----------
    roads : pd.DataFrame
        Canonical road rows with edge_id, u, and v.

    Returns
    -------
    pd.DataFrame
        Roads sorted by canonical edge_id, with zero-based _node_idx.

    Raises
    ------
    ValueError
        If identifiers are absent, null, or duplicated.
    """
    missing = {"edge_id", "u", "v"} - set(roads.columns)
    if missing:
        raise ValueError(f"Missing road columns: {sorted(missing)}")
    if roads.empty:
        raise ValueError("Road network cannot be empty.")
    if roads[["edge_id", "u", "v"]].isna().any().any():
        raise ValueError("Road IDs and endpoint identifiers cannot be null.")
    if not roads["edge_id"].is_unique:
        raise ValueError("Canonical edge_id values must be unique.")

    ordered = roads.sort_values("edge_id").reset_index(drop=True).copy()
    # PyG node indices are positional, not necessarily canonical edge IDs.
    ordered["_node_idx"] = np.arange(len(ordered), dtype=np.int64)
    return ordered


def build_edge_index(
    roads: pd.DataFrame, allow_self_loops: bool = False
) -> torch.Tensor:
    """Link directed road segments that share compatible endpoints.

    Parameters
    ----------
    roads : pd.DataFrame
        Prepared road table with _node_idx, u, and v.
    allow_self_loops : bool, default=False
        Whether a road may have a transition to itself.

    Returns
    -------
    torch.Tensor
        Directed long edge_index with shape [2, num_transitions].
    """
    arrivals = roads[["_node_idx", "v"]].rename(
        columns={"_node_idx": "source", "v": "junction"}
    )
    departures = roads[["_node_idx", "u"]].rename(
        columns={"_node_idx": "target", "u": "junction"}
    )

    # Road A -> B when A.v == B.u. Turn restrictions are not yet enforced.
    connections = arrivals.merge(departures, on="junction")[["source", "target"]]
    if not allow_self_loops:
        connections = connections.loc[connections["source"] != connections["target"]]
    connections = connections.drop_duplicates().sort_values(["source", "target"])

    # PyG uses source IDs in row 0 and destination IDs in row 1.
    return torch.from_numpy(connections.to_numpy(dtype=np.int64).T.copy())


class RoadGraphBuilder:
    """Assemble injected node features, directed topology, and road IDs."""

    def __init__(
        self,
        feature_encoder: RoadFeatureEncoder,
        allow_self_loops: bool = False,
    ) -> None:
        """Initialize graph construction without selecting model features.

        Parameters
        ----------
        feature_encoder : RoadFeatureEncoder
            Encoder configured independently with sklearn preprocessing.
        allow_self_loops : bool, default=False
            Whether to preserve road-to-itself transitions.
        """
        self.feature_encoder = feature_encoder
        self.allow_self_loops = allow_self_loops

    @property
    def fitted_preprocessor(self) -> Pipeline | None:
        """Expose fitted node-feature preprocessing for dataset caching.

        Returns
        -------
        Pipeline | None
            Fitted sklearn transformer or None before fitting.
        """
        return self.feature_encoder.fitted_preprocessor

    def build(self, roads: pd.DataFrame) -> Data:
        """Construct a road-segment graph from canonical road attributes.

        Parameters
        ----------
        roads : pd.DataFrame
            Canonical road GeoDataFrame with directed endpoints.

        Returns
        -------
        Data
            PyG graph containing x, pos, edge_index, edge_id, num_nodes.

        Raises
        ------
        TypeError
            If the roads are not a GeoDataFrame with geometry.
        """
        if not isinstance(roads, gpd.GeoDataFrame):
            raise TypeError("Road feature encoding requires a GeoDataFrame.")

        ordered = prepare_roads(roads)
        x, pos = self.feature_encoder.encode(ordered)

        edge_index = build_edge_index(ordered, self.allow_self_loops)
        edge_ids = torch.from_numpy(
            ordered["edge_id"].to_numpy(dtype=np.int64).copy()
        )

        graph = Data(
            x=x, pos=pos, edge_index=edge_index,
            edge_id=edge_ids, num_nodes=len(ordered),
        )

        logger.info(
            "Built graph: %d roads, %d transitions, %d node features",
            graph.num_nodes, graph.num_edges, graph.num_node_features,
        )

        return graph

    def build_from_file(self, path: str | Path, layer: str = "roads") -> Data:
        """Load a canonical road GeoPackage and construct its PyG graph.

        Parameters
        ----------
        path : str | Path
            Source road-network GeoPackage.
        layer : str, default="roads"
            GeoPackage road layer.

        Returns
        -------
        Data
            Constructed graph in canonical road-ID order.
        """
        logger.info("Reading road network: %s (layer=%s)", path, layer)
        return self.build(gpd.read_file(path, layer=layer))
