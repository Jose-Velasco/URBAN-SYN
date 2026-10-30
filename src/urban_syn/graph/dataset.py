"""PyG in-memory dataset for one canonical physical road network."""
import logging
from collections.abc import Callable
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline
from torch_geometric.data import Data, InMemoryDataset

from urban_syn.graph.protocols import GraphBuilder

logger = logging.getLogger(__name__)


class RoadGraphDataset(InMemoryDataset):
    """Cache a road graph and its fitted feature preprocessor."""

    def __init__(
        self,
        root: str | Path,
        network_path: str | Path,
        builder: GraphBuilder,
        layer: str = "roads",
        pre_transform: Callable[[Data], Data] | None = None,
        force_reload: bool = False,
    ) -> None:
        """Initialize a road dataset, processing it if necessary.

        Parameters
        ----------
        root : str | Path
            Cache directory; use a distinct root per city/configuration.
        network_path : str | Path
            Existing canonical road-network GeoPackage.
        builder : GraphBuilder
            Graph builder
        layer : str, default="roads"
            GeoPackage layer containing road segments.
        pre_transform : Callable[[Data], Data] | None, optional
            Optional graph-level transformation applied once before saving.
        force_reload : bool, default=False
            Reprocess even when PyG finds cached artifacts.
        """
        self.network_path = Path(network_path).expanduser().resolve()
        self.layer = layer
        self.builder = builder

        if not self.network_path.is_file():
            raise FileNotFoundError(f"Road network not found: {self.network_path}")

        # InMemoryDataset checks processed_file_names and invokes process().
        super().__init__(
            root=str(root),
            pre_transform=pre_transform,
            force_reload=force_reload,
            log=False,
        )

        self.load(self.processed_paths[0])
        # joblib is used only with trusted, locally generated artifacts.
        # load previous fitted preprocessor for reuse if needed
        self.preprocessor: Pipeline = joblib.load(self.processed_paths[1])
        logger.info("Loaded cached road graph: %s", self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        """Return the source directory, avoiding a duplicate GeoPackage copy.

        Returns
        -------
        str
            Absolute directory containing the input network.
        """
        return str(self.network_path.parent)

    @property
    def raw_file_names(self) -> list[str]:
        """List the required source files.

        Returns
        -------
        list[str]
            Filename of the canonical GeoPackage.
        """
        return [self.network_path.name]

    @property
    def processed_file_names(self) -> list[str]:
        """List the artifacts PyG requires to consider processing complete.

        Returns
        -------
        list[str]
            Cached graph and fitted scikit-learn pipeline filenames.
        """
        return ["road_graph.pt", "preprocessor.joblib"]

    def process(self) -> None:
        """Construct and save the graph and fitted preprocessing pipeline."""
        logger.info("Processing road network: %s", self.network_path)
        graph = self.builder.build_from_file(self.network_path, layer=self.layer)

        if self.pre_transform is not None:
            graph = self.pre_transform(graph)

        if self.builder.fitted_preprocessor is None:
            raise RuntimeError("Graph builder did not fit a preprocessor.")

        # PyG handles the graph cache; we only persist sklearn's fitted state.
        self.save([graph], self.processed_paths[0])
        joblib.dump(self.builder.fitted_preprocessor, self.processed_paths[1])
        logger.info("Saved processed graph: %s", self.processed_paths[0])
