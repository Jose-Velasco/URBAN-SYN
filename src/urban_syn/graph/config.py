"""Thin YAML adapter for road-graph preprocessing; core APIs stay YAML-free."""
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .builder import RoadGraphBuilder
from .dataset import RoadGraphDataset
from .features import SklearnRoadFeatureEncoder, make_road_preprocessor
from .transforms import make_graph_pre_transform

def load_graph_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML graph experiment configuration.

    Parameters
    ----------
    path : str | Path
        YAML file path (dataset paths within it are relative to working dir).

    Returns
    -------
    dict[str, Any]
        Validated top-level configuration mapping.

    Raises
    ------
    ValueError
        If the YAML root or required sections are invalid.
    """
    with Path(path).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Graph YAML must contain a mapping at its root.")

    for name in ("dataset", "features"):
        if not isinstance(config.get(name), dict):
            raise ValueError(f"Missing or invalid '{name}' YAML section.")
    return config


def build_dataset_from_config(
    config: Mapping[str, Any], force_reload: bool = False,
) -> RoadGraphDataset:
    """Construct the ordinary RoadGraphDataset from Python settings.

    Parameters
    ----------
    config : Mapping[str, Any]
        Configuration with dataset, features and optional graph sections.
    force_reload : bool, default=False
        Rebuild the graph even if the cache exists.

    Returns
    -------
    RoadGraphDataset
        Processed or cached PyG dataset; independent of YAML parsing.

    Raises
    ------
    ValueError
        If required configuration fields are absent or malformed.
    """
    dataset = config["dataset"]
    features = config["features"]
    graph = config.get("graph", {})

    if not isinstance(dataset, Mapping) or not isinstance(features, Mapping) or not isinstance(graph, Mapping):
        raise ValueError("dataset, features, and graph must be mappings.")
    if any(key not in dataset for key in ("root", "network_path")):
        raise ValueError("dataset.root and dataset.network_path are required.")
    if unknown := set(features) - {"numeric", "categorical", "boolean", "geometry"}:
        raise ValueError(f"Unsupported feature groups: {sorted(unknown)}")
    if unknown := set(graph) - {"allow_self_loops", "transforms"}:
        raise ValueError(f"Unsupported graph settings: {sorted(unknown)}")

    # The encoder and builder know nothing about where settings came from.
    preprocessor = make_road_preprocessor(
        numeric_columns=features.get("numeric", ()),
        categorical_columns=features.get("categorical", ()),
        boolean_columns=features.get("boolean", ()),
        geometry_columns=features.get("geometry", ()),
    )

    builder = RoadGraphBuilder(
        feature_encoder=SklearnRoadFeatureEncoder(preprocessor),
        allow_self_loops=graph.get("allow_self_loops", False),
    )

    return RoadGraphDataset(
        root=dataset["root"],
        network_path=dataset["network_path"],
        builder=builder,
        layer=dataset.get("layer", "roads"),
        pre_transform=make_graph_pre_transform(graph.get("transforms")),
        force_reload=force_reload,
    )
