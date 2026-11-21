"""Load link-pretrained GNN weights into a joint diffusion road encoder.

Only GNN weights are transferred. The link decoder, optimizer and sampler
are specific to pretraining and are not used by the diffusion model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch_geometric.data import Data

from urban_syn.gnn.config import parse_gnn_config

from .config import ResolvedExperimentConfig, _read_yaml_mapping


def _selected_graph_settings(config: ResolvedExperimentConfig) -> dict[str, Any]:
    """Read the ordered feature-group selection from the joint graph YAML."""
    graph = _read_yaml_mapping(config.settings.paths.graph_config)
    selected = graph.get("model", {})
    if not isinstance(selected, dict):
        raise ValueError("Joint graph YAML requires a model mapping.")
    return selected


def _gnn_architecture(value: Any) -> dict[str, Any]:
    """Normalize default GNN settings and ignore inactive layer options."""
    options = parse_gnn_config(value).model_dump(mode="json")
    unused = "gine" if options["architecture"] == "gatv2" else "gatv2"
    options.pop(unused, None)
    return options


def _mismatches(
    metadata: dict[str, Any],
    graph: Data,
    config: ResolvedExperimentConfig
) -> list[str]:
    """Explain incompatible encoder input settings without checking road count.

    The number and identity of roads may vary across compatible datasets.
    The encoder parameter shapes depend on input widths and GNN settings,
    while the feature-group order must retain the same meaning.
    """
    issues = []
    selected = _selected_graph_settings(config)
    feature_selections = (
        ("selected_node_features", "node_features", ["x"]),
        ("selected_edge_features", "edge_features", [])
    )
    for saved_key, graph_key, default in feature_selections:
        previous = list(metadata.get(saved_key, default))
        current = list(selected.get(graph_key, default))
        if previous != current:
            issues.append(
                f"{graph_key} selection/order differs: checkpoint={previous}, graph={current}"
            )

    edge_attr = getattr(graph, "edge_attr", None)
    dimensions = {
        "node_features": int(graph.x.shape[1]),
        "edge_features": 0 if edge_attr is None else int(edge_attr.shape[1])
    }
    for key, expected in dimensions.items():
        saved = metadata.get(key)
        if saved != expected:
            issues.append(f"{key} width differs: checkpoint={saved}, graph={expected}")

    try:
        previous_gnn = _gnn_architecture(metadata["gnn"])
        current_gnn = _gnn_architecture(config.joint_settings.gnn.model_dump())
    except (KeyError, TypeError, ValueError) as error:
        issues.append(f"GNN architecture settings could not be read: {error}")
    else:
        if previous_gnn != current_gnn:
            differences = [
                key for key in current_gnn
                if previous_gnn.get(key) != current_gnn[key]
            ]
            issues.append(
                "GNN architecture settings differ: " + ", ".join(differences)
            )
    return issues


def load_pretrained_encoder(
    model: nn.Module,
    graph: Data,
    config: ResolvedExperimentConfig
) -> None:
    """Validate and load ``encoder.pt`` exported by road-link pretraining.

    Args:
        model: Fresh joint model with a ``road_encoder`` attribute.
        graph: Joint graph with selected features in zero-based PyG road order.
        config: Resolved training settings pointing to an encoder-only file.

    Raises:
        FileNotFoundError: The configured checkpoint does not exist.
        ValueError: The file or encoder is incompatible, with concrete reasons.

    Notes:
        Pretraining and diffusion may use different message-passing edges.
        No equality check is imposed on the number/order of roads across files:
        only encoder weights are transferred, not a road-specific embedding table.
    """
    source = config.settings.model.road_encoder.checkpoint
    if source is None:
        raise ValueError("No model.road_encoder.checkpoint was configured.")
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(
            f"Pretrained GNN file not found: {source}. "
            "Set model.road_encoder.checkpoint to the pretraining run's checkpoints/encoder.pt."
        )

    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError(
            f"Unsupported encoder checkpoint: {source}. "
            "Expected the pretraining run's encoder.pt (format_version=1), "
            "not its full best.pt checkpoint."
        )
    state = payload.get("encoder_state_dict")
    metadata = payload.get("model_metadata")
    if not isinstance(state, dict) or not isinstance(metadata, dict):
        raise ValueError(
            f"Invalid encoder checkpoint: {source}. "
            "Expected encoder_state_dict and model_metadata."
        )

    issues = _mismatches(metadata, graph, config)
    if issues:
        reasons = "\n  - ".join(issues)
        raise ValueError(
            f"Pretrained GNN is incompatible with the selected graph/model:\n  - {reasons}\n"
            "Use a checkpoint with matching ordered feature groups, input widths "
            "and GNN architecture. The number of roads does not need to match."
        )

    # Compare shapes before load_state_dict to give useful errors instead of
    # a long PyTorch size-mismatch trace for each incompatible tensor.
    current_state = model.road_encoder.state_dict()
    saved_keys = set(state)
    current_keys = set(current_state)
    if saved_keys != current_keys:
        missing = sorted(current_keys - saved_keys)
        extra = sorted(saved_keys - current_keys)
        raise ValueError(
            "Pretrained GNN has different parameter names. "
            f"Missing={missing[:5]}, unexpected={extra[:5]}. "
            "Check the GNN architecture and code version."
        )
    for name, current in current_state.items():
        saved = state[name]
        if not isinstance(saved, torch.Tensor) or saved.shape != current.shape:
            raise ValueError(
                f"Pretrained GNN tensor {name!r} has shape "
                f"{getattr(saved, 'shape', None)}; current model expects {current.shape}. "
                "Check the selected graph feature widths and GNN settings."
            )

    model.road_encoder.load_state_dict(state, strict=True)
