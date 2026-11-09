import torch
from torch_geometric.utils import degree, contains_self_loops
from torch_geometric.data import Data
from collections.abc import Sequence
from typing import Any

def graph_info(
    data: Data,
    preprocessor: Any | None = None,
    selected_groups: Sequence[str] | None = None,
) -> str:
    """Summarize a road graph and its optional GNN input feature selection.

    Args:
        data: PyG road graph before selecting model-input feature groups.
        preprocessor: Fitted road-feature preprocessor, if available; used
            to retrieve the individual names of columns in ``data.x``.
        selected_groups: Node-feature attributes selected for the GNN,
            in concatenation order (for example, ["x", "degree_features"]).

    Returns:
        Formatted graph statistics, attributes, and feature names.
    """
    if data.edge_index is None or data.num_nodes is None:
        raise ValueError("Graph requires edge_index and num_nodes.")

    num_nodes, num_edges = data.num_nodes, data.num_edges
    lines = ["=== GRAPH SUMMARY ===", "-" * 40]
    lines.append(f"Road nodes: {num_nodes:,}")
    lines.append(
        f"Road transitions: {num_edges:,} "
        f"({'undirected' if data.is_undirected() else 'directed'})"
    )

    for key, label in (
        ("x", "Base node features"),
        ("edge_attr", "Transition features"),
        ("y", "Labels"),
    ):
        value = getattr(data, key, None)
        if isinstance(value, torch.Tensor):
            lines.append(f"{label}: {tuple(value.shape)}")
        elif key == "x":
            lines.append(f"{label}: None")

    pos = getattr(data, "pos", None)
    if isinstance(pos, torch.Tensor):
        lines.append(f"GPS positions (lon, lat): {tuple(pos.shape)}")

    road_ids = getattr(data, "edge_id", None)
    if isinstance(road_ids, torch.Tensor) and road_ids.numel():
        lines.append(
            f"Shared road-network edge IDs: {torch.unique(road_ids).numel():,} "
            f"unique, range [{road_ids.min().item()}, {road_ids.max().item()}]"
        )

    # Retrieve named base features from the fitted sklearn preprocessor.
    base_names: list[str] = []
    if preprocessor is not None:
        base_names = list(map(str, preprocessor.get_feature_names_out()))
        if not isinstance(data.x, torch.Tensor) or len(base_names) != data.x.size(1):
            raise ValueError("Preprocessor feature names do not match data.x width.")
        lines.append(f"Base feature names ({len(base_names)}): {base_names}")

    # GNN input selection is reported without changing the graph or running
    # select_node_features() a second time.
    if selected_groups is not None:
        if not selected_groups or len(set(selected_groups)) != len(selected_groups):
            raise ValueError("Select one or more unique node-feature groups.")

        selected_names: list[str] = []
        selected_width = 0
        lines.append(f"Selected GNN node groups: {list(selected_groups)}")
        for name in selected_groups:
            value = getattr(data, name, None)
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 2
                or value.size(0) != num_nodes
            ):
                raise ValueError(f"Invalid selected node-feature group: {name}")

            width = value.size(1)
            selected_width += width
            if name == "x" and base_names:
                names = base_names
            elif width == 1:
                names = [name]
            elif name == "degree_features" and width == 2:
                names = ["in_degree", "out_degree"]
            else:
                names = [f"{name}[{index}]" for index in range(width)]
            selected_names.extend(names)
            lines.append(f"  {name}: {tuple(value.shape)}")

        lines.append(f"Selected GNN input shape: ({num_nodes}, {selected_width})")
        lines.append(f"Selected GNN feature names ({len(selected_names)}): {selected_names}")

    lines.append(f"Self-loops: {contains_self_loops(data.edge_index)}")
    if num_nodes > 0:
        source, target = data.edge_index
        out_degree = degree(source, num_nodes=num_nodes)
        in_degree = degree(target, num_nodes=num_nodes)
        isolated = ((in_degree == 0) & (out_degree == 0)).sum().item()
        lines.append(f"Has isolated nodes: {isolated > 0}")
        lines.append(f"Isolated nodes: {isolated:,} / {num_nodes:,}")
        lines.append(
            f"Mean in/out degree: {in_degree.mean().item():.2f} / "
            f"{out_degree.mean().item():.2f}"
        )
        lines.append(
            f"Max in/out degree: {in_degree.max().item():.0f} / "
            f"{out_degree.max().item():.0f}"
        )

    possible_edges = num_nodes * (num_nodes - 1)
    density = num_edges / possible_edges if possible_edges else 0.0
    lines.append(f"Density (approx.): {density:.6f}")
    lines.append(f"Device: {data.edge_index.device}")

    lines.append("\nAttributes:")
    for name, value in data:
        if isinstance(value, torch.Tensor):
            lines.append(
                f"  {name}: shape={tuple(value.shape)}, "
                f"dtype={value.dtype}, device={value.device}"
            )
        else:
            lines.append(f"  {name}: {type(value).__name__}")

    total_bytes = sum(
        value.numel() * value.element_size()
        for _, value in data
        if isinstance(value, torch.Tensor)
    )
    lines.append(f"\nApprox. tensor memory: {total_bytes / 1e6:.2f} MB")

    labels = getattr(data, "y", None)
    if (
        isinstance(labels, torch.Tensor)
        and labels.ndim == 1
        and labels.numel() > 0
        and labels.dtype in (torch.int32, torch.int64)
        and labels.min().item() >= 0
    ):
        class_ids, counts = torch.unique(labels, return_counts=True)
        lines.append(
            f"Label distribution: {dict(zip(class_ids.tolist(), counts.tolist()))}"
        )

    lines.append("-" * 40)
    return "\n".join(lines)
