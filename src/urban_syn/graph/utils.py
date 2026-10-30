import torch
from torch_geometric.utils import degree, contains_self_loops
from torch_geometric.data import Data

def graph_info(data: Data) -> str:
    """Return a formatted summary of a PyG graph.

    Parameters
    ----------
    data : Data
        PyTorch Geometric graph to summarize.

    Returns
    -------
    str
        Multiline summary of graph structure, features, and memory usage.
    """
    if data.edge_index is None or data.num_nodes is None:
        raise ValueError("Graph requires edge_index and num_nodes.")

    num_nodes = data.num_nodes
    num_edges = data.num_edges
    lines = ["=== GRAPH SUMMARY ===", "-" * 40]

    # Basic graph information.
    lines.append(f"Nodes: {num_nodes:,}")
    lines.append(
        f"Edges: {num_edges:,} "
        f"({'undirected' if data.is_undirected() else 'directed'})"
    )

    # Node, edge, and label dimensions.
    for name, label in (
        ("x", "Node features"),
        ("edge_attr", "Edge features"),
        ("y", "Labels"),
    ):
        value = getattr(data, name, None)
        if isinstance(value, torch.Tensor):
            lines.append(f"{label}: {tuple(value.shape)}")
        elif name == "x":
            lines.append("Node features: None")

    # Graph connectivity statistics.
    lines.append(
        f"Self-loops: {contains_self_loops(data.edge_index)}"
    )

    if num_nodes > 0:
        source, target = data.edge_index

        # In a directed graph, isolated nodes have neither incoming
        # nor outgoing connections. Sinks are not necessarily isolated.
        out_degree = degree(source, num_nodes=num_nodes)
        in_degree = degree(target, num_nodes=num_nodes)

        isolated = ((in_degree == 0) & (out_degree == 0)).sum().item()

        lines.append(f"Isolated nodes: {isolated:,} / {num_nodes:,}")
        lines.append(
            f"Mean in/out degree: "
            f"{in_degree.mean().item():.2f} / "
            f"{out_degree.mean().item():.2f}"
        )
        lines.append(
            f"Max in/out degree: "
            f"{in_degree.max().item():.0f} / "
            f"{out_degree.max().item():.0f}"
        )

    # Density excludes self-loops from the number of possible edges.
    possible_edges = num_nodes * (num_nodes - 1)
    density = num_edges / possible_edges if possible_edges else 0.0
    lines.append(f"Density (approx.): {density:.6f}")

    lines.append(f"Device: {data.edge_index.device}")

    # Display all graph attributes and their shapes.
    lines.append("\nAttributes:")
    for name, value in data:
        if isinstance(value, torch.Tensor):
            lines.append(
                f"  {name}: shape={tuple(value.shape)}, "
                f"dtype={value.dtype}, device={value.device}"
            )
        else:
            lines.append(f"  {name}: {type(value).__name__}")

    # Estimate storage occupied by tensor attributes.
    # This does not include Python objects or PyTorch runtime overhead.
    total_bytes = sum(
        value.numel() * value.element_size()
        for _, value in data
        if isinstance(value, torch.Tensor)
    )
    lines.append(f"\nApprox. tensor memory: {total_bytes / 1e6:.2f} MB")

    # Optional classification summary (not applicable to our road graph).
    labels = getattr(data, "y", None)
    if (
        isinstance(labels, torch.Tensor)
        and labels.ndim == 1
        and labels.numel() > 0
        and labels.dtype in (torch.int32, torch.int64)
        and labels.min().item() >= 0
    ):
        class_ids, counts = torch.unique(
            labels, return_counts=True
        )
        distribution = dict(zip(
            class_ids.tolist(),
            counts.tolist(),
        ))
        lines.append(f"Label distribution: {distribution}")

    lines.append("-" * 40)

    return "\n".join(lines)