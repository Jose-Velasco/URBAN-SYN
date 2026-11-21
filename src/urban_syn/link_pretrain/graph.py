"""Cached road graph, directed link splits, and topology-only lookups.

PyG owns graph representation and link splitting. This module adds the strict
split checks and full-graph positive exclusion needed for our pretraining task.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .config import GraphSplitSettings, ResolvedLinkPretrainConfig

if TYPE_CHECKING:
    from torch_geometric.data import Data


@dataclass(frozen=True)
class SplitSummary:
    """Counts used to inspect a seeded directed link split."""

    seed: int
    full_positive_count: int
    message_edge_count: int
    train_positive_count: int
    val_positive_count: int
    test_positive_count: int
    eligible_train_seeds: int
    isolated_message_nodes: int


def graph_paths(config: ResolvedLinkPretrainConfig) -> tuple[Path, tuple[Path, Path]]:
    """Resolve the road-network source and graph-cache paths.

    Args:
        config: Validated configuration with repository-root and graph paths.

    Returns:
        The source GeoPackage path and a pair of paths for the cached PyG
        graph and feature preprocessor, respectively.
    """
    dataset = config.graph_settings["dataset"]
    network = Path(dataset["network_path"]).expanduser()
    cache = Path(dataset["root"]).expanduser()
    root = config.project_root
    network = (network if network.is_absolute() else root / network).resolve()
    cache = (cache if cache.is_absolute() else root / cache).resolve()
    processed = cache / "processed"
    return network, (processed / "road_graph.pt", processed / "preprocessor.joblib")


def validate_graph_cache(config: ResolvedLinkPretrainConfig) -> None:
    """Verify that the source road network and graph cache already exist.

    Args:
        config: Validated configuration identifying the network and cache.

    Raises:
        FileNotFoundError: If the network, graph, or preprocessor is missing.
    """
    network, cached = graph_paths(config)
    missing = [path for path in (network, *cached) if not path.is_file()]
    if missing:
        details = "\n  - ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Required road-network/cache files are missing:\n  - {details}\n"
            "Build the shared road graph using the existing graph pipeline first."
        )


def load_pretraining_graph(config: ResolvedLinkPretrainConfig) -> Data:
    """Load the cached road graph with its configured features on CPU.

    Args:
        config: Validated pretraining and graph settings.

    Returns:
        PyG ``Data`` with selected node features and optional edge features,
        using the same road-node ordering as the existing graph pipeline.
    """
    from urban_syn.experiment.factories import load_experiment_graph
    from urban_syn.gnn.config import GNNSettings

    GNNSettings.model_validate(config.graph_settings["model"]["gnn"])
    validate_graph_cache(config)
    return load_experiment_graph(config, torch.device("cpu"))


def directed_edge_keys(edges: Tensor, num_nodes: int) -> Tensor:
    """Encode directed edges as unique integer keys for vectorized lookups.

    Args:
        edges: CPU ``torch.long`` edge-index tensor of shape ``[2, E]``,
            with source indices in row 0 and destination indices in row 1.
        num_nodes: Total number of nodes in the indexed graph.

    Returns:
        One ``int64`` key per directed edge, ``source * num_nodes + target``.
        Reversing an edge produces a different key.

    Raises:
        ValueError: If graph dimensions, indices, dtype, or device are invalid.
    """
    if num_nodes < 1 or num_nodes * num_nodes > torch.iinfo(torch.int64).max:
        raise ValueError("Graph size cannot be encoded safely as int64 edge keys.")
    if edges.dtype != torch.long or edges.ndim != 2 or edges.size(0) != 2:
        raise ValueError("Directed edges must be torch.long with shape [2, E].")
    if edges.device.type != "cpu":
        raise ValueError("Graph topology must remain on CPU during preparation.")
    if edges.numel() and (edges.min().item() < 0 or edges.max().item() >= num_nodes):
        raise ValueError("Directed edge contains an out-of-range PyG node index.")
    return edges[0] * num_nodes + edges[1]


def _assert_unique(keys: Tensor, description: str) -> None:
    """Reject repeated directed edges instead of silently losing split identity."""
    if keys.numel() != torch.unique(keys).numel():
        raise ValueError(f"{description} contains duplicate directed edges.")


def validate_road_graph(graph: Data) -> None:
    """Validate road-node identities, topology, and selected feature alignment.

    Args:
        graph: CPU PyG road graph with sorted physical ``edge_id`` values,
            node features, and optional edge attributes.

    Raises:
        ValueError: If indices, road IDs, or feature dimensions violate the
            graph builder's expected layout.
    """
    nodes = int(graph.num_nodes)
    edge_index = graph.edge_index
    if (
        nodes < 1
        or edge_index is None
        or edge_index.ndim != 2
        or edge_index.size(0) != 2
        or edge_index.size(1) < 1
    ):
        raise ValueError("Road graph requires nonempty edge_index with shape [2, E].")
    keys = directed_edge_keys(edge_index, nodes)
    _assert_unique(keys, "Original road graph")

    if graph.x is None or graph.x.ndim != 2 or graph.x.size(0) != nodes:
        raise ValueError("Selected node features must have shape [N, F].")
    if graph.x.device.type != "cpu":
        raise ValueError("Graph node features must remain on CPU during preparation.")
    ids = getattr(graph, "edge_id", None)
    if ids is None or ids.dtype != torch.long or ids.shape != (nodes,):
        raise ValueError("Road graph requires long edge_id values aligned with PyG nodes.")
    if ids.device.type != "cpu" or ids.unique().numel() != nodes:
        raise ValueError("Road edge_id values must be unique CPU identifiers.")
    # The builder defines node order by increasing physical road ID. It does
    # not equate an edge_id with its zero-based PyG node index.
    if nodes > 1 and not bool(torch.all(ids[1:] > ids[:-1])):
        raise ValueError("Cached road edge_id values must follow builder-sorted node order.")
    edge_attr = getattr(graph, "edge_attr", None)
    if edge_attr is not None and (
        edge_attr.ndim != 2
        or edge_attr.size(0) != edge_index.size(1)
        or edge_attr.device.type != "cpu"
    ):
        raise ValueError("Selected edge_attr must align one-to-one with edge_index.")


def road_ids_to_node_indices(graph: Data, road_ids: Tensor) -> Tensor:
    """Map physical road IDs to zero-based PyG node indices.

    Args:
        graph: Road graph whose ``edge_id`` tensor is sorted and unique.
        road_ids: Tensor of physical ``edge_id`` values in any shape.

    Returns:
        CPU ``torch.long`` indices with the same shape as ``road_ids``.
        These are graph-node indices, not diffusion token IDs.

    Raises:
        ValueError: If any requested road ID is absent from the graph.
    """
    road_ids = road_ids.to(dtype=torch.long, device="cpu")
    if road_ids.numel() == 0:
        return road_ids.clone()
    ordered = graph.edge_id
    positions = torch.searchsorted(ordered, road_ids)
    # Missing IDs may be inserted after the last element. Clamp lookup
    # positions before checking equality to avoid out-of-bounds indexing.
    safe = positions.clamp(max=ordered.numel() - 1)
    if not bool(torch.all((positions < ordered.numel()) & (ordered[safe] == road_ids))):
        raise ValueError("Unknown road edge_id.")
    return positions


class FullPositiveIndex:
    """Read-only ordered-edge membership lookup from the unsplit road graph.

    Use this index only to reject false negatives, including held-out
    positives. Never provide its full topology to the pretraining encoder.
    """

    def __init__(self, edge_index: Tensor, num_nodes: int) -> None:
        """Index all original positive directed edges.

        Args:
            edge_index: Unsplit CPU edge-index tensor of shape ``[2, E]``.
            num_nodes: Number of roads represented by the graph.
        """
        self.num_nodes = num_nodes
        keys = directed_edge_keys(edge_index, num_nodes)
        _assert_unique(keys, "Original road graph")
        self._sorted_keys = keys.sort().values

    def contains(self, sources: Tensor, destinations: Tensor) -> Tensor:
        """Check whether each directed source/destination pair is an edge.

        Args:
            sources: Source node indices, broadcastable with ``destinations``.
            destinations: Destination node indices in zero-based PyG space.

        Returns:
            Boolean CPU tensor of the broadcast shape. A true entry means
            the exact directed pair exists in the full original graph.
        """
        sources = sources.to(dtype=torch.long, device="cpu")
        destinations = destinations.to(dtype=torch.long, device="cpu")
        sources, destinations = torch.broadcast_tensors(sources, destinations)
        pairs = torch.stack((sources.reshape(-1), destinations.reshape(-1)))
        query = directed_edge_keys(pairs, self.num_nodes)
        positions = torch.searchsorted(self._sorted_keys, query)
        # Search may return len(keys) for missing edges; clamp before
        # indexing while keeping the original bound check for correctness.
        safe = positions.clamp(max=self._sorted_keys.numel() - 1)
        found = (positions < self._sorted_keys.numel()) & (
            self._sorted_keys[safe] == query
        )
        return found.reshape(sources.shape)


def _edge_rows(graph: Data, selected_edges: Tensor) -> Tensor:
    """Find original edge-attribute rows for a set of directed links.

    Args:
        graph: Unsplit road graph containing the reference edge order.
        selected_edges: Directed links of shape ``[2, E_selected]``.

    Returns:
        Original edge-row indices in the same order as ``selected_edges``.

    Raises:
        ValueError: If a selected edge is absent from the full graph.
    """
    nodes = graph.num_nodes
    original = directed_edge_keys(graph.edge_index, nodes)
    query = directed_edge_keys(selected_edges, nodes)
    ordered, permutation = original.sort()
    positions = torch.searchsorted(ordered, query)
    safe = positions.clamp(max=ordered.numel() - 1)
    if not bool(torch.all((positions < ordered.numel()) & (ordered[safe] == query))):
        raise ValueError("Split contains a directed edge absent from the original graph.")
    return permutation[safe]


def validate_link_split(graph: Data, train: Data, val: Data, test: Data) -> None:
    """Verify split isolation and exact edge-feature alignment.

    Args:
        graph: Full unsplit road graph, used only as the reference.
        train: Training message-passing graph and training positive labels.
        val: Validation labels paired with training-only message edges.
        test: Test labels paired with training-only message edges.

    Raises:
        ValueError: If supervision overlaps, edges are missing or duplicated,
            message-passing graphs differ, or edge features are misaligned.
    """
    full = directed_edge_keys(graph.edge_index, graph.num_nodes)
    message = directed_edge_keys(train.edge_index, graph.num_nodes)
    labeled = [
        directed_edge_keys(item.edge_label_index, graph.num_nodes)
        for item in (train, val, test)
    ]
    for name, keys in zip(("message", "train", "validation", "test"), (message, *labeled)):
        _assert_unique(keys, name)
    for item in (train, val, test):
        if not torch.equal(item.edge_index, train.edge_index):
            raise ValueError("Every evaluation phase must use training-only message edges.")
        if not torch.equal(item.edge_label, torch.ones_like(item.edge_label)):
            raise ValueError("Link-split labels must contain positive links only.")
        if item.num_nodes != graph.num_nodes or not torch.equal(item.edge_id, graph.edge_id):
            raise ValueError("Graph node ordering changed during link splitting.")
    # Every original directed edge must appear exactly once, either in
    # message passing or as one of the three positive-label partitions.
    all_keys = torch.cat((message, *labeled))
    same_count = all_keys.numel() == full.numel()
    same_edges = torch.equal(all_keys.sort().values, full.sort().values)
    if not (same_count and same_edges):
        raise ValueError(
            "Split message/supervision links must exactly partition all original edges."
        )

    if getattr(graph, "edge_attr", None) is not None:
        expected = graph.edge_attr[_edge_rows(graph, train.edge_index)]
        for item in (train, val, test):
            # Compare raw bytes to detect any reordered or modified feature
            # row, including values that numeric equality can treat as equal.
            if (item.edge_attr is None
                or item.edge_attr.dtype != expected.dtype
                or not torch.equal(
                    item.edge_attr.contiguous().view(torch.uint8),
                    expected.contiguous().view(torch.uint8)
                )):
                raise ValueError("Split edge_attr rows do not match directed message edges.")
    elif any(getattr(item, "edge_attr", None) is not None for item in (train, val, test)):
        raise ValueError("Unexpected edge features introduced by link splitting.")


def split_directed_graph(
    graph: Data, settings: GraphSplitSettings, seed: int
) -> tuple[Data, Data, Data]:
    """Split directed links while isolating held-out supervision edges.

    Args:
        graph: Validated CPU road graph containing all positive edges.
        settings: Train/validation/test fractions and the share of training
            positives reserved for disjoint supervision.
        seed: Random seed for PyG's ``RandomLinkSplit`` operation.

    Returns:
        ``(train, val, test)`` PyG ``Data`` objects with separate positive
        ``edge_label_index`` tensors. All three expose only training
        message-passing edges and the corresponding ``edge_attr`` rows.

    Raises:
        ValueError: If the split is empty, leaks held-out edges into message
            passing, or fails topology/feature validation.
    """
    from torch_geometric.transforms import RandomLinkSplit

    validate_road_graph(graph)
    split = RandomLinkSplit(
        num_val=settings.val_fraction,
        num_test=settings.test_fraction,
        is_undirected=False,
        add_negative_train_samples=False,
        neg_sampling_ratio=0.0,
        disjoint_train_ratio=settings.disjoint_train_ratio
    )
    # Seed PyG's global torch.randperm locally, restoring the caller's RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        train, val, test = split(graph)

    if any(item.edge_label_index.size(1) == 0 for item in (train, val, test)):
        raise ValueError(
            "Link split yielded an empty train, validation, or test positive partition."
        )

    # PyG's ordinary val/test edge_index include extra context by default.
    # Strict transductive pretraining keeps all three on the train message graph.
    for item in (val, test):
        item.edge_index = train.edge_index
        item.edge_attr = getattr(train, "edge_attr", None)
    validate_link_split(graph, train, val, test)
    return train, val, test


def summarize_split(
    graph: Data, splits: tuple[Data, Data, Data], seed: int
) -> SplitSummary:
    """Summarize split sizes and message-graph coverage.

    Args:
        graph: Full road graph used as the split reference.
        splits: Validated ``(train, val, test)`` PyG graphs.
        seed: Seed used to produce the split.

    Returns:
        ``SplitSummary`` with positive-label counts, eligible sources, and
        the number of nodes isolated by the reduced message graph.
    """
    train, val, test = splits
    touched = torch.bincount(train.edge_index.reshape(-1), minlength=graph.num_nodes)
    return SplitSummary(
        seed=seed,
        full_positive_count=graph.num_edges,
        message_edge_count=train.num_edges,
        train_positive_count=train.edge_label_index.size(1),
        val_positive_count=val.edge_label_index.size(1),
        test_positive_count=test.edge_label_index.size(1),
        eligible_train_seeds=int(torch.unique(train.edge_label_index[0]).numel()),
        isolated_message_nodes=int((touched == 0).sum().item())
    )


def prepare_link_split(
    graph: Data, settings: GraphSplitSettings, seed: int
) -> tuple[tuple[Data, Data, Data], SplitSummary]:
    """Prepare a seeded directed split and its summary without saving tensors.

    Args:
        graph: Cached road graph with selected features.
        settings: Directed-link split fractions and supervision ratio.
        seed: Seed used for reproducible split generation.

    Returns:
        A ``(train, val, test)`` graph tuple and a ``SplitSummary``.
    """
    splits = split_directed_graph(graph, settings, seed)
    return splits, summarize_split(graph, splits, seed)
