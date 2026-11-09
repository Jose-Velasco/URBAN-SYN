"""Count training cpath transitions on a canonical PyG road graph."""
import logging
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from itertools import pairwise
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from .smoothing import AdditiveSmoothing, TransitionSmoother

logger = logging.getLogger(__name__)

def parse_road_path(value: object) -> list[int]:
    """Parse an FMM road-ID sequence without removing repeated road IDs.

    Parameters
    ----------
    value : object
        Comma-separated road IDs, optionally enclosed in brackets.

    Returns
    -------
    list[int]
        Road IDs in order, or an empty list for a missing path.

    Raises
    ------
    ValueError
        If a nonempty path is malformed.
    """
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text == "[]":
        return []

    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    if not re.fullmatch(r"\d+(?:\s*,\s*\d+)*", text):
        raise ValueError(f"Invalid FMM road path: {value!r}")
    return [int(part.strip()) for part in text.split(",")]


def _road_id_lookup(data: Data) -> dict[int, int]:
    """Validate graph identifiers and map canonical IDs to PyG node indices.

    Parameters
    ----------
    data : Data
        Physical road graph with edge_index and edge_id.

    Returns
    -------
    dict[int, int]
        Mapping from canonical road ID to zero-based node index.

    Raises
    ------
    ValueError
        If the graph topology or road identifiers are invalid.
    """
    if data.edge_index is None or data.num_nodes is None:
        raise ValueError("Graph requires edge_index and num_nodes.")
    edge_ids = getattr(data, "edge_id", None)
    if not isinstance(edge_ids, Tensor) or edge_ids.ndim != 1:
        raise ValueError("Graph requires one-dimensional canonical edge_id.")
    if edge_ids.numel() != data.num_nodes:
        raise ValueError("edge_id must have one entry per graph node.")
    if data.edge_index.ndim != 2 or data.edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, E].")
    if data.num_edges == 0:
        raise ValueError("Graph has no physical transitions to count.")

    lookup = {int(road_id): idx for idx, road_id in enumerate(edge_ids.cpu().tolist())}
    if len(lookup) != data.num_nodes:
        raise ValueError("Canonical edge_id values must be unique.")
    return lookup


def _align_counts(
    data: Data,
    observed: Counter[tuple[int, int]],
    node_by_id: dict[int, int],
) -> tuple[Tensor, Tensor, Tensor, list[tuple[int, int]]]:
    """Map observed canonical road pairs to existing physical edges.

    Instead of comparing pairs of integers repeatedly, we compare single integer keys.
    torch.searchsorted() efficiently finds candidate matches in the sorted array.
    The code then checks whether those keys actually exist and maps the counts back to the original edge_index order.
    That's why edge_order matters and why we cannot simply return frequencies in sorted-key order.

    Parameters
    ----------
    data : Data
        Physical road graph.
    observed : Counter[tuple[int, int]]
        Counts of consecutive road-ID pairs.
    node_by_id : dict[int, int]
        Canonical road IDs mapped to PyG indices.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, list[tuple[int, int]]]
        Counts aligned with edge_index, matched-pair mask, observed
        frequencies, and observed road pairs.

    Raises
    ------
    ValueError
        If the paths reference unknown roads or topology has duplicates.
    """
    # FMM cpath uses canonical road IDs, which may differ from PyG node indices.
    # All referenced roads must exist in the graph before we can align pairs.
    unknown = {
        road_id for pair in observed for road_id in pair if road_id not in node_by_id
    }
    if unknown:
        raise ValueError(
            f"FMM paths contain {len(unknown)} unknown canonical road IDs "
            f"(sample: {sorted(unknown)[:5]}). Check the road-network version."
        )

    num_nodes = data.num_nodes
    source, target = data.edge_index.cpu().long()

    # Encode each directed physical edge (u, v) as a unique integer:
    #     key = u * num_nodes + v
    # This lets us match road pairs using fast tensor operations instead
    # of building a dense adjacency matrix or looping over graph edges.
    # these are the physical road that connect thus they are the edges of the road network
    # not the observed transitions
    physical_keys = source * num_nodes + target

    # Sort the encoded edges for binary search, but retain their original
    # positions so the final counts match the ordering of data.edge_index.
    # where the sorted_keys are the sorted values and edge_order are the indices of the elements in the original input tensor.
    # sorted_keys, edge_order are the "actual" connected roads in the road network
    sorted_keys, edge_order = physical_keys.sort()
    if torch.unique_consecutive(sorted_keys).numel() != sorted_keys.numel():
        raise ValueError("Physical edge_index contains duplicate directed edges.")

    # Convert observed canonical road pairs into the same integer encoding.
    # "so the roads that were actually taken/drive/used based on the dataset"
    # but we encode them to the nodes in the pyg graph then map then into a key to speeds up computation of match and counts
    # Example: (road_id_A, road_id_B) -> (node_idx_A, node_idx_B) -> key.
    pairs = list(observed)
    # keys is the road pairs we saw in the dataset and is most likely a subset of physical_keys
    keys = torch.tensor(
        [node_by_id[a] * num_nodes + node_by_id[b] for a, b in pairs],
        dtype=torch.long,
    )
    # keys and frequencies should be the same dim since for each observed road pair we have its frequency
    frequencies = torch.tensor([observed[pair] for pair in pairs], dtype=torch.float64)

    # Binary search avoids constructing a dense num_nodes x num_nodes matrix.
    # Binary-search the physical edges (in the road network) for each observed transition.
    # searchsorted returns the insertion position, which does NOT guarantee
    # an exact match, so we must verify the encoded keys afterward.
    """sorted_keys: are basically the roads in the road network"""
    """keys: are basically the observed in the dataset road pairs that were traversed/transitioned"""
    # "kinda filtering dataset observed road pair to compile with actual physically possible roads(physically connected in the road network)"
    positions = torch.searchsorted(sorted_keys, keys)

    # An insertion position may equal the array length when no larger
    # physical key exists. Clamp only for safe indexing, then compare keys
    # to distinguish actual physical connections from missing transitions.
    matched = positions < sorted_keys.numel() # if larger position then we know dataset observed road pairs was not in the road network (simple filter to remove not possible road connections)
    # Clamp insertion positions to valid indices, then verify exact matches
    # since searchsorted may return a position where the key doesn't exist.
    # sorted_keys contains physical graph edges. keys contains observed transitions.
    # Example: sorted_keys=[2, 5, 10], keys=[7] -> insertion index 2,
    # but 10 != 7, so the observed transition is not a physical edge.
    matched &= sorted_keys[positions.clamp(max=sorted_keys.numel() - 1)] == keys

    # Accumulate observed frequencies onto existing physical edges only.
    # edge_order maps positions in the sorted array back to edge_index order.
    # Unmatched transitions are excluded; no new graph edges are created.
    counts = torch.zeros(data.num_edges, dtype=torch.float64)
    # frequencies[matched]: only physically connected roads frequencies we want to count 
    counts.index_add_(0, edge_order[positions[matched]], frequencies[matched])
    return counts, matched, frequencies, pairs


def count_physical_transitions(
    data: Data, paths: Iterable[Sequence[int]]
) -> Tensor:
    """Count training road pairs on the existing physical topology.

    Parameters
    ----------
    data : Data
        PyG graph with canonical edge_id and directed edge_index.
    paths : Iterable[Sequence[int]]
        Training-only canonical road-ID sequences (typically FMM cpath).

    Returns
    -------
    Tensor
        Float64 counts of shape [E], aligned with data.edge_index.

    Raises
    ------
    ValueError
        If road IDs are incompatible or no physical transitions match.
    """
    node_by_id = _road_id_lookup(data)
    observed: Counter[tuple[int, int]] = Counter()
    num_paths = 0
    for path in paths:
        num_paths += 1
        observed.update(pairwise(path))
    
    if not observed:
        raise ValueError("Input paths contain no road transitions.")

    counts, matched, frequencies, pairs = _align_counts(data, observed, node_by_id)
    if not matched.any():
        raise ValueError("No training transitions match physical graph connections.")

    # Count occurrences of unmatched self-transitions (A -> A),
    # where the same road appears consecutively but no self-loop exists in the graph.
    missed_self = sum(
        observed[pair]
        for pair, is_matched in zip(pairs, matched.tolist())
        if not is_matched and pair[0] == pair[1]
    )
    
    logger.info(
        "Training cpath transitions: %d paths, %d/%d unique pairs aligned, "
        "%d/%d traversals aligned; %d unmatched self-transitions",
        num_paths, int(matched.sum()), len(observed),
        int(frequencies[matched].sum()), int(frequencies.sum()), missed_self,
    )
    
    return counts


class AddCPathTransitionFeatures(BaseTransform):
    """Attach counts and smoothed probabilities from prefiltered training cpaths."""

    def __init__(
        self, match_csv: str | Path, smoother: TransitionSmoother | None = None
    ) -> None:
        """Configure the training paths and smoothing strategy.

        Parameters
        ----------
        match_csv : str | Path
            Comma-delimited training FMM match CSV produced by the shared splitter.
        smoother : TransitionSmoother | None, optional
            Counts-to-probabilities strategy; defaults to AdditiveSmoothing().
        """
        self.match_csv = Path(match_csv)
        self.smoother = smoother if smoother is not None else AdditiveSmoothing()

    def forward(self, data: Data) -> Data:
        """Read training cpaths, count transitions, and attach features.

        Parameters
        ----------
        data : Data
            Physical road graph with shared road-network edge_id values
            and directed edge_index connectivity.

        Returns
        -------
        Data
            Graph with per-edge counts and smoothed probabilities.
        """
        # FMM CSV uses semicolons because path values contain commas.
        # Raw FMM outputs use semicolons, while post-FMM split CSVs use commas.
        matches = pd.read_csv(self.match_csv, usecols=["cpath"])

        paths = (parse_road_path(value) for value in matches["cpath"])
        counts = count_physical_transitions(data, paths)
        probs = self.smoother.smooth(counts, data.edge_index.cpu(), data.num_nodes)

        # Counts and probabilities follow exactly the ordering of edge_index.
        data.edge_transition_count = counts.to(
            dtype=torch.float32, device=data.edge_index.device
        ).unsqueeze(1)
        
        data.edge_transition_prob = probs.to(
            dtype=torch.float32, device=data.edge_index.device
        ).unsqueeze(1)
        
        return data

    def __repr__(self) -> str:
        """Describe transform settings for PyG cache metadata.

        Returns
        -------
        str
            CSV path and selected smoothing strategy.
        """
        return (
            f"{type(self).__name__}(match_csv={str(self.match_csv)!r}, "
            f"smoother={self.smoother!r})"
        )
