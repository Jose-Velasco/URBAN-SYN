"""Load road embeddings and sparse graph keys for the conditional pipeline."""
import json
from pathlib import Path

import numpy as np
import torch

from ..models.diff_util import EDGE_KEY_BASE


def load_road_inputs(dataset):
    dataset = Path(dataset)
    with (dataset / "graph/road_embed.txt").open() as stream:
        count, dimension = map(int, stream.readline().split())
        values = np.loadtxt(stream, dtype=np.float32, ndmin=2)
    if values.shape != (count, dimension + 1) or not np.isfinite(values).all():
        raise ValueError("Invalid road embedding shape/values")
    if not np.array_equal(values[:, 0], np.arange(count)):
        raise ValueError("Road embedding IDs must follow the baseline's zero-based order")
    mapping = json.loads((dataset / "extra_file/new2raw_rid.json").read_text())
    if set(map(int, mapping)) != set(range(1, count + 1)):
        raise ValueError("Road embedding and dataset ID mapping differ")
    embeddings = torch.zeros((count + 1, dimension))
    embeddings[1:] = torch.from_numpy(values[:, 1:])
    edges = np.loadtxt(dataset / "graph/graph_edge.edgelist", ndmin=2)
    edges = edges[edges[:, 2] != 0]
    keys = (edges[:, 0].astype(np.int64) + 1) * EDGE_KEY_BASE + edges[:, 1].astype(np.int64) + 1
    return embeddings, torch.from_numpy(np.unique(keys))
