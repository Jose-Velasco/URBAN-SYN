"""
Optional graph-derived features for PyG road graphs.

NetworKit handles graph analytics; PyG handles structural encodings and
edge-index operations. All results are stored in separate feature groups.
"""
import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import networkx as nx
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import (
    LocalDegreeProfile,
    AddLaplacianEigenvectorPE,
    AddRandomWalkPE,
    BaseTransform,
    Compose,
)
from torch_geometric.utils import (
    degree,
    remove_self_loops,
    to_networkit,
    to_undirected,
    to_networkx
)

logger = logging.getLogger(__name__)

def _require_graph(data: Data) -> tuple[Tensor, int]:
    """Return a valid edge index and node count.

    Parameters
    ----------
    data : Data
        Input road graph.

    Returns
    -------
    tuple[Tensor, int]
        Edge indices and number of road nodes.

    Raises
    ------
    ValueError
        If graph topology is missing.
    """
    if data.edge_index is None or data.num_nodes is None:
        raise ValueError("Graph requires edge_index and num_nodes.")
    return data.edge_index, data.num_nodes


def _networkit_graph(data: Data, directed: bool = True) -> Any:
    """Convert to a NetworKit graph without changing PyG node IDs.

    Parameters
    ----------
    data : Data
        PyG road graph.
    directed : bool, default=True
        Preserve orientation or construct an undirected view.

    Returns
    -------
    networkit.Graph
        Graph retaining all road-node IDs, including isolated nodes.
    """
    edge_index, num_nodes = _require_graph(data)

    if not directed:
        # PyG's to_networkit(directed=False) keeps only edges with u <= v.
        # Symmetrize FIRST so directed one-way edges are not silently lost.
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)

    edge_index, _ = remove_self_loops(edge_index)
    return to_networkit(
        edge_index.cpu(), num_nodes=num_nodes, directed=directed
    )


def _as_column(scores: Any, num_nodes: int) -> Tensor:
    """Convert ordered NetworKit scores into a float32 node column.

    Parameters
    ----------
    scores : Any
        Scores in NetworKit's zero-based node order.
    num_nodes : int
        Expected road-node count.

    Returns
    -------
    Tensor
        Feature column with shape [num_nodes, 1].
    """
    values = torch.as_tensor(np.asarray(scores, dtype=np.float32)).reshape(-1, 1)

    if values.size(0) != num_nodes:
        raise ValueError("NetworKit scores do not match graph node count.")

    if not torch.isfinite(values).all():
        raise ValueError("Graph analytics returned non-finite scores.")

    return values


def _edge_feature_difference(data: Data, node_attr: str) -> Tensor:
    """Calculate absolute differences of node encodings per directed edge.

    Parameters
    ----------
    data : Data
        Graph with stored node encodings.
    node_attr : str
        Attribute containing an [N, F] node tensor.

    Returns
    -------
    Tensor
        Per-edge encoding differences with shape [E, F].

    Raises
    ------
    ValueError
        If the requested positional encoding is missing or misaligned.
    """
    edge_index, num_nodes = _require_graph(data)
    encoding = getattr(data, node_attr, None)

    if not isinstance(encoding, Tensor) or encoding.ndim != 2:
        raise ValueError(f"Missing or invalid node encoding: {node_attr}")

    if encoding.size(0) != num_nodes:
        raise ValueError(f"Node encoding is misaligned: {node_attr}")
    source, target = edge_index

    return torch.abs(encoding[source] - encoding[target])


class AddDegreeFeatures(BaseTransform):
    """Attach directed in-degree and out-degree as node features."""

    def __init__(self, log_scale: bool = True) -> None:
        """Configure degree scaling.

        Parameters
        ----------
        log_scale : bool, default=True
            Apply log1p to degree counts.
        """
        self.log_scale = log_scale

    def __repr__(self) -> str:
        """Describe the transform for PyG cache metadata.

        Returns
        -------
        str
            Transform parameters.
        """
        return f"{type(self).__name__}(log_scale={self.log_scale})"

    def forward(self, data: Data) -> Data:
        """Store directed degree tensor [in_degree, out_degree].

        Parameters
        ----------
        data : Data
            Road graph.

        Returns
        -------
        Data
            Graph with `degree_features` of shape [N, 2].
        """
        edge_index, num_nodes = _require_graph(data)
        source, target = edge_index
        
        features = torch.stack([
            degree(target, num_nodes=num_nodes),
            degree(source, num_nodes=num_nodes),
        ], dim=-1)
        
        data.degree_features = torch.log1p(features) if self.log_scale else features
        return data


class AddRoadRandomWalkPE(BaseTransform):
    """Attach directed random-walk return probabilities."""

    def __init__(self, walk_length: int = 3, drop_first: bool = True) -> None:
        """Configure random-walk positional encoding.

        Parameters
        ----------
        walk_length : int, default=3
            Maximum length of the random walks.
        drop_first : bool, default=True
            Drop the one-step return column if all its values are zero.
        """
        if walk_length < (2 if drop_first else 1):
            raise ValueError("walk_length is too small for these settings.")
        
        self.walk_length = walk_length
        self.drop_first = drop_first

    def __repr__(self) -> str:
        """Describe the transform for PyG cache metadata.

        Returns
        -------
        str
            Transform parameters.
        """
        return (f"{type(self).__name__}(walk_length={self.walk_length}, "
                f"drop_first={self.drop_first})")

    def forward(self, data: Data) -> Data:
        """Store random-walk PE as a separate node feature group.

        Parameters
        ----------
        data : Data
            Directed road graph.

        Returns
        -------
        Data
            Graph with `random_walk_pe` of shape [N, W] or [N, W-1].
        """
        _require_graph(data)
        encoded = AddRandomWalkPE(
            walk_length=self.walk_length, attr_name="random_walk_pe"
        )(data)
        
        # PyG's BaseTransform.__call__ returns a shallow copy.
        data.random_walk_pe = encoded.random_walk_pe
        if self.drop_first:
            # Without self-loops, a walk of length one cannot return to its start.
            if torch.count_nonzero(data.random_walk_pe[:, 0]).item():
                raise ValueError("Cannot drop nonzero one-step return probabilities.")
            data.random_walk_pe = data.random_walk_pe[:, 1:].contiguous()
        
        return data


class AddUndirectedLaplacianPE(BaseTransform):
    """Compute symmetric Laplacian PE without altering road direction."""

    def __init__(self, k: int = 16) -> None:
        """Configure Laplacian eigenvector count.

        Parameters
        ----------
        k : int, default=16
            Number of nontrivial eigenvectors.
        """
        if k < 1:
            raise ValueError("k must be positive.")
        
        self.k = k

    def __repr__(self) -> str:
        """Describe the transform for PyG cache metadata.

        Returns
        -------
        str
            Transform parameters.
        """
        return f"{type(self).__name__}(k={self.k})"

    def forward(self, data: Data) -> Data:
        """Store `laplacian_pe` using a temporary undirected graph.

        Parameters
        ----------
        data : Data
            Original directed road graph.

        Returns
        -------
        Data
            Graph with `laplacian_pe` of shape [N, k].
        """
        edge_index, num_nodes = _require_graph(data)
        if num_nodes <= self.k + 1:
            raise ValueError("Laplacian k must be less than num_nodes - 1.")
        
        undirected = Data(
            edge_index=to_undirected(edge_index, num_nodes=num_nodes),
            num_nodes=num_nodes,
        )
        
        undirected = AddLaplacianEigenvectorPE(
            k=self.k, attr_name="laplacian_pe", is_undirected=True,
        )(undirected)
        
        data.laplacian_pe = undirected.laplacian_pe.float()
        return data


class AddPageRank(BaseTransform):
    """Compute directed PageRank with NetworKit."""

    def __init__(self, alpha: float = 0.85, tolerance: float = 1e-8) -> None:
        """Configure PageRank.

        Parameters
        ----------
        alpha : float, default=0.85
            Damping probability.
        tolerance : float, default=1e-8
            Iterative convergence tolerance.
        """
        if not 0 < alpha < 1 or tolerance <= 0:
            raise ValueError("Require 0 < alpha < 1 and tolerance > 0.")
        
        self.alpha = alpha
        self.tolerance = tolerance

    def __repr__(self) -> str:
        """Describe the transform for PyG cache metadata.

        Returns
        -------
        str
            Transform parameters.
        """
        return (f"{type(self).__name__}(alpha={self.alpha}, "
                f"tolerance={self.tolerance})")

    def forward(self, data: Data) -> Data:
        """Store directed PageRank scores in `pagerank_features`.

        Parameters
        ----------
        data : Data
            Directed road graph.

        Returns
        -------
        Data
            Graph with [N, 1] PageRank tensor.
        """
        import networkit as nk

        graph = _networkit_graph(data, directed=True)
        algorithm = nk.centrality.PageRank(
            graph,
            damp=self.alpha,
            tol=self.tolerance,
            # Redistribute PageRank from dangling nodes (roads with no outgoing transitions)
            # rather than allowing their probability mass to disappear.
            # Without sink redistribution: PageRank probability can accumulate at terminal nodes without being passed onward.
            # With sink redistribution: Probability from those nodes is redistributed across the graph
            # consistent with the conventional stochastic PageRank formulation used by NetworkX.
            distributeSinks=nk.centrality.SinkHandling.DistributeSinks,
        )
        algorithm.run()
        
        data.pagerank_features = _as_column(algorithm.scores(), graph.numberOfNodes())
        return data


class AddClusteringCoefficient(BaseTransform):
    """Compute undirected local clustering coefficients with NetworKit."""

    def __init__(self, turbo: bool = False) -> None:
        """Configure triangle-counting algorithm.

        Parameters
        ----------
        turbo : bool, default=False
            Use a faster preprocessing mode at additional memory cost.
        """
        self.turbo = turbo

    def __repr__(self) -> str:
        """Describe the transform for PyG cache metadata.

        Returns
        -------
        str
            Transform parameters.
        """
        return f"{type(self).__name__}(turbo={self.turbo})"

    def forward(self, data: Data) -> Data:
        """Store undirected local clustering scores in `clustering_features`.

        Parameters
        ----------
        data : Data
            Directed road graph, viewed temporarily as undirected.

        Returns
        -------
        Data
            Graph with [N, 1] clustering coefficients.
        """
        import networkit as nk

        graph = _networkit_graph(data, directed=False)
        algorithm = nk.centrality.LocalClusteringCoefficient(graph, turbo=self.turbo)
        algorithm.run()
        
        data.clustering_features = _as_column(
            algorithm.scores(), graph.numberOfNodes()
        )
        return data


class AddApproximateBetweenness(BaseTransform):
    """Estimate directed betweenness with NetworKit's sampled algorithm."""

    def __init__(
        self,
        k: int = 100,
        seed: int = 42,
        normalized: bool = True,
        parallel: bool = False,
    ) -> None:
        """Configure approximate betweenness.

        Parameters
        ----------
        k : int, default=100
            Number of NetworKit approximation samples, not NetworkX sources.
        seed : int, default=42
            NetworKit process-global RNG seed.
        normalized : bool, default=True
            Scale approximate scores to the normalized definition.
        parallel : bool, default=False
            Enable multicore sampling at additional memory cost.
        """
        if k < 1:
            raise ValueError("k must be positive.")
        
        self.k = k
        self.seed = seed
        self.normalized = normalized
        self.parallel = parallel

    def __repr__(self) -> str:
        """Describe the transform for PyG cache metadata.

        Returns
        -------
        str
            Transform parameters.
        """
        return (f"{type(self).__name__}(k={self.k}, seed={self.seed}, "
                f"normalized={self.normalized}, parallel={self.parallel})")

    def forward(self, data: Data) -> Data:
        """Store approximate directed betweenness in `betweenness`.

        Parameters
        ----------
        data : Data
            Directed road graph.

        Returns
        -------
        Data
            Graph with [N, 1] approximate betweenness values.
        """
        import networkit as nk

        graph = _networkit_graph(data, directed=True)

        # NetworKit uses a global RNG; parallel execution can change results.
        nk.setSeed(self.seed, False)
        logger.info(
            "Estimating betweenness with %d samples on %d road nodes",
            self.k, graph.numberOfNodes(),
        )

        algorithm = nk.centrality.EstimateBetweenness(
            graph,
            nSamples=self.k,
            normalized=self.normalized,
            parallel=self.parallel,
        )
        algorithm.run()

        data.betweenness = _as_column(algorithm.scores(), graph.numberOfNodes())
        return data


class AddConstraintStructuralHoles(BaseTransform):
    """Compute Burt's structural-hole constraint using NetworkX."""

    def __repr__(self) -> str:
        """Describe the transform for PyG cache metadata.

        Returns
        -------
        str
            Transform parameters.
        """
        return f"{type(self).__name__}()"

    def forward(self, data: Data) -> Data:
        """Attach structural-hole constraint as a node feature.

        Parameters
        ----------
        data : Data
            PyG road graph.

        Returns
        -------
        Data
            Graph with constraint_features of shape [num_nodes, 1].
        """
        if data.edge_index is None or data.num_nodes is None:
            raise ValueError("Graph requires edge_index and num_nodes.")

        # Preserve directed road transitions and include isolated nodes.
        graph_nx = to_networkx(data, to_undirected=False)

        constraint = nx.constraint(graph_nx)

        # NetworkX returns NaN for isolated nodes.
        values = np.array(
            [constraint[i] for i in range(data.num_nodes)],
            dtype=np.float32,
        )

        # Use a neutral value for undefined constraints.
        values = np.nan_to_num(values, nan=0.0)

        data.constraint_features = torch.from_numpy(
            values.reshape(-1, 1)
        )


        return data


class AddAdamicAdar(BaseTransform):
    """Score existing road transitions by undirected neighbor overlap."""

    def __init__(self, batch_size: int = 25_000) -> None:
        """Configure batch processing for NetworKit's link-pair API.

        Parameters
        ----------
        batch_size : int, default=25000
            Maximum number of edge pairs scored in one batch.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        
        self.batch_size = batch_size

    def __repr__(self) -> str:
        """Describe the transform for PyG cache metadata.

        Returns
        -------
        str
            Transform parameters.
        """
        return f"{type(self).__name__}(batch_size={self.batch_size})"

    def forward(self, data: Data) -> Data:
        """Store Adamic-Adar scores aligned to the original directed edges.

        Parameters
        ----------
        data : Data
            Road graph with directed edge_index.

        Returns
        -------
        Data
            Graph with `adamic_adar_edge_features` of shape [E, 1].
        """
        import networkit as nk

        edge_index, _ = _require_graph(data)
        graph = _networkit_graph(data, directed=False)
        predictor = nk.linkprediction.AdamicAdarIndex(graph)
        
        pairs = edge_index.t().cpu().tolist()
        
        values: list[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = [tuple(edge) for edge in pairs[start:start + self.batch_size]]
            # NetworKit sorts predictions by node pair; restore PyG edge order.
            predictions = dict(predictor.runOn(batch))
            values.extend(predictions[pair] for pair in batch)
        
        data.adamic_adar_edge_features = torch.tensor(
            values, dtype=torch.float32
        ).reshape(-1, 1)
        
        return data


class AddLocalDegreeProfile(BaseTransform):
    """Compute local degree statistics without modifying existing node features."""

    def __init__(self, log_scale: bool = True) -> None:
        """Configure local degree feature scaling.

        Parameters
        ----------
        log_scale : bool, default=True
            Apply log1p to the nonnegative degree statistics.
        """
        self.log_scale = log_scale

    def __repr__(self) -> str:
        """Describe the transform for PyG caching.

        Returns
        -------
        str
            Transform configuration.
        """
        return f"{type(self).__name__}(log_scale={self.log_scale})"

    def forward(self, data: Data) -> Data:
        """Attach local degree statistics as a separate node feature group.

        Parameters
        ----------
        data : Data
            Directed PyG road graph.

        Returns
        -------
        Data
            Graph with `local_degree_profile` of shape [N, 5].
        """
        edge_index, num_nodes = _require_graph(data)

        # Use a temporary graph so PyG does not append features to data.x.
        temp = Data(edge_index=edge_index, num_nodes=num_nodes)
        profile = LocalDegreeProfile()(temp).x

        if self.log_scale:
            profile = torch.log1p(profile.clamp_min(0))

        data.local_degree_profile = profile
        return data

# These are edge features
class AddRandomWalkRelativeDistance(BaseTransform):
    """Store differences between cached node random-walk encodings per edge."""

    def __repr__(self) -> str:
        """Describe the transform for PyG cache metadata.

        Returns
        -------
        str
            Transform parameters.
        """
        return f"{type(self).__name__}()"

    def forward(self, data: Data) -> Data:
        """Compute absolute endpoint differences from `random_walk_pe`.

        Parameters
        ----------
        data : Data
            Graph with precomputed `random_walk_pe`.

        Returns
        -------
        Data
            Graph with `random_walk_edge_features` of shape [E, W].
        """
        data.random_walk_edge_features = _edge_feature_difference(
            data, "random_walk_pe"
        )
        return data


class AddLaplacianEigenvectorRelativeDistance(BaseTransform):
    """Store absolute differences of Laplacian eigenvectors per edge."""

    def __repr__(self) -> str:
        """Describe the transform for PyG cache metadata.

        Returns
        -------
        str
            Transform parameters.
        """
        return f"{type(self).__name__}()"

    def forward(self, data: Data) -> Data:
        """Compute endpoint differences from cached `laplacian_pe`.

        Parameters
        ----------
        data : Data
            Graph with precomputed Laplacian positional encodings.

        Returns
        -------
        Data
            Graph with `laplacian_edge_features` of shape [E, k].
        """
        data.laplacian_edge_features = _edge_feature_difference(
            data, "laplacian_pe"
        )
        return data


def make_graph_pre_transform(config: Mapping[str, Any] | None) -> Compose | None:
    """Build optional graph-mining transforms from explicit settings.

    Parameters
    ----------
    config : Mapping[str, Any] | None
        Transform names mapped to options, including `enabled`.

    Returns
    -------
    Compose | None
        Ordered PyG preprocessing transforms, or None if none enabled.

    Raises
    ------
    ValueError
        If a transform or setting is unsupported.
    """
    if not config:
        return None

    factories = {
        "degree_features": (AddDegreeFeatures, {"log_scale"}),
        "local_degree_profile": (AddLocalDegreeProfile, {"log_scale"}),
        "random_walk_pe": (AddRoadRandomWalkPE, {"walk_length", "drop_first"}),
        "laplacian_pe": (AddUndirectedLaplacianPE, {"k"}),
        "betweenness": (
            AddApproximateBetweenness,
            {"k", "seed", "normalized", "parallel"},
        ),
        "pagerank_features": (AddPageRank, {"alpha", "tolerance"}),
        "clustering_features": (AddClusteringCoefficient, {"turbo"}),
        "constraint_features": (AddConstraintStructuralHoles, set()),
        "adamic_adar_edge_features": (AddAdamicAdar, {"batch_size"}),
        "random_walk_edge_features": (AddRandomWalkRelativeDistance, set()),
        "laplacian_edge_features": (
            AddLaplacianEigenvectorRelativeDistance,
            set(),
        ),
    }
    
    unknown = set(config) - set(factories)
    if unknown:
        raise ValueError(f"Unknown graph transforms: {sorted(unknown)}")
    
    transforms: list[BaseTransform] = []
    for name, options in config.items():
        if not isinstance(options, Mapping):
            raise ValueError(f"Transform '{name}' must be a mapping.")
        
        if not options.get("enabled", False):
            continue
        
        factory, accepted = factories[name]
        kwargs = {key: value for key, value in options.items() if key != "enabled"}
        
        unexpected = set(kwargs) - accepted
        if unexpected:
            raise ValueError(f"Unknown {name} options: {sorted(unexpected)}")
        
        transforms.append(factory(**kwargs))
    
    return Compose(transforms) if transforms else None
