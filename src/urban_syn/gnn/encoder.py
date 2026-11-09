"""Directed, edge-aware PyG road encoder for trajectory-generation models."""
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn.resolver import activation_resolver

from .config import GNNSettings
from .factory import ConvFactory

def _build_norm(name: str | None, dim: int) -> nn.Module:
    """Create normalization for intermediate convolution outputs."""
    if name in {None, "none"}:
        return nn.Identity()
    if name in {"layer", "layer_norm"}:
        return nn.LayerNorm(dim)
    if name in {"batch", "batch_norm"}:
        return nn.BatchNorm1d(dim)
    raise ValueError(f"Unsupported GNN normalization: {name!r}")

class RoadGNNEncoder(nn.Module):
    """Learn one embedding per directed road segment.

    The injected factory controls convolution construction. The encoder manages
    shared depth, feature widths, normalization, dropout, and residuals.
    """

    def __init__(
        self,
        input_dim: int,
        edge_dim: int | None,
        settings: GNNSettings,
        conv_factory: ConvFactory,
    ) -> None:
        """Create graph layers from configured widths and a convolution factory.

        Parameters
        ----------
        input_dim : int
            Width of selected road-node features.
        edge_dim : int | None
            Width of selected physical-edge features, or None if unused.
        settings : GNNSettings
            Validated model architecture settings.
        conv_factory : ConvFactory
            Callable constructing each GNN convolution layer.
        """
        super().__init__()
        if input_dim < 1 or (edge_dim is not None and edge_dim < 1):
            raise ValueError("Input and edge dimensions must be positive.")

        self.input_dim = input_dim
        self.edge_dim = edge_dim
        self.settings = settings
        self.activation = activation_resolver(settings.activation)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residuals = nn.ModuleList()

        for layer_idx in range(settings.num_layers):
            is_final = layer_idx == settings.num_layers - 1
            in_dim = input_dim if layer_idx == 0 else settings.hidden_dim
            out_dim = settings.output_dim if is_final else settings.hidden_dim
            self.convs.append(conv_factory(in_dim, out_dim, edge_dim, is_final))

            # Keep final road embeddings unrestricted for diffusion integration.
            self.norms.append(nn.Identity() if is_final else _build_norm(settings.normalization, out_dim))
            
            self.residuals.append(
                nn.Linear(in_dim, out_dim, bias=False)
                if settings.residual and in_dim != out_dim
                else nn.Identity()
            )

    @classmethod
    def from_graph(
        cls,
        graph: Data,
        settings: GNNSettings,
        conv_factory: ConvFactory,
    ) -> "RoadGNNEncoder":
        """Infer input dimensions from a graph with selected features.

        Parameters
        ----------
        graph : Data
            Graph after selecting ``x`` and optional ``edge_attr``.
        settings : GNNSettings
            Validated architecture settings.
        conv_factory : ConvFactory
            Layer constructor chosen by the caller.

        Returns
        -------
        RoadGNNEncoder
            Encoder ready for training.
        """
        if graph.x is None or graph.x.ndim != 2:
            raise ValueError("Graph must have a two-dimensional x tensor.")
        
        edge_attr = getattr(graph, "edge_attr", None)
        if edge_attr is not None and edge_attr.ndim != 2:
            raise ValueError("graph.edge_attr must be a two-dimensional tensor.")
        
        return cls(graph.x.size(-1), None if edge_attr is None else edge_attr.size(-1), settings, conv_factory)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
    ) -> Tensor:
        """Compute embeddings aligned with PyG road-node indices.

        Parameters
        ----------
        x : Tensor
            Node features [N, input_dim].
        edge_index : Tensor
            Directed physical transitions [2, E].
        edge_attr : Tensor | None
            Optional physical-edge features [E, edge_dim].

        Returns
        -------
        Tensor
            Road embeddings [N, output_dim].
        """
        if x.ndim != 2 or x.size(1) != self.input_dim:
            raise ValueError(f"Expected node features [N, {self.input_dim}].")
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, E].")
        if self.edge_dim is None:
            if edge_attr is not None:
                raise ValueError("Encoder was initialized without edge features.")
        elif edge_attr is None or edge_attr.ndim != 2 or edge_attr.shape != (edge_index.size(1), self.edge_dim):
            raise ValueError(f"Expected edge_attr [E, {self.edge_dim}].")

        for idx, (conv, norm, residual) in enumerate(zip(self.convs, self.norms, self.residuals)):
            previous = x
            x = conv(x, edge_index, edge_attr)
            if idx < len(self.convs) - 1:
                x = self.activation(norm(x))
                x = F.dropout(x, p=self.settings.dropout, training=self.training)
            if self.settings.residual:
                x = x + residual(previous)
        return x
