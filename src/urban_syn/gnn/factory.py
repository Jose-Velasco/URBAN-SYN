"""Convolution builders for the configurable road GNN encoder."""
from collections.abc import Callable
from functools import partial

from torch import nn
from torch_geometric.nn import GATv2Conv, GINEConv, MLP

from .config import GATv2Settings, GINESettings, GNNSettings

# Receives input:
# width, target output width, optional edge width, and final-layer flag.
ConvFactory = Callable[[int, int, int | None, bool], nn.Module]

def build_gatv2_conv(
    in_dim: int,
    out_dim: int,
    edge_dim: int | None,
    is_final: bool,
    options: GATv2Settings,
) -> GATv2Conv:
    """Construct GATv2 with output width independent of head aggregation."""
    concat = options.final_concat if is_final else options.concat
    
    # In concatenation mode PyG's out_channels is the width of *each* head.
    if concat and out_dim % options.heads:
        raise ValueError("GATv2 target width must be divisible by heads when concat=true.")
    
    per_head_dim = out_dim // options.heads if concat else out_dim
    return GATv2Conv(
        in_channels=in_dim,
        out_channels=per_head_dim,
        heads=options.heads,
        concat=concat,
        dropout=options.attention_dropout,
        add_self_loops=options.add_self_loops,
        edge_dim=edge_dim,
        residual=False,  # RoadGNNEncoder manages residuals for both architectures.
    )


def build_gine_conv(
    in_dim: int,
    out_dim: int,
    edge_dim: int | None,
    is_final: bool,
    options: GINESettings,
) -> GINEConv:
    """Construct GINE using PyG's configurable MLP."""
    if edge_dim is None:
        raise ValueError("GINE requires selected edge features (graph.edge_attr).")
    
    mlp_options = options.mlp
    mlp = MLP(
        in_channels=in_dim,
        hidden_channels=mlp_options.hidden_dim,
        out_channels=out_dim,
        num_layers=mlp_options.num_layers,
        act=mlp_options.activation,
        norm=mlp_options.normalization,
        dropout=mlp_options.dropout,
        plain_last=mlp_options.plain_last,
    )

    return GINEConv(
        nn=mlp,
        edge_dim=edge_dim,
        eps=options.eps,
        train_eps=options.train_eps,
    )


def make_conv_factory(settings: GNNSettings) -> ConvFactory:
    """Select a convolution builder without coupling the encoder to PyG variants."""
    if settings.architecture == "gatv2":
        return partial(build_gatv2_conv, options=settings.gatv2)
    if settings.architecture == "gine":
        return partial(build_gine_conv, options=settings.gine)

    raise ValueError(f"Unsupported GNN architecture: {settings.architecture!r}.")
