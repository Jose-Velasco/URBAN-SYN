"""Trainable road encoders for synthetic mobility models."""
from .config import GNNSettings, parse_gnn_config
from .encoder import RoadGNNEncoder
from .factory import ConvFactory, make_conv_factory

__all__ = ["ConvFactory", "GNNSettings", "RoadGNNEncoder", "make_conv_factory", "parse_gnn_config"]
