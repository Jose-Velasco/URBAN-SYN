"""Directional link prediction with the existing road-GNN encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from .config import DecoderSettings, ResolvedLinkPretrainConfig
from .data import LinkBatch
from .validation import validate_decoder_inputs, validate_link_logits

if TYPE_CHECKING:
    from torch_geometric.data import Data


@dataclass(frozen=True)
class LinkScores:
    """Raw link logits with positive-to-negative row alignment.

    Attributes:
        positive: One logit per observed directed link, shape ``[P]``.
        negative: One row of negative-link logits per positive, shape
            ``[P, amount]``. These scores are not probabilities.
    """

    positive: Tensor
    negative: Tensor

    def __post_init__(self) -> None:
        """Require nonempty, source-aligned logits with matching dtype and device."""
        validate_link_logits(self.positive, self.negative)


class DirectionalLinkDecoder(nn.Module):
    """Score ordered road pairs using a configurable PyG MLP.

    Concatenation retains direction: the scores of ``(u, v)`` and ``(v, u)``
    may differ. The last layer always outputs a raw, unrestricted logit.
    """

    def __init__(self, embedding_dim: int, settings: DecoderSettings) -> None:
        """Create the decoder from validated architecture settings.

        Args:
            embedding_dim: Width of a single road embedding from the GNN.
            settings: Decoder depth, hidden width, activation, dropout, and
                hidden-layer normalization, loaded from YAML.

        Raises:
            ValueError: If the embedding dimension is not positive.
        """
        super().__init__()
        if embedding_dim < 1:
            raise ValueError("Decoder embedding dimension must be positive.")

        from torch_geometric.nn import MLP

        self.embedding_dim = embedding_dim
        self.mlp = MLP(
            in_channels=2 * embedding_dim,
            hidden_channels=settings.hidden_dim,
            out_channels=1,
            num_layers=settings.num_layers,
            act=settings.activation,
            norm=settings.normalization,
            dropout=settings.dropout,
            plain_last=True
        )

    def forward(self, embeddings: Tensor, sources: Tensor, destinations: Tensor) -> Tensor:
        """Return logits for ordered pairs of road indices.

        Args:
            embeddings: Road embeddings, shape ``[N, embedding_dim]``.
            sources: Zero-based source indices, shape ``[P]`` or
                ``[P, amount]``.
            destinations: Destination indices with the same shape as sources.

        Returns:
            Floating-point logits with the same shape as ``sources``.

        Raises:
            ValueError: If input shapes, dtypes, or devices do not match.
            IndexError: If an index is outside the range of road embeddings.
        """
        validate_decoder_inputs(embeddings, sources, destinations, self.embedding_dim)

        # Look up only the requested road embeddings, not every possible pair.
        source_features = embeddings[sources]
        destination_features = embeddings[destinations]
        pair_features = torch.cat((source_features, destination_features), dim=-1)

        # The decoder accepts 2D features. Flatten aligned negative pairs and
        # restore their [P, amount] shape after computing one logit per pair.
        flattened_features = pair_features.reshape(-1, 2 * self.embedding_dim)
        logits = self.mlp(flattened_features)
        return logits.reshape(sources.shape)


class RoadLinkPredictionModel(nn.Module):
    """Combine the road encoder and a directional link scorer."""

    def __init__(self, encoder: nn.Module, decoder: DirectionalLinkDecoder) -> None:
        """Keep the encoder and decoder together for optimization/checkpointing.

        Args:
            encoder: Road encoder that maps node/edge features to ``[N, D]``.
            decoder: Ordered-pair scorer that accepts embeddings of width ``D``.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, train_graph: Data) -> Tensor:
        """Embed roads using only the training message-passing graph.

        Args:
            train_graph: Device-ready PyG training split with node features
                ``x``, training ``edge_index``, and optional aligned
                ``edge_attr``. Held-out edges must not be included.

        Returns:
            Embeddings for all roads, shape ``[N, embedding_dim]``.
        """
        edge_features = getattr(train_graph, "edge_attr", None)
        return self.encoder(train_graph.x, train_graph.edge_index, edge_features)

    def score_batch(
        self,
        embeddings: Tensor,
        batch: LinkBatch,
        non_blocking: bool = True
    ) -> LinkScores:
        """Score sampled links using embeddings computed once for the graph.

        Args:
            embeddings: Road embeddings, shape ``[N, embedding_dim]``.
            batch: CPU sampling output with sources and positives ``[P]``
                and source-aligned negative destinations ``[P, amount]``.
            non_blocking: Permit asynchronous index transfers when CPU
                batches are pinned and embeddings are on CUDA.

        Returns:
            Positive and negative logits on the embeddings' device.

        Raises:
            ValueError: If the sampled batch has no positive links.
        """
        if batch.positive_src.numel() == 0:
            raise ValueError("Cannot score an empty link batch; skip this training step.")

        # Sampling stays on the CPU. Only the selected road IDs are moved
        # to the GPU; the graph and its embeddings already reside there.
        sources = batch.positive_src.to(embeddings.device, non_blocking=non_blocking)
        destinations = batch.positive_dst.to(embeddings.device, non_blocking=non_blocking)
        negatives = batch.negative_dst.to(embeddings.device, non_blocking=non_blocking)

        # Each positive source must be paired with every negative in its row.
        negative_sources = sources.unsqueeze(1).expand_as(negatives)
        positive_logits = self.decoder(embeddings, sources, destinations)
        negative_logits = self.decoder(embeddings, negative_sources, negatives)

        return LinkScores(positive=positive_logits, negative=negative_logits)

    def forward(
        self,
        train_graph: Data,
        batch: LinkBatch,
        non_blocking: bool = True
    ) -> LinkScores:
        """Encode the training graph and score one sampled link batch.

        Args:
            train_graph: Device-ready training-only message graph.
            batch: CPU ``LinkBatch`` returned by the sampling DataLoader.
            non_blocking: Allow asynchronous index transfers if pinned.

        Returns:
            ``LinkScores`` with ``[P]`` positives and ``[P, amount]`` negatives.
        """
        embeddings = self.encode(train_graph)
        return self.score_batch(embeddings, batch, non_blocking=non_blocking)


def build_link_prediction_model(
    train_graph: Data,
    config: ResolvedLinkPretrainConfig
) -> RoadLinkPredictionModel:
    """Construct the existing road encoder and configured directional MLP.

    Args:
        train_graph: PyG training message graph with selected node and edge
            features. No validation or test links may be used for messages.
        config: Validated experiment settings and the graph's GNN settings.

    Returns:
        New model containing a ``RoadGNNEncoder`` and ``DirectionalLinkDecoder``.
        The caller is responsible for moving it to the selected device.

    Raises:
        ValueError: If the graph features or GNN settings are invalid.
    """
    from urban_syn.gnn.config import parse_gnn_config
    from urban_syn.gnn.encoder import RoadGNNEncoder
    from urban_syn.gnn.factory import make_conv_factory

    gnn_settings = parse_gnn_config(config.graph_settings["model"]["gnn"])
    encoder = RoadGNNEncoder.from_graph(
        train_graph,
        gnn_settings,
        make_conv_factory(gnn_settings)
    )
    decoder = DirectionalLinkDecoder(
        embedding_dim=gnn_settings.output_dim,
        settings=config.settings.pretraining.decoder
    )
    return RoadLinkPredictionModel(encoder, decoder)
