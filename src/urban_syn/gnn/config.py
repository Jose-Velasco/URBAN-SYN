"""Runtime-validated, YAML-independent settings for the road GNN encoder."""
from collections.abc import Mapping
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

class _Settings(BaseModel):
    """Reject unexpected settings and prevent changes after validation."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

class GATv2Settings(_Settings):
    """Options for GATv2 attention layers."""

    heads: int = Field(default=4, gt=0)
    concat: bool = True
    final_concat: bool = False
    attention_dropout: float = Field(default=0.1, ge=0, lt=1)
    add_self_loops: bool = False

class GINEMLPSettings(_Settings):
    """Uniform hidden-width PyG MLP options for each GINE layer."""

    hidden_dim: int = Field(default=128, gt=0)
    num_layers: int = Field(default=2, gt=0)
    activation: str = "relu"
    dropout: float = Field(default=0.1, ge=0, lt=1)
    normalization: str | None = "layer_norm"
    plain_last: bool = True

class GINESettings(_Settings):
    """Options for GINE convolution layers."""

    train_eps: bool = True
    eps: float = Field(default=0.0, allow_inf_nan=False)
    mlp: GINEMLPSettings = Field(default_factory=GINEMLPSettings)

class GNNSettings(_Settings):
    """GNN architecture options, independent of graph feature widths."""

    architecture: Literal["gatv2", "gine"] = "gatv2"
    hidden_dim: int = Field(default=128, gt=0)
    output_dim: int = Field(default=64, gt=0)
    num_layers: int = Field(default=2, gt=0)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    activation: str = "relu"
    normalization: Literal["layer", "layer_norm", "batch", "batch_norm", "none"] | None = "layer"
    residual: bool = True
    gatv2: GATv2Settings = Field(default_factory=GATv2Settings)
    gine: GINESettings = Field(default_factory=GINESettings)

    @model_validator(mode="after")
    def validate_head_dimensions(self) -> Self:
        """Ensure concatenated attention heads preserve requested layer widths."""
        if self.architecture != "gatv2":
            return self

        # A 1-layer model has no hidden convolution output to validate.
        if self.num_layers > 1 and self.gatv2.concat and self.hidden_dim % self.gatv2.heads:
            raise ValueError("gnn.hidden_dim must be divisible by gatv2.heads when concat=true.")
        if self.gatv2.final_concat and self.output_dim % self.gatv2.heads:
            raise ValueError("gnn.output_dim must be divisible by gatv2.heads when final_concat=true.")
        return self

def parse_gnn_config(config: Mapping[str, Any]) -> GNNSettings:
    """Validate the ``model.gnn`` section of a YAML configuration.

    Parameters
    ----------
    config : Mapping[str, Any]
        GNN-specific settings, not the complete YAML document.

    Returns
    -------
    GNNSettings
        Immutable, validated architecture settings.

    Raises
    ------
    ValueError
        If settings are missing, unknown, or invalid.
    """
    if not isinstance(config, Mapping):
        raise ValueError("model.gnn must be a mapping.")

    return GNNSettings.model_validate(dict(config))
