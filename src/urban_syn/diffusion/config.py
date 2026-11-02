"""Configuration for joint road-GNN and diffusion training."""
from pathlib import Path
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Self
from urban_syn.gnn.config import GNNSettings

class _Settings(BaseModel):
    """Use validated, immutable settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

class DiffusionSettings(_Settings):
    """Settings for Yun's diffusion denoiser and noise schedule."""

    hidden_dim: int = Field(default=128, gt=0)
    residual_layers: int = Field(default=10, gt=0)
    diffusion_steps: int = Field(default=500, gt=0)
    beta_start: float = Field(default=0.0001, gt=0, lt=1)
    beta_end: float = Field(default=0.02, gt=0, lt=1)
    similarity_chunk_size: int = Field(default=128, gt=0)

    @model_validator(mode="after")
    def validate_beta_range(self) -> Self:
        """Validate the diffusion noise schedule."""
        if self.beta_start > self.beta_end:
            raise ValueError("beta_start must not exceed beta_end.")
        if self.hidden_dim < 4 or self.hidden_dim % 2:
            raise ValueError("hidden_dim must be even and at least 4.")
        return self

class TrainingSettings(_Settings):
    """Settings for joint optimization."""

    epochs: int = Field(default=30, gt=0)
    batch_size: int = Field(default=256, gt=0)
    learning_rate: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    grad_clip_norm: float = Field(default=1.0, gt=0)
    max_batches: int | None = Field(default=None, gt=0)

class JointTrainingSettings(_Settings):
    """Validated settings required by the joint training pipeline."""

    gnn: GNNSettings
    diffusion: DiffusionSettings
    training: TrainingSettings

def load_joint_training_config(
    path: str | Path,
) -> JointTrainingSettings:
    """Load the relevant sections from the existing graph YAML.

    Parameters
    ----------
    path : str | Path
        YAML containing model.gnn, model.diffusion, and training.

    Returns
    -------
    JointTrainingSettings
        Validated joint model and training settings.
    """
    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return JointTrainingSettings.model_validate({
        "gnn": config["model"]["gnn"],
        "diffusion": config["model"]["diffusion"],
        "training": config["training"],
    })
