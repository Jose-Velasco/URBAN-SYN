"""Shared interface for trajectory diffusion samplers."""

from collections.abc import Mapping
from typing import Any, Protocol

from torch import Tensor, nn


class Sampler(Protocol):
    """Generate trajectory representations using a trained noise predictor."""

    def sample(
        self,
        denoiser: nn.Module,
        schedule: Mapping[str, Any],
        batch_size: int,
        sequence_length: int,
        feature_dim: int
    ) -> Tensor:
        """Sample clean trajectory representations.

        Parameters
        ----------
        denoiser : nn.Module
            Trained noise-prediction network.
        schedule : Mapping[str, Any]
            Diffusion schedule. The DDPM implementation uses T, alpha,
            and alpha_bar as produced by the trained model.
        batch_size : int
            Number of trajectories to generate.
        sequence_length : int
            Positions per generated trajectory.
        feature_dim : int
            Number of features per position, including the position-rate channel.

        Returns
        -------
        Tensor
            Sampled trajectory representations [B, T, F].
        """
        ...
