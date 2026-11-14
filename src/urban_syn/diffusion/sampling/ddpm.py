"""Original Diff-RNTraj DDPM reverse sampler.

The reverse update and noise schedule are preserved from the existing
implementation. This module has no responsibility for road-ID decoding.
"""

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn


class DDPMSampler:
    """Sample with the existing stochastic DDPM reverse process."""

    @torch.no_grad()
    def sample(
        self,
        denoiser: nn.Module,
        schedule: Mapping[str, Any],
        batch_size: int,
        sequence_length: int,
        feature_dim: int
    ) -> Tensor:
        """Generate trajectory representations using the trained schedule.

        Parameters
        ----------
        denoiser : nn.Module
            Trained noise predictor accepting (noisy_trajectories, timestep).
        schedule : Mapping[str, Any]
            Trained diffusion schedule containing T, alpha, and alpha_bar.
        batch_size : int
            Number of trajectories to generate.
        sequence_length : int
            Number of positions in each generated trajectory.
        feature_dim : int
            Number of features per position, including normalized rate.

        Returns
        -------
        Tensor
            Generated clean trajectory representations [B, T, F].
        """
        diffusion_steps = int(schedule["T"])
        alpha = schedule["alpha"]
        alpha_bar = schedule["alpha_bar"]

        diff_input = torch.randn(
            (batch_size, sequence_length, feature_dim),
            device=alpha.device,
            dtype=alpha.dtype
        )

        for t in range(diffusion_steps - 1, -1, -1):
            predicted_noise = denoiser(diff_input, t)

            coeff1 = 1 / torch.sqrt(alpha[t])
            coeff2 = (1 - alpha[t]) / torch.sqrt(1 - alpha_bar[t])
            diff_input = coeff1 * (diff_input - coeff2 * predicted_noise)

            if t > 0:
                sigma = torch.sqrt(
                    (1 - alpha_bar[t - 1])
                    / (1 - alpha_bar[t])
                    * (1 - alpha[t])
                )
                diff_input = diff_input + sigma * torch.randn_like(diff_input)

        return diff_input


def cal_x0_from_noise_ddpm(
    net: nn.Module,
    diffusion_hyperparams: dict,
    batchsize: int,
    length: int,
    feature: int
) -> Tensor:
    """Provide the old DDPM sampling function API for existing callers.

    Parameters
    ----------
    net : nn.Module
        Trained noise-prediction network.
    diffusion_hyperparams : dict
        Diffusion schedule with T, alpha, and alpha_bar.
    batchsize : int
        Number of trajectories to generate.
    length : int
        Number of positions in each trajectory.
    feature : int
        Number of features per position.

    Returns
    -------
    Tensor
        Sampled clean trajectory representations [B, T, F].
    """
    return DDPMSampler().sample(net, diffusion_hyperparams, batchsize, length, feature)
