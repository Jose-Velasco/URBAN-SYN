"""Scale diffusion gradients reaching the road GNN without changing embeddings."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def normalized_sigmoid_scale(
    step: int,
    warmup_steps: int,
    gamma_min: float = 0.0,
    k: float = 10.0,
    c: float = 0.8
) -> float:
    """Calculate the normalized-sigmoid gradient coefficient from LDLM.

    Args:
        step: Zero-based optimizer step, not the diffusion noise timestep.
        warmup_steps: Number of optimizer steps until the scale reaches one.
        gamma_min: Initial coefficient in [0, 1).
        k: Positive sigmoid steepness.
        c: Fractional location of the sigmoid center, in (0, 1).

    Returns:
        A coefficient in [gamma_min, 1]. It equals gamma_min at step zero
        and reaches exactly one at or after warmup_steps.

    Raises:
        ValueError: If any setting is invalid or the sigmoid range collapses.
    """
    _validate_schedule_parameters(step, warmup_steps, gamma_min, k, c)

    if step == 0:
        return gamma_min
    if step >= warmup_steps:
        return 1.0

    logits = torch.tensor(
        [-k * c, k * (1.0 - c), k * (step / warmup_steps - c)],
        dtype=torch.float64
    )
    start, end, current = torch.sigmoid(logits).tolist()

    sigmoid_range = end - start
    if sigmoid_range <= 0:
        raise ValueError("The sigmoid parameters produce an indistinguishable warmup range.")

    normalized_progress = (current - start) / sigmoid_range
    scale = gamma_min + (1.0 - gamma_min) * normalized_progress
    scale = max(scale, gamma_min)
    return min(scale, 1.0)


def apply_gradient_warmup(embeddings: Tensor, scale: float) -> Tensor:
    """Keep embeddings unchanged in the forward pass and scale their gradients.

    Args:
        embeddings: Road GNN output tensor, including any padding row.
        scale: Finite gradient coefficient in [0, 1].

    Returns:
        Tensor with the original embedding values and gradients to the
        source embeddings multiplied by scale.

    Raises:
        ValueError: If scale is non-finite or outside [0, 1].
    """
    if not math.isfinite(scale) or not 0 <= scale <= 1:
        raise ValueError("Gradient scale must be finite and in [0, 1].")
    if scale == 1.0:
        return embeddings

    detached = embeddings.detach()
    return detached + scale * (embeddings - detached)


def _validate_schedule_parameters(
    step: int,
    warmup_steps: int,
    gamma_min: float,
    k: float,
    c: float
) -> None:
    """Validate the schedule parameters before evaluating the sigmoid."""
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a nonnegative integer.")
    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 1:
        raise ValueError("warmup_steps must be a positive integer.")
    if not math.isfinite(gamma_min) or not 0 <= gamma_min < 1:
        raise ValueError("gamma_min must be finite and in [0, 1).")
    if not math.isfinite(k) or k <= 0:
        raise ValueError("k must be finite and positive.")
    if not math.isfinite(c) or not 0 < c < 1:
        raise ValueError("c must be finite and in (0, 1).")
