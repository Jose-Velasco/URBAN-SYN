"""Independent training objectives for road-aware diffusion experiments.

Each objective receives a shared context and produces one scalar. The composer
controls selection, weights, and reporting without knowing the loss equations.
"""

from dataclasses import dataclass, field
from math import isfinite
from typing import Mapping, Protocol

import torch
import torch.nn.functional as F
from torch import Tensor

from .diff_util import DiffusionForwardOutput
from .road_ids import encode_transition_keys


@dataclass(frozen=True)
class LossContext:
    """Data shared by loss functions during one training forward pass.

    Attributes
    ----------
    diffusion : DiffusionForwardOutput
        Clean/noisy representations and denoiser predictions for this batch.
    road_embeddings : Tensor
        Road embedding table [N + 1, D], including model padding ID 0.
    valid_transition_keys : Tensor
        Encoded directed road transitions in model-ID space.
    road_id_base : int
        Size of the model road vocabulary, including padding.
    similarity_chunk_size : int
        Number of reconstructed road vectors decoded per chunk.
    batch : Mapping[str, Tensor]
        Optional auxiliary batch tensors for future objectives.
    """

    diffusion: DiffusionForwardOutput
    road_embeddings: Tensor
    valid_transition_keys: Tensor
    road_id_base: int
    similarity_chunk_size: int = 128
    batch: Mapping[str, Tensor] = field(default_factory=dict)


class LossFunction(Protocol):
    """Callable contract implemented by independently testable objectives."""

    def __call__(self, context: LossContext) -> Tensor:
        """Compute one scalar objective from the shared training context.

        Parameters
        ----------
        context : LossContext
            Denoising results and optional auxiliary training tensors.

        Returns
        -------
        Tensor
            Scalar objective or diagnostic value.
        """
        ...


class NoiseMSE:
    """Mean-squared error between predicted and sampled Gaussian noise."""

    def __call__(self, context: LossContext) -> Tensor:
        """Calculate the noise-prediction loss.

        Parameters
        ----------
        context : LossContext
            Contains predicted noise and the noise sampled during noising.

        Returns
        -------
        Tensor
            Differentiable scalar noise MSE.
        """
        output = context.diffusion
        return F.mse_loss(output.predicted_noise, output.sampled_noise)


class ReconstructionMSE:
    """Road-embedding reconstruction MSE, excluding the position-rate channel."""

    def __call__(self, context: LossContext) -> Tensor:
        """Calculate the original Diff-RNTraj road reconstruction loss.

        Parameters
        ----------
        context : LossContext
            Contains the clean and reconstructed trajectory representations.

        Returns
        -------
        Tensor
            Differentiable scalar MSE over road embedding channels only.
        """
        output = context.diffusion
        return F.mse_loss(output.reconstructed[..., :-1], output.clean[..., :-1])


@torch.no_grad()
def count_invalid_transitions(context: LossContext) -> Tensor:
    """Count hard-decoded consecutive road pairs missing from the directed graph.
    the original Diff-RNTraj implementation treats connectivity as a loss despite using non-differentiable hard decoding.

    Padding ID 0 is kept among the candidates to reproduce the original
    training-time connectivity calculation. Decoding uses dot product, not
    the cosine similarity employed by the generation path.

    Parameters
    ----------
    context : LossContext
        Reconstructed trajectories, road embeddings, and valid transition keys.

    Returns
    -------
    Tensor
        Scalar count over the complete batch, without a gradient.
    """
    if context.similarity_chunk_size < 1:
        raise ValueError("similarity_chunk_size must be positive.")

    road_vectors = context.diffusion.reconstructed[..., :-1]
    batch_size, length, width = road_vectors.shape
    queries = road_vectors.reshape(-1, width)
    predicted = []

    for block in queries.split(context.similarity_chunk_size):
        predicted.append((block @ context.road_embeddings.T).argmax(dim=-1))

    model_ids = torch.cat(predicted).reshape(batch_size, length)
    pair_keys = encode_transition_keys(
        model_ids[:, :-1],
        model_ids[:, 1:],
        context.road_id_base
    )
    return (~torch.isin(pair_keys, context.valid_transition_keys)).sum().float()


class HardConnectivity:
    """Original hard connectivity count, available as an objective or metric.
       
       the original Diff-RNTraj implementation treats connectivity as a loss despite using non-differentiable hard decoding.
    """

    def __call__(self, context: LossContext) -> Tensor:
        """Calculate the original non-differentiable connectivity term.

        Parameters
        ----------
        context : LossContext
            Contains reconstructed road vectors and directed graph transitions.

        Returns
        -------
        Tensor
            Scalar invalid-transition count with no gradient.
        """
        return count_invalid_transitions(context)


@dataclass(frozen=True)
class ConfiguredLoss:
    """One selected objective and its weighting/reporting policy.

    Attributes
    ----------
    name : str
        Unique identifier recorded in training metrics.
    function : LossFunction
        Implementation that calculates a scalar value.
    weight : float
        Multiplier applied when the value is included in the total.
    include_in_total : bool
        Whether to include the weighted value in the reported objective.
    every_steps : int
        Evaluation frequency for metric-only terms.
    differentiable : bool
        Whether the objective is intended to provide gradients.
    """

    name: str
    function: LossFunction
    weight: float = 1.0
    include_in_total: bool = True
    every_steps: int = 1
    differentiable: bool = True


@dataclass(frozen=True)
class LossResult:
    """Weighted total plus independently logged loss and diagnostic values.

    Attributes
    ----------
    total : Tensor
        Sum of selected weighted terms, including any hard connectivity term.
    gradient_total : Tensor
        Sum of weighted terms declared differentiable. For diagnostics only;
        use ``total.backward()`` to preserve the selected experiment objective.
    values : dict[str, Tensor]
        Unweighted scalar values of computed components.
    weighted_values : dict[str, Tensor]
        Weighted scalar values of components included in the total.
    """

    total: Tensor
    gradient_total: Tensor
    values: dict[str, Tensor]
    weighted_values: dict[str, Tensor]


class LossComposer:
    """Compute registered losses and combine exactly the enabled objectives."""

    def __init__(self, losses: list[ConfiguredLoss]):
        """Validate and retain the selected loss implementations.

        Parameters
        ----------
        losses : list[ConfiguredLoss]
            Active loss components with configured weights and frequencies.
        """
        if not losses:
            raise ValueError("At least one loss component is required.")

        names = [loss.name for loss in losses]
        if len(set(names)) != len(names):
            raise ValueError("Loss names must be unique.")

        for loss in losses:
            if not isfinite(loss.weight) or loss.weight < 0:
                raise ValueError(f"Invalid non-negative weight for {loss.name}.")
            if loss.every_steps < 1:
                raise ValueError(f"every_steps must be positive for {loss.name}.")
            if loss.include_in_total and loss.every_steps != 1:
                raise ValueError(
                    f"{loss.name} contributes to total and must run every step."
                )

        if not any(
            loss.include_in_total and loss.differentiable and loss.weight > 0
            for loss in losses
        ):
            raise ValueError("At least one differentiable loss must contribute to total.")

        self.losses = tuple(losses)

    def __call__(self, context: LossContext, step: int = 0) -> LossResult:
        """Evaluate selected loss terms and sum their weighted contributions.

        Parameters
        ----------
        context : LossContext
            Shared batch output and graph-related tensors.
        step : int, default=0
            Zero-based optimizer step used to throttle metric-only components.

        Returns
        -------
        LossResult
            Reported total, gradient-bearing total, and per-component values.
        """
        if step < 0:
            raise ValueError("step cannot be negative.")

        values: dict[str, Tensor] = {}
        weighted_values: dict[str, Tensor] = {}
        total = context.diffusion.clean.new_zeros(())
        gradient_total = context.diffusion.clean.new_zeros(())

        for loss in self.losses:
            if step % loss.every_steps != 0:
                continue

            value = loss.function(context)
            if value.ndim != 0:
                raise ValueError(f"{loss.name} must return a scalar tensor.")
            values[loss.name] = value

            if loss.include_in_total:
                weighted = loss.weight * value
                weighted_values[loss.name] = weighted
                total = total + weighted
                if loss.differentiable:
                    gradient_total = gradient_total + weighted

        return LossResult(total, gradient_total, values, weighted_values)


def original_loss_composer() -> LossComposer:
    """Build the original three-term Diff-RNTraj training objective.

    Returns
    -------
    LossComposer
        Noise MSE + hard connectivity count + road reconstruction MSE,
        each with unit weight and included in the reported total.
    """
    return LossComposer([
        ConfiguredLoss("noise_mse", NoiseMSE()),
        ConfiguredLoss("hard_connectivity", HardConnectivity(), differentiable=False),
        ConfiguredLoss("reconstruction_mse", ReconstructionMSE())
    ])
