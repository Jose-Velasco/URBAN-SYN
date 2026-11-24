"""Restore the frozen directional scorer paired with a pretrained road GNN.

The study launcher supplies encoder.pt, and road-link pretraining publishes
best.pt in the same checkpoints directory. This module loads the decoder
from that selected pretraining run without rerunning link pretraining.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from urban_syn.link_pretrain.config import DecoderSettings
from urban_syn.link_pretrain.model import DirectionalLinkDecoder

from .config import ResolvedExperimentConfig
from .pretrained_encoder import _mismatches


def resolve_scorer_checkpoint(config: ResolvedExperimentConfig) -> Path | None:
    """Locate the full checkpoint matching the selected pretrained encoder.

    Args:
        config (ResolvedExperimentConfig): Validated experiment configuration,
            including scorer settings and the selected encoder checkpoint.

    Returns:
        Path | None: Explicit scorer checkpoint, paired ``best.pt`` for
            automatic selection, or ``None`` when the scorer is disabled.

    Raises:
        ValueError: If automatic selection lacks a compatible ``encoder.pt``
            location. Study-launched jobs supply that location after pretraining.
    """
    settings = config.settings.model.connectivity_scorer
    if not settings.enabled:
        return None

    if isinstance(settings.checkpoint, Path):
        return settings.checkpoint

    encoder = config.settings.model.road_encoder.checkpoint
    if encoder is None or encoder.name != "encoder.pt" or encoder.parent.name != "checkpoints":
        raise ValueError(
            "Automatic connectivity scorer selection requires the matching "
            "pretraining checkpoints/encoder.pt path."
        )
    return encoder.with_name("best.pt")


def _decoder_state(checkpoint: dict[str, Any], path: Path) -> dict[str, Tensor]:
    """Extract the directional MLP state from a complete pretraining checkpoint.

    Args:
        checkpoint (dict[str, Any]): Loaded full pretraining checkpoint.
        path (Path): Checkpoint path used in validation errors.

    Returns:
        dict[str, Tensor]: MLP parameter names and saved values, excluding the
            ``decoder.`` prefix used by the pretraining model.

    Raises:
        ValueError: If the checkpoint lacks a decoder state.
    """
    full_state = checkpoint.get("model_state_dict")
    if not isinstance(full_state, dict):
        raise ValueError(f"Missing model_state_dict in scorer checkpoint: {path}")

    prefix = "decoder."
    decoder_state = {
        key[len(prefix):]: value
        for key, value in full_state.items()
        if isinstance(key, str) and key.startswith(prefix)
    }
    if not decoder_state:
        raise ValueError(f"No directional scorer weights in checkpoint: {path}")
    return decoder_state


def _verify_matching_encoder(
    model: nn.Module,
    checkpoint: dict[str, Any],
    path: Path
) -> None:
    """Verify that the scorer and selected road GNN share pretrained weights.

    Args:
        model (nn.Module): Joint model with its pretrained road encoder loaded.
        checkpoint (dict[str, Any]): Full checkpoint containing encoder weights.
        path (Path): Source checkpoint path for validation messages.

    Returns:
        None: Raises if the encoder states differ.

    Raises:
        ValueError: If encoder keys, parameter values, or tensor types differ.

    This check is used for new training runs. When resuming joint training, the
    joint checkpoint can replace the initially loaded encoder parameters.
    """
    state = checkpoint["model_state_dict"]
    current = model.road_encoder.state_dict()
    prefix = "encoder."
    saved = {
        key[len(prefix):]: value
        for key, value in state.items()
        if isinstance(key, str) and key.startswith(prefix)
    }
    if set(saved) != set(current):
        raise ValueError(
            f"Scorer checkpoint {path} has an encoder state incompatible with "
            "the selected encoder.pt."
        )
    for key, tensor in current.items():
        other = saved[key]
        if not isinstance(other, Tensor) or not torch.equal(tensor.detach().cpu(), other):
            raise ValueError(
                f"Scorer checkpoint {path} does not match the selected encoder.pt "
                f"at {key!r}. Select a scorer from the same pretraining run."
            )


def load_pretrained_scorer(
    model: nn.Module,
    graph: Data,
    config: ResolvedExperimentConfig
) -> DirectionalLinkDecoder:
    """Load a pretrained directional scorer and freeze its parameters.

    Args:
        model (nn.Module): Joint model holding the selected road encoder.
        graph (Data): Road graph used to validate pretrained model metadata.
        config (ResolvedExperimentConfig): Resolved scorer and encoder settings.

    Returns:
        DirectionalLinkDecoder: Scorer in evaluation mode with frozen weights.
            Gradients may still propagate through its input embeddings.

    Raises:
        FileNotFoundError: If the selected pretraining checkpoint is missing.
        ValueError: If the checkpoint or its encoder/scorer state is incompatible.
    """
    path = resolve_scorer_checkpoint(config)
    if path is None:
        raise ValueError("model.connectivity_scorer.enabled must be true.")
    if not path.is_file():
        raise FileNotFoundError(
            f"Pretrained connectivity scorer not found: {path}. "
            "Pretraining must publish best.pt beside its encoder.pt."
        )

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError(f"Unsupported road-link checkpoint format: {path}")

    metadata = checkpoint.get("model_metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"Missing scorer architecture metadata in: {path}")
    issues = _mismatches(metadata, graph, config)
    if issues:
        raise ValueError(
            "Pretrained connectivity scorer is incompatible with the joint graph/GNN:\n  - "
            + "\n  - ".join(issues)
        )
    if metadata.get("objective") not in ("bce", "bpr"):
        raise ValueError(f"Invalid BCE/BPR objective metadata in: {path}")
    losses = getattr(config.settings.training, "losses", None)
    for objective in ("bce", "bpr"):
        guidance = getattr(losses, f"connectivity_{objective}", None)
        if getattr(guidance, "enabled", False) and metadata["objective"] != objective:
            raise ValueError(
                f"{objective.upper()} connectivity guidance requires a "
                f"{objective.upper()}-pretrained scorer. "
                f"Checkpoint {path} uses {metadata['objective']!r}."
            )
    decoder_options = metadata.get("decoder")
    if not isinstance(decoder_options, dict):
        raise ValueError(f"Missing directional decoder settings in: {path}")

    decoder_settings = DecoderSettings.model_validate(decoder_options)
    decoder = DirectionalLinkDecoder(
        embedding_dim=config.joint_settings.gnn.output_dim,
        settings=decoder_settings
    )
    saved = _decoder_state(checkpoint, path)
    actual = decoder.state_dict()
    if set(saved) != set(actual):
        raise ValueError(
            f"Scorer checkpoint {path} has different parameter names. "
            "Verify its decoder architecture and PyG version."
        )
    for name, expected in actual.items():
        value = saved[name]
        if not isinstance(value, Tensor) or value.shape != expected.shape:
            raise ValueError(
                f"Scorer parameter {name!r} does not match the current architecture."
            )

    if (config.settings.training.checkpoint.resume_from is None
            and config.settings.model.road_encoder.initialization == "pretrained"):
        _verify_matching_encoder(model, checkpoint, path)

    decoder.load_state_dict(saved, strict=True)
    decoder.pretrained_objective = metadata["objective"]
    decoder.requires_grad_(False)
    decoder.eval()
    return decoder
