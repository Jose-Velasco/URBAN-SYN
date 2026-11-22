"""Pretraining stage declarations using the shared experiment Stage contract."""

from __future__ import annotations

from .config import ResolvedLinkPretrainConfig
from urban_syn.pipeline.stage import Stage


TARGETS = ("validate", "prepare", "train", "test", "all")


def _stage_map(config: ResolvedLinkPretrainConfig) -> dict[str, Stage]:
    """Describe pretraining work and prerequisite source settings.

    The ``validate`` handler owns graph-cache existence and feature checks,
    so the same checks do not need to be implemented in the stage planner.
    """
    graph_yaml = (config.settings.paths.graph_config,)
    return {
        "validate": Stage(
            name="validate",
            description="Validate configuration, cached graph, and selected features.",
            required_files=graph_yaml
        ),
        "prepare": Stage(
            name="prepare",
            description="Validate a seeded directed split in memory and log its summary.",
            required_files=graph_yaml,
            depends_on=("validate",)
        ),
        "train": Stage(
            name="train",
            description="Pretrain the road GNN and directional link scorer.",
            required_files=graph_yaml,
            depends_on=("prepare",)
        ),
        "test": Stage(
            name="test",
            description="Evaluate held-out links using the selected checkpoint.",
            required_files=graph_yaml,
            depends_on=("train",)
        )
    }


def stages_for_target(config: ResolvedLinkPretrainConfig, target: str) -> tuple[Stage, ...]:
    """Select ordered primitive pipeline steps for a CLI target.

    Args:
        config: Resolved experiment configuration used for prerequisites.
        target: One of ``validate``, ``prepare``, ``train``, ``test``, or ``all``.

    Returns:
        Ordered ``Stage`` descriptions to execute for the target. The
        ``test`` target is separate and does not imply retraining.

    Raises:
        ValueError: If the requested target is unknown.
    """
    if target not in TARGETS:
        raise ValueError(f"Unknown pretraining target: {target}")
    stage_map = _stage_map(config)
    if target == "all":
        names = ("validate", "prepare", "train", "test")
    elif target == "test":
        names = ("validate", "test")
    elif target == "validate":
        names = ("validate",)
    elif target == "prepare":
        names = ("validate", "prepare")
    else:
        names = ("validate", "prepare", "train")
    return tuple(stage_map[name] for name in names)
