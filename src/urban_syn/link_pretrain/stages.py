"""Stage declarations for road-link pretraining and graph preparation."""

from __future__ import annotations

from .config import ResolvedLinkPretrainConfig
from urban_syn.pipeline.stage import Stage


TARGETS = ("validate", "prepare-graph", "prepare", "train", "test", "all")


def _stage_map(config: ResolvedLinkPretrainConfig) -> dict[str, Stage]:
    """Describe link-pretraining stages using the shared road-graph builder."""
    graph_yaml = config.settings.paths.graph_config
    return {
        "prepare-graph": Stage(
            name="prepare-graph",
            description="Reuse or build the cached road graph and feature preprocessor.",
            required_files=(graph_yaml,)
        ),
        "validate": Stage(
            name="validate",
            description="Validate the cached graph and selected features.",
            required_files=(graph_yaml,)
        ),
        "prepare": Stage(
            name="prepare",
            description="Validate a seeded directed split in memory and log its summary.",
            required_files=(graph_yaml,),
            depends_on=("validate",)
        ),
        "train": Stage(
            name="train",
            description="Pretrain the road GNN and directional link scorer.",
            required_files=(graph_yaml,),
            depends_on=("prepare",)
        ),
        "test": Stage(
            name="test",
            description="Evaluate held-out links using the selected checkpoint.",
            required_files=(graph_yaml,),
            depends_on=("validate",)
        )
    }


def stages_for_target(config: ResolvedLinkPretrainConfig, target: str) -> tuple[Stage, ...]:
    """Include graph building in full runs without changing standalone targets.

    Args:
        config: Validated pretraining settings and graph configuration.
        target: Requested CLI target.

    Returns:
        Ordered stages. Full runs prepare the graph; standalone stages reuse it.
    """
    if target not in TARGETS:
        raise ValueError(f"Unknown pretraining target: {target}")
    stage_map = _stage_map(config)
    if target == "all":
        names = ("prepare-graph", "validate", "prepare", "train", "test")
    elif target in ("validate", "prepare-graph"):
        names = (target,)
    elif target == "test":
        names = ("validate", "test")
    elif target == "prepare":
        names = ("validate", "prepare")
    else:
        names = ("validate", "prepare", "train")
    return tuple(stage_map[name] for name in names)
