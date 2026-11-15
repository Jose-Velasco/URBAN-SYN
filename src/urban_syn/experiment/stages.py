"""Define source checks and execution stages for the experiment pipeline."""

from pathlib import Path

from .config import ResolvedExperimentConfig
from .graph_preparation import graph_paths, graph_rebuild_sources
from .stage import Stage


TARGETS = ("validate", "prepare-graph", "train", "generate", "export-eval", "all")


def _required_items(paths: list[Path | None]) -> tuple[Path, ...]:
    """Return configured source paths once, preserving their order.

    Parameters
    ----------
    paths : list[Path | None]
        Possible source files, including optional paths.

    Returns
    -------
    tuple[Path, ...]
        Non-null unique paths.
    """
    return tuple(dict.fromkeys(path for path in paths if path is not None))


def _stage_map(config: ResolvedExperimentConfig) -> dict[str, Stage]:
    """Declare inputs, separating source files from generated graph caches.

    Parameters
    ----------
    config : ResolvedExperimentConfig
        Validated experiment settings.

    Returns
    -------
    dict[str, Stage]
        Stages with only pre-existing inputs listed as requirements.
    """
    paths = config.settings.paths
    _, graph_network, _ = graph_paths(config)

    graph_files = [paths.graph_config, graph_network, *graph_rebuild_sources(config)]
    training_files = [*graph_files, paths.train_sequences]
    if config.settings.training.validation.enabled:
        training_files.append(paths.trip_id_map)
    training_inputs = _required_items(training_files)

    generation_missing = tuple(
        name for name, value in (
            ("paths.length_distribution", paths.length_distribution),
            ("paths.road_network", paths.road_network)
        ) if value is None
    )
    export_missing = (
        ("paths.test_reference_gps",) if paths.test_reference_gps is None else ()
    )

    return {
        "validate": Stage(
            name="validate",
            description="Check training and road-network source inputs.",
            required_files=training_inputs
        ),
        "prepare-graph": Stage(
            name="prepare-graph",
            description="Reuse or build the PyG road graph and fitted preprocessor.",
            required_files=_required_items(graph_files)
        ),
        "train": Stage(
            name="train",
            description="Train the joint road GNN and Diff-RNTraj model.",
            required_files=training_inputs,
            depends_on=("prepare-graph",)
        ),
        "generate": Stage(
            name="generate",
            description="Generate trajectories using a trained joint checkpoint.",
            required_files=_required_items([paths.road_network, paths.length_distribution]),
            missing_settings=generation_missing,
            depends_on=("prepare-graph",)
        ),
        "export-eval": Stage(
            name="export-eval",
            description="Prepare generated trajectories and references for shared evaluation.",
            required_files=_required_items([paths.test_reference_gps]),
            missing_settings=export_missing,
            depends_on=("generate",)
        )
    }


def stages_for_target(config: ResolvedExperimentConfig, target: str) -> tuple[Stage, ...]:
    """Select stages, inserting graph preparation before graph consumers.

    Parameters
    ----------
    config : ResolvedExperimentConfig
        Validated experiment configuration.
    target : str
        Requested pipeline stage or ``all``.

    Returns
    -------
    tuple[Stage, ...]
        Stages in execution order. Standalone generation never retrains.
    """
    if target not in TARGETS:
        raise ValueError(f"Unknown target {target!r}. Choose from: {', '.join(TARGETS)}")

    stage_map = _stage_map(config)
    if target == "all":
        names = ["validate", "prepare-graph", "train"]
        if config.settings.generation.enabled:
            names.append("generate")
            if config.settings.paths.test_reference_gps is not None:
                names.append("export-eval")
    elif target in ("train", "generate"):
        names = ["prepare-graph", target]
    else:
        names = [target]

    if target == "generate" and config.settings.generation.checkpoint is None:
        stage = stage_map["generate"]
        stage_map["generate"] = Stage(
            name=stage.name,
            description=stage.description,
            required_files=stage.required_files,
            missing_settings=(*stage.missing_settings, "generation.checkpoint"),
            depends_on=stage.depends_on
        )
    elif target == "generate":
        configured = config.settings.generation.checkpoint
        checkpoint = configured if configured.is_absolute() else config.project_root / configured
        stage = stage_map["generate"]
        stage_map["generate"] = Stage(
            name=stage.name,
            description=stage.description,
            required_files=(*stage.required_files, checkpoint.resolve()),
            missing_settings=stage.missing_settings,
            depends_on=stage.depends_on
        )

    return tuple(stage_map[name] for name in names)
