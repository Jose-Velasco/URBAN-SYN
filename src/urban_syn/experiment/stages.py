"""Build pipeline stages from an already validated experiment configuration."""

from pathlib import Path

import yaml

from .config import ResolvedExperimentConfig
from .stage import Stage


TARGETS = ("validate", "train", "generate", "export-eval", "all")


def _project_path(value: str, root: Path) -> Path:
    """Interpret graph-YAML paths relative to the repository root."""
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else root / candidate).resolve()


def _graph_inputs(config: ResolvedExperimentConfig) -> tuple[Path, ...]:
    """Locate the source GeoPackage and preprocessed PyG graph files.

    ``RoadGraphDataset.processed_file_names`` defines these two cache files.
    We check for them without constructing a graph or triggering processing.
    """
    graph_yaml = config.settings.paths.graph_config
    with graph_yaml.open("r", encoding="utf-8") as stream:
        graph_settings = yaml.safe_load(stream)

    dataset = graph_settings.get("dataset") if isinstance(graph_settings, dict) else None
    if not isinstance(dataset, dict):
        raise ValueError(f"Graph configuration requires a dataset mapping: {graph_yaml}")

    for name in ("root", "network_path"):
        if not isinstance(dataset.get(name), str) or not dataset[name].strip():
            raise ValueError(f"Graph configuration requires dataset.{name}: {graph_yaml}")

    root = _project_path(dataset["root"], config.project_root)
    network = _project_path(dataset["network_path"], config.project_root)
    return (
        network,
        root / "processed" / "road_graph.pt",
        root / "processed" / "preprocessor.joblib"
    )


def _required_items(paths: list[Path | None]) -> tuple[Path, ...]:
    """Return configured file paths while preserving order."""
    return tuple(dict.fromkeys(path for path in paths if path is not None))


def _stage_map(config: ResolvedExperimentConfig) -> dict[str, Stage]:
    """Define input checks and dependencies without executing model code."""
    paths = config.settings.paths
    graph_inputs = _graph_inputs(config)
    training_files = [paths.graph_config, paths.train_sequences]
    training_files.extend(graph_inputs)
    if config.settings.training.validation.enabled:
        training_files.append(paths.trip_id_map)
    training_inputs = _required_items(training_files)

    # Generation requires a full road-geometry GeoPackage, specified in YAML.
    road_network = paths.road_network
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
            description="Check existing road graph and training inputs without loading a GPU model.",
            required_files=training_inputs
        ),
        "train": Stage(
            name="train",
            description="Train the joint road GNN and Diff-RNTraj model.",
            required_files=training_inputs,
            depends_on=("validate",)
        ),
        "generate": Stage(
            name="generate",
            description="Generate trajectories using a trained joint checkpoint.",
            required_files=_required_items([road_network, paths.length_distribution]),
            missing_settings=generation_missing,
            depends_on=("train",)
        ),
        "export-eval": Stage(
            name="export-eval",
            description="Prepare generated trajectories and held-out references for shared evaluation.",
            required_files=_required_items([paths.test_reference_gps]),
            missing_settings=export_missing,
            depends_on=("generate",)
        )
    }


def stages_for_target(config: ResolvedExperimentConfig, target: str) -> tuple[Stage, ...]:
    """Select either one independent stage or the configured full pipeline.

    Selecting ``generate`` alone does not silently retrain the model.
    ``all`` respects ``generation.enabled`` in the experiment YAML.
    """
    if target not in TARGETS:
        raise ValueError(f"Unknown target {target!r}. Choose from: {', '.join(TARGETS)}")

    stage_map = _stage_map(config)
    if target == "all":
        names = ["validate", "train"]
        if config.settings.generation.enabled:
            names.append("generate")
            if config.settings.paths.test_reference_gps is not None:
                names.append("export-eval")
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
