"""Prepare the cached PyG road graph using the existing graph builder."""

from pathlib import Path

import yaml

from .config import ResolvedExperimentConfig


def _project_path(value: str, project_root: Path) -> Path:
    """Resolve a graph-YAML path relative to the repository root.

    Parameters
    ----------
    value : str
        Path from the graph YAML.
    project_root : Path
        Resolved repository root.

    Returns
    -------
    Path
        Absolute path to a graph input or cache directory.
    """
    path = Path(value).expanduser()
    return (path if path.is_absolute() else project_root / path).resolve()


def resolve_graph_settings(config: ResolvedExperimentConfig) -> dict:
    """Resolve file paths in the graph configuration consistently.

    Both graph preparation and later graph loads must use the same normalized
    paths. PyG records the preprocessing transform representation in its cache;
    changing only the spelling of a source path triggers a mismatch warning.

    Parameters
    ----------
    config : ResolvedExperimentConfig
        Experiment settings identifying the graph YAML and project root.

    Returns
    -------
    dict
        Graph settings with absolute input and cache paths.
    """
    from urban_syn.graph.config import load_graph_config

    settings = load_graph_config(config.settings.paths.graph_config)
    dataset = dict(settings["dataset"])
    for name in ("root", "network_path"):
        dataset[name] = str(_project_path(dataset[name], config.project_root))
    settings["dataset"] = dataset

    graph = settings.get("graph") or {}
    transforms = graph.get("transforms") or {}
    transition = transforms.get("edge_transition_count") or {}
    if transition.get("enabled", False):
        match_csv = transition.get("match_csv")
        if not isinstance(match_csv, str) or not match_csv.strip():
            raise ValueError("graph.transforms.edge_transition_count.match_csv is required.")
        settings["graph"] = {
            **graph,
            "transforms": {
                **transforms,
                "edge_transition_count": {
                    **transition,
                    "match_csv": str(_project_path(match_csv, config.project_root))
                }
            }
        }
    return settings


def graph_paths(config: ResolvedExperimentConfig) -> tuple[Path, Path, tuple[Path, Path]]:
    """Find the source road network and both builder-produced cache files.

    Parameters
    ----------
    config : ResolvedExperimentConfig
        Experiment configuration referencing the graph YAML.

    Returns
    -------
    tuple[Path, Path, tuple[Path, Path]]
        Graph cache root, source GeoPackage, and the two processed artifacts.
    """
    graph_yaml = config.settings.paths.graph_config
    with graph_yaml.open(encoding="utf-8") as stream:
        graph_settings = yaml.safe_load(stream)

    dataset = graph_settings.get("dataset") if isinstance(graph_settings, dict) else None
    if not isinstance(dataset, dict):
        raise ValueError(f"Graph configuration requires a dataset mapping: {graph_yaml}")
    for name in ("root", "network_path"):
        if not isinstance(dataset.get(name), str) or not dataset[name].strip():
            raise ValueError(f"Graph configuration requires dataset.{name}: {graph_yaml}")

    root = _project_path(dataset["root"], config.project_root)
    network = _project_path(dataset["network_path"], config.project_root)
    cached_files = (
        root / "processed" / "road_graph.pt",
        root / "processed" / "preprocessor.joblib"
    )
    return root, network, cached_files


def graph_rebuild_sources(config: ResolvedExperimentConfig) -> tuple[Path, ...]:
    """Return extra source files needed only when rebuilding the graph.

    Parameters
    ----------
    config : ResolvedExperimentConfig
        Experiment settings locating the graph YAML.

    Returns
    -------
    tuple[Path, ...]
        Optional precomputed transition CSV inputs. An existing complete cache
        does not require these files to be present.
    """
    _, _, cached_files = graph_paths(config)
    if all(path.is_file() for path in cached_files):
        return ()

    with config.settings.paths.graph_config.open(encoding="utf-8") as stream:
        settings = yaml.safe_load(stream)
    transforms = (settings.get("graph") or {}).get("transforms") or {}
    transition = transforms.get("edge_transition_count") or {}
    if not transition.get("enabled", False):
        return ()
    match_csv = transition.get("match_csv")
    if not isinstance(match_csv, str) or not match_csv.strip():
        raise ValueError("graph.transforms.edge_transition_count.match_csv is required.")
    return (_project_path(match_csv, config.project_root),)


def prepare_experiment_graph(config: ResolvedExperimentConfig) -> None:
    """Build the road graph only when one or both cache files are absent.

    The underlying RoadGraphDataset owns both artifacts. It performs feature
    preprocessing and writes the graph and fitted sklearn transformer.

    Parameters
    ----------
    config : ResolvedExperimentConfig
        Experiment settings with the graph YAML and project root.

    Returns
    -------
    None
        Both cache files exist after preparation.
    """
    root, network, cached_files = graph_paths(config)
    if all(path.is_file() for path in cached_files):
        print(f"  Road graph cache already exists: {root / 'processed'}", flush=True)
        return

    if not network.is_file():
        raise FileNotFoundError(f"Road network required to build graph: {network}")

    # Load the graph implementation only during execution, never during planning
    # or --validate-only. Use exactly the same path resolution as graph loading.
    from urban_syn.graph.config import build_dataset_from_config

    graph_settings = resolve_graph_settings(config)
    sources = graph_rebuild_sources(config)
    missing_sources = [path for path in sources if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(
            f"Graph transform requires: {', '.join(map(str, missing_sources))}"
        )

    print(f"  Preparing road graph from: {network}", flush=True)
    build_dataset_from_config(graph_settings)

    missing = [path for path in cached_files if not path.is_file()]
    if missing:
        raise RuntimeError(f"Graph builder did not produce: {', '.join(map(str, missing))}")
    print(f"  Road graph cache ready: {root / 'processed'}", flush=True)
