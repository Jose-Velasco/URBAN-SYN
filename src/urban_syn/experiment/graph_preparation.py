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
    # or --validate-only. This also reuses the feature transforms from graph YAML.
    from urban_syn.graph.config import build_dataset_from_config, load_graph_config

    graph_settings = load_graph_config(config.settings.paths.graph_config)
    dataset = dict(graph_settings["dataset"])
    dataset["root"] = str(root)
    dataset["network_path"] = str(network)
    graph_settings["dataset"] = dataset

    # The optional transition-count transform reads a file directly. Resolve
    # it against the repository just like the network and cache locations.
    transforms = (graph_settings.get("graph") or {}).get("transforms") or {}
    transition = transforms.get("edge_transition_count") or {}
    if transition.get("enabled", False):
        sources = graph_rebuild_sources(config)
        missing_sources = [path for path in sources if not path.is_file()]
        if missing_sources:
            raise FileNotFoundError(
                f"Graph transform requires: {', '.join(map(str, missing_sources))}"
            )
        graph_settings["graph"]["transforms"]["edge_transition_count"] = {
            **transition, "match_csv": str(sources[0])
        }

    print(f"  Preparing road graph from: {network}", flush=True)
    build_dataset_from_config(graph_settings)

    missing = [path for path in cached_files if not path.is_file()]
    if missing:
        raise RuntimeError(f"Graph builder did not produce: {', '.join(map(str, missing))}")
    print(f"  Road graph cache ready: {root / 'processed'}", flush=True)
