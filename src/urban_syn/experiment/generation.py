"""Generate GPS trajectories from the trained joint model using experiment YAML.

This module reuses the existing DDPM sampler, checkpoint validation, road-ID
mapping, and GPS reconstruction. It does not implement a new generator.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from time import perf_counter

import torch

from urban_syn.diffusion.generate import (
    generate_gps_trajectories,
    load_generation_model,
    select_generation_lengths
)
from urban_syn.diffusion.reconstruction import load_road_geometries

from .config import ResolvedExperimentConfig
from .factories import build_sampler, load_experiment_graph, resolve_device
from .tracking import ExperimentTracker


logger = logging.getLogger(__name__)


def resolve_generation_checkpoint(
    config: ResolvedExperimentConfig,
    training_run_dir: Path | None = None
) -> Path:
    """Resolve the checkpoint explicitly or from the current pipeline run.

    Parameters
    ----------
    config : ResolvedExperimentConfig
        Experiment configuration with checkpoint selection and optional path.
    training_run_dir : Path | None, default=None
        Run directory returned by the training stage of this process.

    Returns
    -------
    Path
        Existing best or last checkpoint that belongs to the selected run.
    """
    generation = config.settings.generation
    if training_run_dir is not None:
        checkpoint = training_run_dir / "checkpoints" / f"{generation.checkpoint_choice}.pt"
    elif generation.checkpoint is not None:
        source = generation.checkpoint.expanduser()
        checkpoint = source if source.is_absolute() else config.project_root / source
    else:
        raise ValueError(
            "Standalone generation requires generation.checkpoint in experiment YAML. "
            "The 'all' target automatically uses the checkpoint produced by training."
        )
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Generation checkpoint not found: {checkpoint}")
    return checkpoint


def _generation_run_dir(checkpoint: Path) -> Path:
    """Find the experiment run directory that holds the trained checkpoint.

    Parameters
    ----------
    checkpoint : Path
        Source checkpoint under the run's checkpoints directory.

    Returns
    -------
    Path
        Experiment run directory for saving generated outputs.
    """
    if checkpoint.parent.name != "checkpoints":
        raise ValueError("Generation checkpoint must be inside an experiment checkpoints/ directory.")
    return checkpoint.parent.parent


def run_generation(
    config: ResolvedExperimentConfig,
    tracker: ExperimentTracker,
    training_run_dir: Path | None = None
) -> Path:
    """Generate GPS Parquet from the selected checkpoint and log artifacts.

    Parameters
    ----------
    config : ResolvedExperimentConfig
        Resolved graph, diffusion, generation, and tracking settings.
    tracker : ExperimentTracker
        Injected tracking backend; may be disabled.
    training_run_dir : Path | None, default=None
        Directory from training when executing the full pipeline.

    Returns
    -------
    Path
        Path of generated GPS observations in Parquet format.
    """
    settings = config.settings
    checkpoint = resolve_generation_checkpoint(config, training_run_dir)
    run_dir = _generation_run_dir(checkpoint)
    if settings.paths.length_distribution is None:
        raise ValueError("Generation requires paths.length_distribution.")

    device = resolve_device(settings.runtime.device)
    torch.manual_seed(settings.experiment.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings.experiment.seed)

    graph = load_experiment_graph(config, device)
    model = load_generation_model(checkpoint, graph, config.joint_settings, device)
    road_vocab = graph.edge_id.detach().cpu().long().clone()
    model.eval()
    with torch.inference_mode():
        road_embeddings = model.encode_roads(graph)

    # Cached embeddings make repeated full-graph encoder calls unnecessary.
    del graph
    del model.road_encoder

    network_path = settings.paths.road_network
    if network_path is None:
        raise ValueError("Generation requires paths.road_network for GPS reconstruction.")
    road_geometries = load_road_geometries(network_path, layer=settings.generation.road_layer)
    missing = set(road_vocab.tolist()) - set(road_geometries)
    if missing:
        raise ValueError(f"Road GeoPackage lacks {len(missing)} graph road IDs.")

    length_counts = select_generation_lengths(
        settings.paths.length_distribution,
        num_trajectories=settings.generation.num_trajectories,
        seed=settings.experiment.seed
    )
    sampler = build_sampler(settings.generation.sampler.name)
    start = perf_counter()
    generated = generate_gps_trajectories(
        model=model,
        road_vocab=road_vocab,
        road_embeddings=road_embeddings,
        road_geometries=road_geometries,
        length_counts=length_counts,
        sampler=sampler,
        batch_size=settings.generation.batch_size
    )
    elapsed_seconds = perf_counter() - start

    output = run_dir / "generated" / settings.generation.output_name
    output.parent.mkdir(parents=True, exist_ok=True)
    generated.to_parquet(output, index=False)

    info = torch.load(checkpoint, map_location="cpu", weights_only=True)
    manifest = {
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": int(info.get("epoch", -1)),
        "sampler": settings.generation.sampler.name,
        "generated_gps": str(output),
        "test_reference_gps": str(settings.paths.test_reference_gps) if settings.paths.test_reference_gps else None,
        "test_sequences": str(settings.paths.test_sequences) if settings.paths.test_sequences else None,
        "num_trajectories": int(generated["trajectory_id"].nunique()),
        "num_points": len(generated),
        "elapsed_seconds": elapsed_seconds
    }
    manifest_path = output.parent / "generation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    previous = info.get("tracking_reference")
    if previous is None and info.get("mlflow_run_id"):
        previous = {
            "provider": "mlflow",
            "run_id": info["mlflow_run_id"],
            "uri": info.get("tracking_uri")
        }
    with tracker.start_run(previous=previous) as run:
        run.set_tags({"generation_sampler": settings.generation.sampler.name})
        run.log_metrics({
            "generation/trajectories": float(manifest["num_trajectories"]),
            "generation/points": float(manifest["num_points"]),
            "generation/duration_seconds": float(elapsed_seconds)
        }, step=int(info.get("global_step", 0)))
        run.log_artifact(manifest_path, artifact_path="generated")
        if settings.tracking.artifacts.log_generated_trajectories:
            run.log_artifact(output, artifact_path="generated")

    logger.info("Generated %d trajectories (%d GPS points): %s", manifest["num_trajectories"], len(generated), output)
    return output


def export_evaluation_handoff(config: ResolvedExperimentConfig, generated_path: Path) -> Path:
    """Create the file-path handoff for the team's existing evaluator.

    Parameters
    ----------
    config : ResolvedExperimentConfig
        Configuration specifying the held-out GPS reference data.
    generated_path : Path
        Generated GPS Parquet output from the generation stage.

    Returns
    -------
    Path
        JSON file identifying generated and held-out GPS inputs.
    """
    reference = config.settings.paths.test_reference_gps
    if reference is None or not reference.is_file():
        raise FileNotFoundError("A valid paths.test_reference_gps is required for evaluator handoff.")
    if not generated_path.is_file():
        raise FileNotFoundError(f"Generated GPS trajectories not found: {generated_path}")
    output = generated_path.parent / "evaluation_handoff.json"
    output.write_text(json.dumps({
        "generated_gps": str(generated_path),
        "test_reference_gps": str(reference),
        "status": "Input paths identified; confirm schema with the shared evaluator"
    }, indent=2), encoding="utf-8")
    return output
