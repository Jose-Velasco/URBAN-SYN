"""Run a named Hydra experiment locally, including its optional link pretraining.

This entry point orchestrates existing pipeline CLIs. It never constructs a model,
implements training, or submits a Slurm job.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import yaml

from resolve_submission import Submission, resolve_submission
from config_validation import validate_local_encoder, validate_local_request


def _project_path(root: Path, value: str | Path) -> Path:
    """Resolve a repository-relative path without allowing it to leave the checkout."""
    path = Path(value).expanduser()
    resolved = (path if path.is_absolute() else root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Local experiment paths must stay inside the repository: {resolved}")
    return resolved


def _read_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping used by an existing pipeline."""
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return content


def _write_yaml(path: Path, values: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")


def _local_tracking(root: Path, settings: dict[str, Any] | None) -> dict[str, Any]:
    """Use one local MLflow database for pretraining and joint runs.

    Tracking storage is independent of each run's checkpoint/output directory.
    """
    store = root / "outputs/experiments/mlflow"
    return {
        **(settings or {}),
        "store_root": store.relative_to(root).as_posix(),
        "tracking_uri": f"sqlite:///{(store / 'tracking.db').as_posix()}",
        "storage_scope": "shared"
    }


def _prepare_configs(
    root: Path, folder: Path, selected: Submission, stage: str
) -> tuple[Path | None, Path | None]:
    """Save isolated graph, joint, and optional pretraining configurations.

    Args:
        root: Repository root.
        folder: Fresh per-run output directory.
        selected: Validated study selection and Hydra-composed configuration.
        stage: Local workflow stage.

    Returns:
        Paths to joint and pretraining YAML files, respectively.
    """
    source = selected.composed_config if selected.pipeline == "joint" else _read_yaml(selected.config)
    if source is None:
        raise ValueError("Local joint runs require a Hydra preset in the study catalog.")

    graph = _read_yaml(_project_path(root, source["paths"]["graph_config"]))
    _write_yaml(folder / "graph.yaml", graph)
    graph_path = (folder / "graph.yaml").relative_to(root).as_posix()

    joint_path = None
    pretrain_path = None
    if selected.pipeline == "joint":
        joint = {
            **source,
            "paths": {
                **source["paths"],
                "graph_config": graph_path,
                "output_root": (folder / "outputs").relative_to(root).as_posix()
            },
            "tracking": _local_tracking(root, source.get("tracking"))
        }
        if joint["tracking"]["tags"].get("profile") == "smoke":
            encoder = joint["model"].get("road_encoder", {})
            warmup = encoder.get("gradient_warmup", {})
            if warmup.get("enabled"):
                budget = joint["training"]["epochs"] * joint["training"]["max_batches"]
                original_steps = warmup["warmup_steps"]
                if original_steps > budget:
                    joint["model"] = {
                        **joint["model"],
                        "road_encoder": {
                            **encoder,
                            "gradient_warmup": {**warmup, "warmup_steps": budget}
                        }
                    }
                    print(f"[local] Smoke warmup: {original_steps} -> {budget} optimizer steps", flush=True)
        joint_path = folder / "joint.yaml"
        _write_yaml(joint_path, joint)
    else:
        joint = None

    if selected.pipeline == "link_pretrain" or (selected.pretrain_config and stage in ("all", "pretrain")):
        pretrain_source = selected.config if selected.pipeline == "link_pretrain" else Path(selected.pretrain_config)
        pretrain = _read_yaml(pretrain_source)
        pretrain["paths"] = {**pretrain["paths"], "graph_config": graph_path}
        pretrain["tracking"] = _local_tracking(root, pretrain.get("tracking"))
        if joint is not None and joint["tracking"]["tags"].get("profile") == "smoke":
            # One shared smoke budget, selected through Hydra, for both model phases.
            for key in ("epochs", "max_batches", "batch_size"):
                pretrain["training"][key] = joint["training"][key]
        pretrain_path = folder / "pretrain.yaml"
        _write_yaml(pretrain_path, pretrain)

    return joint_path, pretrain_path


def _run_pretraining(root: Path, folder: Path, config: Path) -> Path:
    """Run link pretraining and return its published encoder checkpoint."""
    output = folder / "pretraining"
    result = folder / "pretraining_result.json"
    command = [
        sys.executable, "-m", "urban_syn.link_pretrain", "all",
        "--config", str(config), "--project-root", str(root),
        "--run-dir", str(output), "--result-file", str(result)
    ]
    print("\n[local] Pretraining road GNN and connectivity scorer", flush=True)
    subprocess.run(command, cwd=root, check=True)
    published = json.loads(result.read_text(encoding="utf-8"))
    if published.get("format_version") != 1:
        raise ValueError("Unsupported pretraining result format.")
    encoder = Path(published["encoder_checkpoint"]).resolve()
    scorer = Path(published["best_checkpoint"]).resolve()
    if encoder != output / "checkpoints/encoder.pt" or scorer != output / "checkpoints/best.pt":
        raise ValueError("Pretraining result refers to an unexpected checkpoint location.")
    if not encoder.is_file() or not scorer.is_file():
        raise FileNotFoundError("Pretraining did not publish both encoder.pt and best.pt.")
    return encoder


def _execute(
    root: Path, experiment: str, study: str, stage: str,
    encoder_checkpoint: str | None, output_dir: str | None, dry_run: bool
) -> Path | None:
    """Orchestrate selected pipelines without duplicating their training logic."""
    selected = resolve_submission(root, study=study, experiment_key=experiment, pending_pretrain=True)
    validate_local_request(selected, stage, encoder_checkpoint)

    if dry_run:
        print(f"Experiment: {experiment} ({selected.pipeline})")
        if selected.pretrain_config and stage in ("all", "pretrain") and not encoder_checkpoint:
            print(f"Pretraining: {selected.pretrained_from} ({selected.pretrain_config})")
        if selected.pipeline == "joint" and stage != "pretrain":
            print(f"Joint training + generation: {experiment}")
        print("Dry run: no models trained, files written, or jobs submitted.")
        return None

    checkpoint = validate_local_encoder(root, encoder_checkpoint) if encoder_checkpoint else None
    if checkpoint is not None:
        # Reuse the same compatibility check used by HPC submission snapshots.
        from submission_snapshot import _validate_encoder_checkpoint
        tags = selected.composed_config["tracking"]["tags"]
        graph = _read_yaml(_project_path(root, selected.composed_config["paths"]["graph_config"]))
        _validate_encoder_checkpoint(checkpoint, graph, tags["pretraining_objective"])

    folder = (_project_path(root, output_dir) if output_dir else
              root / "outputs/experiments/local" / f"{experiment}-{uuid.uuid4().hex[:12]}")
    folder.mkdir(parents=True, exist_ok=False)
    joint_config, pretrain_config = _prepare_configs(root, folder, selected, stage)
    print(f"[local] Experiment: {experiment}\n[local] Output: {folder}", flush=True)

    if selected.pipeline == "link_pretrain" or (selected.pretrained_from and stage in ("all", "pretrain") and checkpoint is None):
        if pretrain_config is None:
            raise ValueError("The selected experiment has no pretraining configuration.")
        checkpoint = _run_pretraining(root, folder, pretrain_config)

    if selected.pipeline == "joint" and stage != "pretrain":
        if joint_config is None:
            raise ValueError("The selected experiment has no joint configuration.")
        command = [
            sys.executable, "-m", "urban_syn.experiment", "all",
            "--config", str(joint_config), "--project-root", str(root)
        ]
        if checkpoint is not None:
            command.extend(["--encoder-checkpoint", str(checkpoint)])
        print("\n[local] Joint GNN/diffusion training and trajectory generation", flush=True)
        subprocess.run(command, cwd=root, check=True)
    print(f"\n[local] Completed: {folder}", flush=True)
    return folder


def main() -> int:
    """Select an experiment from the same Hydra study used by Slurm."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="Named experiment from study.yaml")
    parser.add_argument("--study", default="configs/experiments/connectivity/study.yaml")
    parser.add_argument("--stage", choices=("all", "pretrain", "joint"), default="all")
    parser.add_argument("--encoder-checkpoint", help="Reuse a local pretraining run's encoder.pt")
    parser.add_argument("--output-dir", help="New repository-local output folder (must not already exist)")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--dry-run", action="store_true", help="Show the planned pipeline only")
    args = parser.parse_args()
    try:
        _execute(
            args.project_root.resolve(), args.experiment, args.study, args.stage,
            args.encoder_checkpoint, args.output_dir, args.dry_run
        )
    except (ValueError, RuntimeError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"ERROR: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
