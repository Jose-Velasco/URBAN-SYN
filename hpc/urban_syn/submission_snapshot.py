"""Save and verify per-submission YAML files for Hydra-backed Slurm jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import uuid
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from resolve_submission import Submission


def _sha256(path: Path) -> str:
    """Hash a saved file so queued jobs can detect unexpected modification."""
    digest = hashlib.sha256()
    with path.open("rb") as content:
        for chunk in iter(lambda: content.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _validate_encoder_checkpoint(path: Path, graph: dict, objective: str) -> None:
    """Check trusted checkpoint metadata before allocating a GPU job."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Checkpoint preflight requires PyTorch on the submission Python environment.") from error
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError(f"Unsupported pretrained encoder checkpoint: {path}")
    metadata = payload.get("model_metadata")
    if not isinstance(metadata, dict) or not isinstance(payload.get("encoder_state_dict"), dict):
        raise ValueError(f"Missing encoder state/metadata in {path}")
    expected = graph["model"]
    saved_gnn = metadata.get("gnn", {})
    current_gnn = expected["gnn"]
    if (not isinstance(saved_gnn, dict) or
            saved_gnn.get("architecture") != current_gnn.get("architecture") or
            any(saved_gnn.get(key) != current_gnn.get(key)
                for key in ("hidden_dim", "output_dim", "num_layers"))):
        raise ValueError("Pretrained encoder architecture differs from selected graph YAML.")
    if (metadata.get("selected_node_features") != expected.get("node_features") or
            metadata.get("selected_edge_features") != expected.get("edge_features")):
        raise ValueError("Pretrained encoder feature groups differ from selected graph YAML.")
    if metadata.get("objective") != objective:
        raise ValueError("Pretrained scorer objective differs from selected BCE/BPR experiment.")


def _write_yaml(path: Path, values: dict) -> None:
    import yaml

    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")


def save_submission_snapshot(
    root: Path, selected: Submission, source_job_id: str = "", selected_encoder: str = ""
) -> Submission:
    """Freeze a composed experiment, its graph, and any newly submitted pretraining source.

    Args:
        root: Repository checkout containing the Hydra study.
        selected: Read-only submission decision from resolve_submission().
        source_job_id: Existing Slurm pretraining job ID, if supplied.
        selected_encoder: Manually chosen encoder checkpoint path, if supplied.

    Returns:
        Selection pointing at newly created, independently named YAML files.
    """
    if selected.composed_config is None:
        return selected
    import yaml

    root = Path(root).resolve()
    parent = root / "outputs/experiments/submissions"
    parent.mkdir(parents=True, exist_ok=True)
    # Random ID means concurrent submitters never overwrite each other's files.
    folder = parent / f"{selected.experiment_key}-{uuid.uuid4().hex[:12]}"
    folder.mkdir(mode=0o755)
    try:
        relative = folder.relative_to(root).as_posix()
        raw = selected.composed_config
        original_graph = (root / raw["paths"]["graph_config"]).resolve()
        if not original_graph.is_relative_to(root):
            raise ValueError("Road graph YAML must be within the repository.")
        graph = yaml.safe_load(original_graph.read_text(encoding="utf-8"))
        if selected.encoder_checkpoint or selected_encoder:
            raw_checkpoint = selected_encoder or selected.encoder_checkpoint
            checkpoint = (root / raw_checkpoint.removeprefix("/workspace/")
                          if raw_checkpoint.startswith("/workspace/") else Path(raw_checkpoint))
            if not checkpoint.is_absolute():
                checkpoint = root / checkpoint
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Encoder checkpoint does not exist: {checkpoint}")
            objective = raw.get("tracking", {}).get("tags", {}).get("pretraining_objective")
            if objective not in ("bce", "bpr"):
                raise ValueError("A pretrained source must select BCE or BPR.")
            _validate_encoder_checkpoint(checkpoint, graph, objective)
        _write_yaml(folder / "graph.yaml", graph)
        prepared = {**raw, "paths": {**raw["paths"], "graph_config": f"{relative}/graph.yaml"}}
        _write_yaml(folder / "experiment.yaml", prepared)
        files = ["experiment.yaml", "graph.yaml"]
        pretrain_file = ""
        if selected.pretrain_config:
            original = yaml.safe_load(Path(selected.pretrain_config).read_text(encoding="utf-8"))
            original["paths"] = {**original["paths"], "graph_config": f"{relative}/graph.yaml"}
            _write_yaml(folder / "pretrain.yaml", original)
            pretrain_file = str(folder / "pretrain.yaml")
            files.append("pretrain.yaml")
        manifest = {
            "format_version": 1,
            "study": selected.study_name,
            "experiment": selected.experiment_key,
            "source_study": selected.source_study_name or None,
            "source_experiment": selected.pretrained_from or None,
            "source_job_id": source_job_id or None,
            "encoder_checkpoint": selected_encoder or selected.encoder_checkpoint or None,
            "git_revision": _git_revision(root),
            "files": {name: _sha256(folder / name) for name in files}
        }
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        for name in [*files, "manifest.json"]:
            (folder / name).chmod(0o444)
        return replace(selected, config=folder / "experiment.yaml", pretrain_config=pretrain_file)
    except BaseException:
        # A partial snapshot must not be submitted.
        for path in folder.iterdir():
            path.chmod(0o644)
            path.unlink()
        folder.rmdir()
        raise


def verify_snapshot(config: Path) -> None:
    """Reject modified or incomplete snapshots before the training process starts."""
    config = Path(config).resolve()
    if config.name not in {"experiment.yaml", "pretrain.yaml"}:
        raise ValueError("Snapshot verification expects an experiment or pretrain YAML.")
    folder = config.parent
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1 or not isinstance(manifest.get("files"), dict):
        raise ValueError("Invalid immutable submission manifest.")
    if config.name not in manifest["files"]:
        raise ValueError("The submitted YAML is not recorded in its snapshot manifest.")
    if not {"graph.yaml", "experiment.yaml"}.issubset(manifest["files"]):
        raise ValueError("Snapshot must contain the experiment and road graph.")
    for name, checksum in manifest["files"].items():
        if name not in {"experiment.yaml", "graph.yaml", "pretrain.yaml"}:
            raise ValueError(f"Unexpected snapshot file: {name}")
        if not re.fullmatch(r"[a-f0-9]{64}", str(checksum)) or _sha256(folder / name) != checksum:
            raise ValueError(f"Submission snapshot changed or is missing: {folder / name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path, required=True, help="Saved experiment/pretraining YAML")
    args = parser.parse_args()
    try:
        verify_snapshot(args.verify)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        parser.exit(2, f"ERROR: {error}\n")
    print(f"Verified immutable submission: {args.verify}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
