"""Record the encoder selected by a completed study pretraining job.

A study stores one small reference file per pretraining experiment. Each reference
points to an immutable, uniquely named training run; checkpoints are not copied.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _safe_name(name: str) -> str:
    """Require a directory-safe study or experiment name."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError(f"Invalid study/experiment name: {name!r}")
    return name


def _in_project(value: str, root: Path) -> Path:
    """Translate container paths to their host equivalents and check containment."""
    if value.startswith("/workspace/"):
        path = root / value.removeprefix("/workspace/")
    else:
        supplied = Path(value).expanduser()
        path = supplied if supplied.is_absolute() else root / supplied
    path = path.resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Checkpoint result references a file outside the project: {path}")
    return path


def publish_encoder(
    root: Path,
    study_name: str,
    experiment_key: str,
    result_file: Path | None,
    job_id: str,
    run_dir: str | None = None,
) -> Path:
    """Publish a successful pretraining run without searching for recent files.

    Args:
        root: Host repository root; the same directory bound to /workspace.
        study_name: Name declared in study.yaml.
        experiment_key: Key of the producing pretraining experiment.
        result_file: JSON written by a completed pretraining pipeline, if set.
        job_id: Slurm job ID recorded for traceability.
        run_dir: Existing successful run to select manually instead of a result file.

    Returns:
        Path to the updated study reference JSON.

    Raises:
        ValueError: If the result does not identify a valid checkpoint and run.
        FileNotFoundError: If a referenced checkpoint is absent.
    """
    root = root.resolve()
    study_name = _safe_name(study_name)
    experiment_key = _safe_name(experiment_key)
    if (result_file is None) == (run_dir is None):
        raise ValueError("Provide exactly one of result_file or run_dir.")
    if run_dir is not None:
        # Also support a previously completed run, such as the first HPC
        # smoke test performed before automatic study references existed.
        requested = _in_project(run_dir, root)
        result = {
            "format_version": 1,
            "run_dir": str(requested),
            "encoder_checkpoint": str(requested / "checkpoints/encoder.pt"),
            "best_checkpoint": str(requested / "checkpoints/best.pt"),
        }
    else:
        assert result_file is not None
        result = json.loads(result_file.read_text(encoding="utf-8"))
        if not isinstance(result, dict) or result.get("format_version") != 1:
            raise ValueError(f"Invalid pretraining result file: {result_file}")

    run_dir = _in_project(str(result["run_dir"]), root)
    encoder = _in_project(str(result["encoder_checkpoint"]), root)
    best = _in_project(str(result["best_checkpoint"]), root)
    expected_dir = run_dir / "checkpoints"
    if encoder != expected_dir / "encoder.pt" or best != expected_dir / "best.pt":
        raise ValueError("Pretraining result must reference encoder.pt and best.pt in its run directory.")
    for checkpoint in (encoder, best):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Completed pretraining checkpoint not found: {checkpoint}")

    # Repository-relative paths remain valid if the checkout is moved or bound
    # to /workspace. Subsequent joint submissions resolve them against their root.
    reference = {
        "format_version": 1,
        "study_name": study_name,
        "experiment_key": experiment_key,
        "run_id": run_dir.name,
        "run_dir": run_dir.relative_to(root).as_posix(),
        "encoder_checkpoint": encoder.relative_to(root).as_posix(),
        "best_checkpoint": best.relative_to(root).as_posix(),
        "slurm_job_id": str(job_id),
        "published_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    selected = root / "outputs/experiments/studies" / study_name / experiment_key / "selected_encoder.json"
    selected.parent.mkdir(parents=True, exist_ok=True)
    temporary = selected.with_name(f".{selected.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(reference, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, selected)
    finally:
        temporary.unlink(missing_ok=True)
    return selected


def main() -> int:
    """Publish a pretraining result after the full requested job succeeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--experiment-key", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--result-file", type=Path)
    source.add_argument("--run-dir", help="Manually select a previously completed pretraining run")
    parser.add_argument("--job-id", default="manual")
    args = parser.parse_args()
    try:
        selected = publish_encoder(
            args.project_root, args.study_name, args.experiment_key,
            args.result_file, args.job_id, run_dir=args.run_dir
        )
    except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as error:
        parser.exit(1, f"ERROR: Cannot publish pretraining checkpoint: {error}\n")
    print(f"Selected pretrained encoder: {selected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
