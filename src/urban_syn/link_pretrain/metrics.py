"""Append-only local metrics and analysis-friendly Parquet export.

JSONL remains the recovery-friendly source of local training history. Parquet
is generated at run completion or when a handled failure ends the run.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path


def append_metric_history(
    path: Path,
    epoch: int,
    global_step: int,
    metrics: Mapping[str, float]
) -> None:
    """Append one training or evaluation event to local metrics history.

    Args:
        path: Append-only JSONL path; parent directories are created.
        epoch: One-based completed epoch, or zero for initial diagnostics.
        global_step: Completed optimizer update count.
        metrics: Named scalar measurements such as ``train/loss`` or
            ``validation/mrr``. Values must be JSON-serializable floats.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"epoch": epoch, "global_step": global_step, **metrics}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")


def export_metric_history(jsonl_path: Path, parquet_path: Path) -> Path | None:
    """Export all recorded metrics to a tidy Parquet table for analysis.

    Args:
        jsonl_path: Historical epoch/test events recorded during this run.
        parquet_path: Output path for the final ``.parquet`` dataset.

    Returns:
        Written Parquet path, or None when no history has been recorded.

    Raises:
        ValueError: If a complete JSONL record is invalid.
        ImportError: If Pandas or a Parquet engine (such as PyArrow) is absent.
    """
    if not jsonl_path.is_file():
        return None

    records: list[dict[str, float | int | str]] = []
    with jsonl_path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            event = json.loads(line)
            epoch = int(event["epoch"])
            step = int(event["global_step"])
            for name, value in event.items():
                if name in ("epoch", "global_step"):
                    continue
                # Older runs used unprefixed keys (for example, val_loss).
                # New records use the same metric names as MLflow.
                if "/" in name:
                    split, metric = name.split("/", maxsplit=1)
                elif name.startswith("val_"):
                    split, metric = "validation", name.removeprefix("val_")
                elif name.startswith("test_"):
                    split, metric = "test", name.removeprefix("test_")
                else:
                    split, metric = "train", name.removeprefix("train_")
                records.append({
                    "epoch": epoch,
                    "global_step": step,
                    "split": split,
                    "metric": metric,
                    "value": float(value)
                })

    if not records:
        return None
    import pandas as pd

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records(
        records, columns=["epoch", "global_step", "split", "metric", "value"]
    )
    frame.to_parquet(parquet_path, index=False)
    return parquet_path
