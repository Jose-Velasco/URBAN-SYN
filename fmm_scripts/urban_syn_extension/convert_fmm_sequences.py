"""Prepare Diff-RNTraj road sequences from shared, split FMM results.

The output retains shared road ``edge_id`` values, including the valid ID 0.
Mapping to PyG model road IDs happens after the graph has been built.

Read train/test FMM match files produced by the shared post-match splitter.
The original FMM CSV files are never changed.

example usage:

uv run python -m urban_syn_extension.convert_fmm_sequences \
  --train_fmm_matches ./outputs/nyc_15s_enriched_moves/nyc_15s_train_fmm_matches.csv \
  --test_fmm_matches ./outputs/nyc_15s_enriched_moves/nyc_15s_test_fmm_matches.csv \
  --output_dir ./outputs/urban_syn_ready \
  --prefix nyc_15s_enriched_moves \
  --log_file ./outputs/logs/convert_fmm_sequences.log
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd

from utils import setup_logging

logger = logging.getLogger(__name__)

_INT_PATTERN = re.compile(r"-?\d+")
_FLOAT_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def parse_args() -> argparse.Namespace:
    """Parse the split FMM inputs and destination for prepared sequences."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_fmm_matches", required=True, type=Path)
    parser.add_argument("--test_fmm_matches", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--prefix", default="nyc_15s")
    parser.add_argument("--log_file", type=Path, default=None)
    return parser.parse_args()


def _read_fmm_matches(path: Path) -> pd.DataFrame:
    """Read FMM results; detect the delimiter from the header only."""
    with path.open("r", encoding="utf-8-sig") as source:
        header = source.readline().rstrip("\r\n")

    delimiter = next((sep for sep in (";", "\t", "|") if sep in header), ",")
    matches = pd.read_csv(path, sep=delimiter)
    
    required = {"opath", "offset", "length"}
    if not required.issubset(matches.columns):
        raise ValueError(f"{path} is missing FMM columns: {sorted(required - set(matches))}")
    if "id" not in matches and "fmm_id" not in matches:
        raise ValueError(f"{path} has no FMM trajectory identifier ('id' or 'fmm_id').")
    
    return matches


def _parse_int_sequence(value: object) -> list[int]:
    """Parse FMM's observation-level road IDs, preserving repeated IDs."""
    if pd.isna(value):
        return []
    return [int(token) for token in _INT_PATTERN.findall(str(value))]


def _parse_float_sequence(value: object) -> list[float]:
    """Parse FMM's offset or road-length values for GPS observations."""
    if pd.isna(value):
        return []
    return [float(token) for token in _FLOAT_PATTERN.findall(str(value))]


def encode_rates(
    edge_ids: list[int],
    offsets: list[float],
    lengths: list[float],
) -> list[float]:
    """Encode position rates using the original Diff-RNTraj convention.

    On a new road, rate = offset / length. On the same road, rate is the
    fraction of the *remaining* segment traversed since the previous point.
    Backward GPS jitter is clamped; repeated road IDs are not discarded.

    Args:
        edge_ids: Shared road IDs from ``opath``, one per GPS observation.
        offsets: Matched distances from the beginning of each road.
        lengths: Road lengths in the same units as ``offsets``.

    Returns:
        Rates in [0, 1], aligned one-to-one with ``edge_ids``.
    """
    rates: list[float] = []
    previous_edge_id: int | None = None
    previous_position = 0.0

    for edge_id, offset, length in zip(edge_ids, offsets, lengths, strict=True):
        if not math.isfinite(offset) or not math.isfinite(length) or length <= 0:
            position = 0.0
        else:
            # (offset / length) is the percent of the road traversed
            # can only 0-1.0 "percent" of the road completed/traversed
            position = min(1.0, max(0.0, offset / length))

        # true when "we" are still on the same road
        if edge_id == previous_edge_id:
            # "we" are still on the same road but we have already completed/traversed "100%""
            if previous_position >= 1.0:
                rate = 0.0
            # we are on the same road but have not traversed it 100% yet
            else:
                # let d = (position - previous_position) is the distance traversed since last gps point on the same road. its the delta distance relative to the total remaining road to traverse
                # let t_r = (1 - previous_position) since we already traversed some of the road before we need to deduct it from the total remaining road to traverse
                # thus d/t_r tells us what percentage of the total remaining road that has been traversed 
                rate = min(1.0, max(0.0, (position - previous_position) / (1 - previous_position)))

            # Update the decoded position, not the raw (possibly backward) offset.
            # (1.0 - previous_position) is total road traverse since last "step" aka previous_position
            # ( * rate) gives is the percentage of the total remaining road that has been traversed on the current "step" aka position
            # [let n_d = (1.0 - previous_position) * rate] gives the distance traversed since previous_position
            # (previous_position += ) adds the distance n_d that was traversed since previous_position
            # thus previous_position holds the total distance traversed of the current road at this current "step"
            previous_position += (1.0 - previous_position) * rate
            # "we" are still on the same road but we have already completed/traversed aka (previous_position >= 1.0)
            # then rate is zero and we stay in the same "position" distance
        else:
            # we are on a new road thus position tells us the percent of the total new road traversed
            rate = position
            # what we have traversed is where we will be at next time if we end up in the same road
            previous_position = position

        rates.append(float(rate))
        previous_edge_id = edge_id

    return rates


def _prepare_split(path: Path) -> tuple[pd.DataFrame, dict]:
    """Convert one split while recording why unusable FMM rows are skipped. Basically validating with logging metadata

    Returns:
        Prepared sequences and counts of accepted/skipped FMM rows.
    """
    matches = _read_fmm_matches(path)
    id_column = "id" if "id" in matches else "fmm_id"

    prepared: list[dict] = []
    skipped: Counter[str] = Counter()

    for row in matches.itertuples(index=False):
        record = row._asdict()
        try:
            raw_id = float(record[id_column])
            if not math.isfinite(raw_id) or not raw_id.is_integer() or raw_id <= 0:
                skipped["invalid_fmm_id"] += 1
                continue
            
            fmm_id = int(raw_id)
        except (TypeError, ValueError, OverflowError):
            skipped["invalid_fmm_id"] += 1
            continue

        edge_ids = _parse_int_sequence(record["opath"])
        offsets = _parse_float_sequence(record["offset"])
        lengths = _parse_float_sequence(record["length"])

        if not edge_ids:
            skipped["empty_opath"] += 1
            continue
        if not (len(edge_ids) == len(offsets) == len(lengths)):
            skipped["observation_count_mismatch"] += 1
            continue
        if any(edge_id < 0 for edge_id in edge_ids):
            skipped["negative_edge_id"] += 1
            continue

        prepared.append({
            "fmm_id": fmm_id,
            "edge_ids": edge_ids,
            "rates": encode_rates(edge_ids, offsets, lengths),
            "num_points": len(edge_ids),
        })

    frame = pd.DataFrame(prepared, columns=["fmm_id", "edge_ids", "rates", "num_points"])
    if frame["fmm_id"].duplicated().any():
        raise ValueError(f"{path} contains repeated valid FMM trajectory IDs.")

    summary = {
        "fmm_rows": len(matches),
        "prepared_trajectories": len(frame),
        "prepared_observations": int(frame["num_points"].sum()),
        "skipped": dict(sorted(skipped.items())),
    }
    return frame, summary


def prepare_fmm_sequences(
    train_fmm_matches: Path,
    test_fmm_matches: Path,
    output_dir: Path,
    prefix: str,
) -> dict:
    """Write edge-ID-based train/test Parquet files and a conversion summary.

    Args:
        prefix: Filename prefix identifying the dataset/input variant.

    Returns:
        Per-split counts and rejection reasons for the preparation run.
    """
    if not prefix or Path(prefix).name != prefix:
        raise ValueError("prefix must be a nonempty filename prefix.")

    outputs = {}
    summary = {}
    for split, path in (("train", train_fmm_matches), ("test", test_fmm_matches)):
        outputs[split], summary[split] = _prepare_split(path)

    # Prevent accidental mixing of the same FMM trajectory into both splits
    common_ids = set(outputs["train"]["fmm_id"]) & set(outputs["test"]["fmm_id"])
    if common_ids:
        raise ValueError(f"Train and test contain {len(common_ids)} overlapping fmm_id values.")

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, frame in outputs.items():
        path = output_dir / f"{prefix}_{split}_prepared_road_sequences.parquet"
        frame.to_parquet(path, index=False)
        logger.info("Wrote %d %s road sequences: %s", len(frame), split, path)

        # Index = sequence length, value = number of trajectories of that length
        # Computed from accepted FMM sequences, independently of the road graph
        lengths = frame["num_points"].to_numpy(dtype=np.int64)
        length_distribution = (
            np.bincount(lengths)
            if lengths.size
            else np.zeros(1, dtype=np.int64)
        )

        distribution_path = output_dir / f"{prefix}_{split}_length_distribution.npy"
        np.save(distribution_path, length_distribution)
        logger.info("Saved %s length distribution: %s", split, distribution_path)

    summary_path = output_dir / f"{prefix}_sequence_preparation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", summary_path)
    return summary


def main() -> None:
    """Prepare road sequences without building a graph or remapping road IDs."""
    args = parse_args()
    setup_logging(args.log_file)
    summary = prepare_fmm_sequences(
        train_fmm_matches=args.train_fmm_matches,
        test_fmm_matches=args.test_fmm_matches,
        output_dir=args.output_dir,
        prefix=args.prefix,
    )
    logger.info("Sequence preparation summary: %s", summary)


if __name__ == "__main__":
    main()
