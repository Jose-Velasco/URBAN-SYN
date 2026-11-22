"""Split one shared FMM result into train/test files by original trip identity.

Example 1. if split_identity_columns = [uid, tid] then each trajectory and all of its gps points will
be put under the same split

Example 2. split_identity_columns = [uid, tid, move_id, piece] using the 15s rule of the urban_syn_extension then
each tid trajectory was chunked and the piece is an integer that tells which chunk a gps point, of its tid trajectory total chunks, it is in.
This means that this split_identity_columns will keep trajectory chunks in the same split but a single trajectory is made of multiple chunks
thus it chunks could in ether train or test split. 


The raw FMM output is never modified. FMM can emit trajectory ID 0 when its
candidate search fails, so ID 0 (and other unresolvable IDs) are quarantined
rather than assigned to either split. Road edge_id=0 remains a valid road ID.

Run from the fmm_scripts directory after one execution of FMM.

Example usage:

uv run split_fmm_matches.py \
  --config ./config/nyc.yaml \
  --fmm_matches ./outputs/nyc_15s_fmm_match.csv \
  --trip_id_map ./data/nyc_15s_fmm_trip_id_map.csv \
  --test_trips ./data/nyc_test.csv \
  --output_dir ./outputs/nyc_15s_enriched_moves \
  --prefix nyc_15s \
  --log_file ./outputs/logs/split_fmm_matches.log
"""

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd

from utils import load_config, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse paths and the source of train/test membership."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--fmm_matches", required=True, type=Path,
                        help="Complete, unmodified FMM output CSV.")
    parser.add_argument("--trip_id_map", required=True, type=Path,
                        help="CSV produced by prepare_fmm_gps.py (fmm_id + original IDs).")

    split_source = parser.add_mutually_exclusive_group(required=True)
    split_source.add_argument(
        "--test_trips", type=Path,
        help="CSV of test (uid, tid) pairs; all other prepared trips are train.",
    )
    split_source.add_argument(
        "--split_assignments", type=Path,
        help="CSV assigning each original trajectory to a train/test split.",
    )

    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--prefix", default="nyc",
                        help="Output prefix, e.g. nyc_original or nyc_15s.")
    parser.add_argument("--log_file", type=Path, default=None)
    return parser.parse_args()


def _check_identity_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    """Ensure the configured original-trajectory identity is usable."""
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("split_identity_columns must contain unique column names.")
    
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing identity columns: {sorted(missing)}")
    if df[columns].isna().any().any():
        raise ValueError(f"{name} contains null trajectory identities.")


def _read_identities(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read identity columns as strings, preserving textual trip IDs."""
    return pd.read_csv(path, dtype={column: "string" for column in columns})


def _read_trip_map(path: Path, identity_columns: list[str]) -> pd.DataFrame:
    """Load the one-based FMM-ID mapping created before map matching."""
    trip_map = _read_identities(path, identity_columns)
    _check_identity_columns(trip_map, identity_columns, "Trip-ID map")
    
    if "fmm_id" not in trip_map.columns:
        raise ValueError("Trip-ID map is missing 'fmm_id'.")

    ids = pd.to_numeric(trip_map["fmm_id"], errors="raise")
    if ids.isna().any() or not ids.ge(1).all() or not ids.eq(ids.round()).all():
        raise ValueError("Trip-ID map requires positive integer fmm_id values.")
    
    trip_map["fmm_id"] = ids.astype("int64")
    if trip_map["fmm_id"].duplicated().any():
        raise ValueError("Trip-ID map contains duplicate fmm_id values.")
    return trip_map


def _assign_from_test_trips(
    trip_map: pd.DataFrame, test_csv: Path, identity_columns: list[str]
) -> pd.DataFrame:
    """Return trip mappings labeled using a CSV of held-out test identities."""
    test_trips = _read_identities(test_csv, identity_columns)
    _check_identity_columns(test_trips, identity_columns, "Test-trip CSV")
    test_trips = test_trips[identity_columns].drop_duplicates()

    assigned = trip_map.merge(
        test_trips.assign(_is_test=True), on=identity_columns,
        how="left", validate="many_to_one",
    )

    assigned["split"] = "train"
    assigned.loc[assigned["_is_test"].eq(True), "split"] = "test"
    assigned = assigned.drop(columns="_is_test")

    matched_test_trips = assigned.loc[
        assigned["split"].eq("test"), identity_columns
    ].drop_duplicates()

    if matched_test_trips.empty:
        raise ValueError(
            "None of the test-trip identities match the FMM trip-ID map. "
            "Check dataset variant, identity columns, and test CSV."
        )
    if len(matched_test_trips) < len(test_trips):
        logger.warning(
            "%d test identities are absent from the prepared GPS dataset.",
            len(test_trips) - len(matched_test_trips),
        )

    return assigned


def _assign_from_split_assignments(
    trip_map: pd.DataFrame, assignments_csv: Path, identity_columns: list[str]
) -> pd.DataFrame:
    """Join explicit train/test assignments; reject missing/conflicting labels."""
    assignments = _read_identities(assignments_csv, identity_columns)
    _check_identity_columns(assignments, identity_columns, "Split assignments")
    if "split" not in assignments.columns:
        raise ValueError("Split assignments must contain a 'split' column.")

    assignments = assignments[identity_columns + ["split"]].copy()
    assignments["split"] = assignments["split"].str.strip().str.lower()
    if not assignments["split"].isin(["train", "test"]).all():
        raise ValueError("Split assignment values must be 'train' or 'test'.")
    
    assignments = assignments.drop_duplicates()
    if assignments.duplicated(identity_columns).any():
        raise ValueError("Conflicting train/test labels for a trajectory identity.")

    assigned = trip_map.merge(
        assignments, on=identity_columns, how="left", validate="many_to_one",
    )
    
    if assigned["split"].isna().any():
        raise ValueError(
            "Some prepared trajectories are missing from the split assignments. "
            "A complete split is required to prevent data leakage."
        )
    
    return assigned


def _detect_delimiter(path: Path) -> str:
    """Inspect only the FMM CSV header; sequence values contain commas."""
    with path.open("r", encoding="utf-8-sig") as file:
        header = file.readline().rstrip("\r\n")
    for delimiter in (";", "\t", "|"):
        if delimiter in header:
            return delimiter
    return ","


def _split_match_rows(
    matches: pd.DataFrame, trip_map: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, str]:
    """Partition FMM rows while preserving all original columns.

    Returns:
        (train, test, unassigned, raw_ids, id_column). ``raw_ids`` and
        ``id_column`` support diagnostics for missing or zero FMM IDs.
    """
    # FMM normally writes 'id' even when the GPS input column is 'fmm_id'.
    id_column = "id" if "id" in matches else "fmm_id"
    if id_column not in matches:
        raise ValueError("FMM output must contain 'id' or 'fmm_id'.")

    raw_ids = pd.to_numeric(matches[id_column], errors="coerce")
    integer_ids = raw_ids.notna() & raw_ids.ge(1) & raw_ids.eq(raw_ids.round())
    valid_ids = raw_ids.where(integer_ids)

    # Map only known, positive trajectory IDs. ID 0 is an FMM failure artifact.
    split_by_id = trip_map.set_index("fmm_id")["split"]
    labels = valid_ids.map(split_by_id)
    train = matches.loc[labels.eq("train")].copy()
    test = matches.loc[labels.eq("test")].copy()
    unassigned = matches.loc[labels.isna()].copy()
    return train, test, unassigned, raw_ids, id_column


def _count_matched_observations(opath: object) -> int:
    """Count FMM opath road IDs for diagnostics only (never filter here)."""
    if pd.isna(opath) or not str(opath).strip():
        return 0
    return len(re.findall(r"-?\d+", str(opath)))


def _summarize(
    matches: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    unassigned: pd.DataFrame,
    raw_ids: pd.Series,
    assigned_map: pd.DataFrame,
    id_column: str,
) -> tuple[dict, pd.DataFrame]:
    """Summarize FMM output and find input IDs without identifiable results.

    Returns:
        (summary_counts, missing_id_rows).
    """
    observed_ids = set(pd.to_numeric(
        pd.concat([train[id_column], test[id_column]]), errors="coerce"
    ).dropna().astype("int64"))
    missing = assigned_map.loc[~assigned_map["fmm_id"].isin(observed_ids)].copy()

    summary = {
        "input_trajectories": len(assigned_map),
        "input_train_trajectories": int(assigned_map["split"].eq("train").sum()),
        "input_test_trajectories": int(assigned_map["split"].eq("test").sum()),
        "raw_fmm_rows": len(matches),
        "train_fmm_rows": len(train),
        "test_fmm_rows": len(test),
        "unassigned_fmm_rows": len(unassigned),
        "unassigned_id_zero_rows": int(raw_ids.eq(0).sum()),
        "unassigned_other_rows": int(len(unassigned) - raw_ids.eq(0).sum()),
        "input_ids_without_identifiable_fmm_row": len(missing),
    }
    if "opath" in matches:
        for name, frame in (("raw", matches), ("train", train), ("test", test)):
            lengths = frame["opath"].map(_count_matched_observations)
            summary[f"{name}_empty_opath_rows"] = int(lengths.eq(0).sum())
            summary[f"{name}_single_opath_rows"] = int(lengths.eq(1).sum())
            summary[f"{name}_multi_opath_rows"] = int(lengths.ge(2).sum())
    return summary, missing


def split_fmm_matches(
    matches_path: Path,
    trip_map_path: Path,
    output_dir: Path,
    prefix: str,
    identity_columns: list[str],
    test_trips: Path | None = None,
    split_assignments: Path | None = None,
) -> dict:
    """Write train/test FMM CSVs without changing the combined raw results.

    Args:
        identity_columns: Original trajectory keys used to look up split labels.
        test_trips: CSV of held-out identities; other known trips are training.
        split_assignments: CSV with identity columns and a ``split`` label
            for every prepared trajectory. Alternative to ``test_trips``.

    Returns:
        Counts of input, train, test, and unassigned FMM results.
    """
    trip_map = _read_trip_map(trip_map_path, identity_columns)
    if test_trips is not None:
        assigned_map = _assign_from_test_trips(trip_map, test_trips, identity_columns)
    elif split_assignments is not None:
        assigned_map = _assign_from_split_assignments(trip_map, split_assignments, identity_columns)
    else:
        raise ValueError("Supply test_trips or split_assignments.")

    matches = pd.read_csv(matches_path, sep=_detect_delimiter(matches_path))
    train, test, unassigned, raw_ids, id_column = _split_match_rows(matches, assigned_map)
    
    summary, missing = _summarize(
        matches, train, test, unassigned, raw_ids, assigned_map, id_column
    )

    outputs = {
        f"{prefix}_train_fmm_matches.csv": train,
        f"{prefix}_test_fmm_matches.csv": test,
        f"{prefix}_unassigned_fmm_matches.csv": unassigned,
        f"{prefix}_fmm_ids_without_match.csv": missing,
    }
    # No output may point at the raw input: keep the exact original FMM file.
    for name in outputs:
        if (output_dir / name).resolve() == matches_path.resolve():
            raise ValueError("Output would overwrite the raw FMM results.")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False)
        logger.info("Wrote %s (%d rows)", output_dir / name, len(frame))

    summary_path = output_dir / f"{prefix}_fmm_split_summary.json"
    if summary_path.resolve() == matches_path.resolve():
        raise ValueError("Summary path would overwrite raw FMM results.")
    
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    
    logger.info("Wrote %s", summary_path)
    logger.info("FMM split summary: %s", summary)
    
    if summary["unassigned_id_zero_rows"]:
        logger.warning(
            "FMM returned %d rows with ID 0; their original trip IDs cannot "
            "be recovered from the output CSV alone.",
            summary["unassigned_id_zero_rows"],
        )
    return summary


def main() -> None:
    """Run the shared post-FMM split without invoking map matching."""
    args = parse_args()
    setup_logging(args.log_file)
    
    settings = (load_config(args.config).get("trajectories") or {})
    identity_columns = list(settings.get("split_identity_columns", ("uid", "tid")))
    split_fmm_matches(
        matches_path=args.fmm_matches,
        trip_map_path=args.trip_id_map,
        output_dir=args.output_dir,
        prefix=args.prefix,
        identity_columns=identity_columns,
        test_trips=args.test_trips,
        split_assignments=args.split_assignments,
    )


if __name__ == "__main__":
    main()
