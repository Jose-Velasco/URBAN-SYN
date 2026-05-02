import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd
import geopandas as gpd
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from dateutil import tz

from dataclass_models import BuildCsvStats, DuplicateAction, EdgeTimePoint, InterpolationStats

# Logging
class TqdmLoggingHandler(logging.Handler):
    """
    Logging handler that writes through tqdm so log messages do not
    corrupt the active progress bar.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)

def setup_logger(
        name: str,
        log_dir: Path = Path("./logs"),
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        time_zone_info = tz.gettz('America/Los_Angeles')
    ) -> logging.Logger:
    """
    Create a logger that writes to both terminal and a timestamped log file.

    Terminal logs use tqdm.write so they do not break tqdm progress bars.
    File logs include DEBUG messages for detailed troubleshooting.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(time_zone_info).strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{name}_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Prevent duplicate logs if the script is rerun in the same Python session.
    logger.handlers.clear()
    logger.propagate = False

    # # Prevent duplicate handlers if re-run in notebook/dev
    # if logger.hasHandlers():
    #     logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info(f"Logging to {log_file}")

    return logger



def ensure_dir(path: Path) -> None:
    """
    Create directory if it does not exist.

    Parameters
    ----------
    path : Path
        Directory path to create.
    """
    path.mkdir(parents=True, exist_ok=True)


def geometry_to_coordinate_string(geom) -> str:
    """
    Convert a LineString geometry into a JSON string of coordinates.

    Expected format for TS-TrajGen:
        [[lon, lat], [lon, lat], ...]

    Parameters
    ----------
    geom : shapely.geometry.LineString

    Returns
    -------
    str
        JSON string of coordinate pairs.
    """
    coords = [[float(x), float(y)] for x, y in geom.coords]
    return json.dumps(coords, separators=(",", ":"))

# .GEO
def build_geo(network_path: Path):
    """
    Build `.geo` file and mapping from fid -> geo_id.

    The `.geo` file represents road segments with geometry.

    Parameters
    ----------
    network_path : Path
        Path to shapefile / geopackage containing road edges.

    Returns
    -------
    geo_df : pd.DataFrame
        DataFrame ready to save as `.geo`.
    edges_df : pd.DataFrame
        Original edges with added `geo_id`.
    fid_to_geo : dict[int, int]
        Mapping from FMM edge IDs (fid) to geo_id.
    """
    edges = gpd.read_file(network_path).copy()

    required = {"fid", "u", "v", "geometry"}
    missing = required - set(edges.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    edges = edges.sort_values("fid").reset_index(drop=True)
    edges["geo_id"] = range(len(edges))

    # fid_to_geo = dict(zip(edges["fid"], edges["geo_id"]))
    fid_to_geo = dict(zip(edges["fid"].astype(int), edges["geo_id"].astype(int)))

    geo_df = pd.DataFrame({
        "geo_id": edges["geo_id"].astype(int),
        "type": "LineString",
        "coordinates": edges["geometry"].apply(geometry_to_coordinate_string),
    })

    return geo_df, edges, fid_to_geo


# .REL
def build_rel(edges_df: pd.DataFrame):
    """
    Build `.rel` file representing road segment adjacency.

    Note:
        Edges are directed (u -> v). Here we are building EDGE-to-EDGE connections,
        NOT node-to-node.

        An edge A can transition to edge B if the end of A (v) matches the start
        of B (u), i.e., A.v == B.u. This represents a valid movement from one
        road segment to the next in the network.

    A connection exists if:
        edge A -> edge B where A.v == B.u
    
    whats being matched it:
        end(A) -> start(B)
    
    We are looking at:
        end of edge A == start of edge B
    
    not:
        u -> v defines relation

    Parameters
    ----------
    edges_df : pd.DataFrame
        Road network with columns [geo_id, u, v].

    Returns
    -------
    pd.DataFrame
        Relation DataFrame for `.rel`.
    """
    df = edges_df[["geo_id", "u", "v"]].copy()

    # origin edge exits at node = junction
    left = df.rename(columns={"geo_id": "origin_id", "v": "junction"})
    # destination edge enters at node = junction
    right = df.rename(columns={"geo_id": "destination_id", "u": "junction"})

    rel = left.merge(right, on="junction")[["origin_id", "destination_id"]].drop_duplicates()

    rel.insert(0, "rel_id", range(len(rel)))
    rel.insert(1, "type", "geo")

    return rel

# def parse_cpath_like(value: Any) -> list[int]:
    # """
    # Parse a comma-separated or Python-list-like edge sequence.

    # Supported formats
    # -----------------
    # - "1,2,3"
    # - "[1, 2, 3]"
    # - ""
    # - NaN

    # Parameters
    # ----------
    # value : Any
    #     Raw serialized edge sequence.

    # Returns
    # -------
    # list[int]
    #     Parsed edge ID sequence.
    # """
    # if pd.isna(value):
    #     return []

    # s = str(value).strip()
    # if not s:
    #     return []

    # if s.startswith("[") and s.endswith("]"):
    #     try:
    #         return [int(x) for x in ast.literal_eval(s)]
    #     except Exception:
    #         pass

    # return [int(x.strip()) for x in s.split(",") if x.strip()]

def _is_empty_sequence(value: Any) -> bool:
    """
    Return True if the input represents an empty or missing sequence.
    """
    return pd.isna(value) or not str(value).strip()


def _parse_python_list(s: str) -> list[int] | None:
    """
    Attempt to parse a Python list string safely.

    Returns None if parsing fails.
    """
    if not (s.startswith("[") and s.endswith("]")):
        return None

    try:
        parsed = ast.literal_eval(s)

        # Ensure it's actually iterable and cast elements to int
        return [int(x) for x in parsed]

    except Exception:
        # We silently fall back to comma parsing instead of crashing
        return None


def _parse_comma_separated(s: str) -> list[int]:
    """
    Parse comma-separated string into integers.

    Ignores empty tokens caused by malformed input like "1,,2".
    """
    return [
        int(token.strip())
        for token in s.split(",")
        if token.strip()
    ]


def parse_cpath_like(value: Any) -> list[int]:
    """
    Parse a serialized edge sequence into a list of integers.

    Supported formats
    -----------------
    - "1,2,3"
    - "[1, 2, 3]"
    - ""
    - NaN

    Parameters
    ----------
    value : Any
        Raw serialized edge sequence.

    Returns
    -------
    list[int]
        Parsed edge ID sequence.

    Notes
    -----
    - Attempts Python list parsing first (safer for structured inputs).
    - Falls back to comma-separated parsing if needed.
    - Invalid tokens are ignored rather than raising errors.
    """

    if _is_empty_sequence(value):
        return []

    s = str(value).strip()

    # Try structured parsing first (more reliable if valid)
    parsed_list = _parse_python_list(s)
    if parsed_list is not None:
        return parsed_list

    # Fallback: simple comma-separated parsing
    return _parse_comma_separated(s)


# def parse_tpath(value: Any) -> list[list[int]]:
#     """
#     Parse FMM `tpath` into per-GPS-interval edge segments.

#     Example
#     -------
#     "2|2,5,13|13,14|14,23"
#     ->
#     [[2], [2, 5, 13], [13, 14], [14, 23]]

#     Parameters
#     ----------
#     value : Any
#         Raw tpath string from FMM output.

#     Returns
#     -------
#     list[list[int]]
#         One edge list per consecutive GPS-point interval.
#     """
#     if pd.isna(value):
#         return []

#     s = str(value).strip()
#     if not s:
#         return []

#     segments: list[list[int]] = []
#     for chunk in s.split("|"):
#         chunk = chunk.strip()
#         if not chunk:
#             segments.append([])
#             continue
#         segments.append(parse_cpath_like(chunk))

#     return segments


def _is_empty_tpath(value: Any) -> bool:
    """
    Check whether a tpath value is empty or invalid.
    """
    return pd.isna(value) or not str(value).strip()

def _split_tpath_chunks(value: Any) -> list[str]:
    """
    Split raw tpath string into raw segment chunks.

    Example
    -------
    "2|2,5,13|13,14"
    -> ["2", "2,5,13", "13,14"]
    """
    return str(value).strip().split("|")

def _parse_tpath_chunk(chunk: str) -> list[int]:
    """
    Parse a single tpath chunk into a list of integers.

    Empty chunks are valid and represent missing segments.
    """
    chunk = chunk.strip()

    if not chunk:
        return []

    # Delegates actual parsing logic to existing function
    # so we keep parsing rules consistent across pipeline
    return parse_cpath_like(chunk)

def parse_tpath(value: Any) -> list[list[int]]:
    """
    Parse FMM `tpath` into per-GPS-interval edge segments.

    Each segment corresponds to one GPS interval:
    GPS[i] → GPS[i+1]

    Example
    -------
    "2|2,5,13|13,14|14,23"
    ->
    [
        [2],
        [2, 5, 13],
        [13, 14],
        [14, 23],
    ]

    Parameters
    ----------
    value : Any
        Raw `tpath` value from FMM output.

    Returns
    -------
    list[list[int]]
        List of edge sequences per GPS interval.

    Notes
    -----
    - Empty segments are preserved to maintain alignment with GPS timestamps.
    - Uses `parse_cpath_like` to ensure consistent edge parsing across pipeline.
    """

    if _is_empty_tpath(value):
        return []

    chunks = _split_tpath_chunks(value)

    # Preserve empty segments to maintain GPS interval alignment
    return [_parse_tpath_chunk(chunk) for chunk in chunks]


# Edge length and interpolation helpers
def _safe_timestamp(value: Any) -> pd.Timestamp:
    """Convert a value to pandas Timestamp."""
    return pd.to_datetime(value)

def _get_edge_lengths(
    edge_ids: list[int],
    geo_to_length: dict[int, float],
    stats: InterpolationStats | None = None,
    ) -> list[float]:
    """
    Return non-negative edge lengths with fallback defaults.

    Missing lengths default to 1.0 so interpolation can still run.
    """
    lengths = [
        max(float(geo_to_length.get(edge_id, 1.0)), 0.0)
        for edge_id in edge_ids
    ]

    if sum(lengths) <= 0:
        if stats is not None:
            stats.invalid_length_segments += 1

        # Equal weights avoid divide-by-zero while preserving ordering.
        return [1.0] * len(edge_ids)

    return lengths


def _minimum_required_seconds(
    edge_count: int,
    min_delta_seconds: float,
    ) -> float:
    """
    Return minimum time needed to create strictly increasing timestamps.

    N edge timestamps require N-1 gaps.
    """
    return max(edge_count - 1, 0) * min_delta_seconds

def _use_fallback_interval(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    edge_count: int,
    min_delta_seconds: float,
    stats: InterpolationStats | None,
    logger: logging.Logger | None,
) -> tuple[pd.Timestamp, float]:
    """
    Validate the GPS interval and return the original end_time and duration.

    We do not extend timestamps beyond the observed GPS interval. If the
    interval is too short, we only log/count it and still interpolate within
    the real bounds.
    """
    total_seconds = (end_time - start_time).total_seconds()
    min_required_seconds = _minimum_required_seconds(edge_count, min_delta_seconds)

    if total_seconds <= 0:
        if stats is not None:
            stats.non_positive_time_segments += 1
            stats.fallback_segments += 1

        if logger is not None:
            logger.debug(
                "Non-positive GPS interval; preserving original timestamps. "
                f"edges={edge_count}, total_seconds={total_seconds:.3f}, "
                f"start={start_time}, end={end_time}"
            )

        return end_time, total_seconds


    if total_seconds < min_required_seconds:
        if stats is not None:
            stats.short_time_segments += 1
            stats.fallback_segments += 1

        if logger is not None:
            logger.debug(
                "Short GPS interval; interpolating within original bounds. "
                f"edges={edge_count}, total_seconds={total_seconds:.3f}, "
                f"min_required_seconds={min_required_seconds:.3f}, "
                f"start={start_time}, end={end_time}"
            )

    return end_time, total_seconds


# def _use_fallback_interval(
#     start_time: pd.Timestamp,
#     end_time: pd.Timestamp,
#     edge_count: int,
#     min_delta_seconds: float,
#     stats: InterpolationStats | None,
#     logger: logging.Logger | None,
#     ) -> tuple[pd.Timestamp, float]:
#     """
#     Validate the GPS interval and return a usable end_time and total_seconds.

#     If GPS timing is invalid or too short for the number of edges, this creates
#     a synthetic minimum interval while tracking why fallback was needed.
#     """
#     total_seconds = (end_time - start_time).total_seconds()
#     min_required_seconds = _minimum_required_seconds(edge_count, min_delta_seconds)

#     # GPS timestamps are invalid (same or reversed order), so we cannot
#     # derive a meaningful duration; fallback to synthetic minimum interval.
#     if total_seconds <= 0:
#         if stats is not None:
#             stats.non_positive_time_segments += 1
#             stats.fallback_segments += 1

#         if logger is not None:
#             logger.debug(
#                 "Fallback: non-positive time segment | "
#                 f"edges={edge_count}, total_seconds={total_seconds:.3f}, "
#                 f"start={start_time}, end={end_time}"
#             )

#         fallback_end_time = start_time + pd.to_timedelta(
#             min_required_seconds,
#             unit="s",
#         )
#         return fallback_end_time, min_required_seconds

#     # The GPS interval is too short to assign strictly increasing timestamps
#     # across all edges; fallback ensures each edge gets at least min_delta spacing.
#     if total_seconds < min_required_seconds:
#         if stats is not None:
#             stats.short_time_segments += 1
#             stats.fallback_segments += 1

#         if logger is not None:
#             logger.debug(
#                 "Fallback: short-time segment | "
#                 f"edges={edge_count}, total_seconds={total_seconds:.3f}, "
#                 f"min_required_seconds={min_required_seconds:.3f}, "
#                 f"start={start_time}, end={end_time}"
#             )

#         fallback_end_time = start_time + pd.to_timedelta(
#             min_required_seconds,
#             unit="s",
#         )
#         return fallback_end_time, min_required_seconds

#     return end_time, total_seconds

def _interpolate_middle_edge_points(
    start_time: pd.Timestamp,
    total_seconds: float,
    edge_ids: list[int],
    lengths: list[float],
    ) -> list[EdgeTimePoint]:
    """
    Interpolate timestamps for middle edges only.

    First and last edges are handled separately because they preserve GPS
    anchor timestamps.
    """
    if len(edge_ids) <= 2:
        return []

    total_length = sum(lengths)
    
    # Start cumulative distance at the first edge so interpolation for middle
    # edges reflects distance traveled *before entering* each edge.
    cumulative_length = lengths[0]
    middle_points: list[EdgeTimePoint] = []

    for edge_id, length in zip(edge_ids[1:-1], lengths[1:-1]):
        # Middle edge time is based on distance traveled before entering it.
        fraction = cumulative_length / total_length

        # Compute edge-entry time by mapping cumulative distance fraction to time,
        # assuming traversal time is proportional to edge length.
        # Convert fractional seconds to timedelta.
        # Even though unit="s", pandas preserves sub-second precision (ns-level),
        # so interpolated timestamps retain millisecond resolution.
        timestamp = start_time + pd.to_timedelta(total_seconds * fraction, unit="s")

        middle_points.append(
            EdgeTimePoint(
                edge_id=edge_id,
                timestamp=timestamp,
                is_anchor=False,
            )
        )
        cumulative_length += length

    return middle_points

def interpolate_edge_time_points_by_length(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    edge_ids: list[int],
    geo_to_length: dict[int, float],
    min_delta_seconds: float = 1.0,
    stats: InterpolationStats | None = None,
    logger: logging.Logger | None = None,
    ) -> list[EdgeTimePoint]:
    """
    Assign timestamps to road edges while preserving GPS timestamps as anchors.

    The first edge receives `start_time` as a GPS anchor. The last edge receives
    `end_time` as a GPS anchor when the segment has two or more edges. Middle
    edges receive length-weighted interpolated timestamps.

    If the GPS interval is invalid or too short to create strictly increasing
    timestamps, the function creates a minimum-length synthetic interval and
    records the fallback reason in `stats`.
    """
    if not edge_ids:
        return []

    if stats is not None:
        stats.total_segments += 1

    start_time = _safe_timestamp(start_time)
    end_time = _safe_timestamp(end_time)

    if len(edge_ids) == 1:
        return [EdgeTimePoint(edge_ids[0], start_time, is_anchor=True)]

    end_time, total_seconds = _use_fallback_interval(
        start_time=start_time,
        end_time=end_time,
        edge_count=len(edge_ids),
        min_delta_seconds=min_delta_seconds,
        stats=stats,
        logger=logger,
    )

    lengths = _get_edge_lengths(edge_ids, geo_to_length, stats=stats)

    return [
        EdgeTimePoint(edge_ids[0], start_time, is_anchor=True),
        *_interpolate_middle_edge_points(
            start_time=start_time,
            total_seconds=total_seconds,
            edge_ids=edge_ids,
            lengths=lengths,
        ),
        EdgeTimePoint(edge_ids[-1], end_time, is_anchor=True),
    ]

# Duplicate policy helpers

def _resolve_consecutive_duplicate(
    prev: EdgeTimePoint,
    curr: EdgeTimePoint,
    ) -> DuplicateAction:
    """
    Decide how to handle consecutive duplicate edge IDs.

    Rules:
    - Different edge IDs are appended normally.
    - GPS anchors beat interpolated points.
    - Interpolated duplicates are dropped.
    - Anchor-anchor duplicates are kept only when timestamps differ.
    """
    # If edges are different, no duplication: keep normally
    if prev.edge_id != curr.edge_id:
        return DuplicateAction.APPEND

    # Prefer real GPS timestamp over inferred timestamp
    if prev.is_anchor and not curr.is_anchor:
        return DuplicateAction.DROP_CURRENT

    if not prev.is_anchor and curr.is_anchor:
        return DuplicateAction.REPLACE_PREVIOUS

    # Both are synthetic, redundant: keep only one
    if not prev.is_anchor and not curr.is_anchor:
        return DuplicateAction.DROP_CURRENT

    # anchor vs anchor
    # Same edge and same anchor timestamp is redundant boundary_noise/observation.
    # Same timestamp: duplicate artifact (no new information)
    if prev.timestamp == curr.timestamp:
        return DuplicateAction.DROP_CURRENT

    # Different timestamps: real repeated observation on same road
    # same edge and different anchor timestamp = real repeated/stationary observation
    return DuplicateAction.APPEND

def _record_duplicate_action(
    action: DuplicateAction,
    prev: EdgeTimePoint,
    curr: EdgeTimePoint,
    stats: InterpolationStats | None,
    ) -> None:
    """
    Update duplicate-resolution counters.
    """
    if stats is None or action == DuplicateAction.APPEND:
        return

    if action == DuplicateAction.REPLACE_PREVIOUS:
        stats.duplicate_previous_replaced += 1
        return

    stats.duplicate_current_dropped += 1

    if prev.is_anchor and curr.is_anchor and prev.timestamp == curr.timestamp:
        stats.duplicate_anchor_same_time_dropped += 1

def _append_with_duplicate_policy(
    stitched_points: list[EdgeTimePoint],
    point: EdgeTimePoint,
    stats: InterpolationStats | None = None,
    ) -> None:
    """
    Append a point while resolving consecutive duplicate edge IDs.

    The policy preserves GPS anchors over interpolated values and removes
    redundant consecutive duplicate edges.
    """
    if not stitched_points:
        stitched_points.append(point)
        return

    prev = stitched_points[-1]
    action = _resolve_consecutive_duplicate(prev, point)

    _record_duplicate_action(action, prev, point, stats)

    if action == DuplicateAction.APPEND:
        stitched_points.append(point)
        return

    if action == DuplicateAction.REPLACE_PREVIOUS:
        # Replace synthetic timing with the real GPS anchor for the same edge.
        stitched_points[-1] = point

# tpath helpers

def _map_fids_to_geo_ids(
    fids: list[int],
    fid_to_geo: dict[int, int],
    ) -> list[int]:
    """
    Convert FMM edge IDs to geo_id values.

    Unknown FMM IDs are skipped so one missing edge does not discard the entire
    trajectory.
    """
    return [fid_to_geo[fid] for fid in fids if fid in fid_to_geo]

def _format_timestamp(ts: pd.Timestamp) -> str:
    """
    Format timestamp with millisecond precision.

    This prevents multiple interpolated timestamps within the same second
    from collapsing into identical values when serialized.
    """
    return ts.isoformat(timespec="milliseconds").replace("+00:00", "Z")

def _format_time_points(
    points: list[EdgeTimePoint],
    ) -> tuple[list[int], list[str]]:
    """
    Convert stitched EdgeTimePoint objects into rid_list and time_list.
    """
    rid_list = [point.edge_id for point in points]
    time_list = [
        _format_timestamp(point.timestamp)
        for point in points
    ]
    # time_list = [
    #     point.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    #     for point in points
    # ]

    return rid_list, time_list

def _iter_usable_tpath_intervals(
    tpath_segments_fid: list[list[int]],
    point_times: list[pd.Timestamp],
    ):
    """
    Yield tpath segments aligned to GPS timestamp intervals.

    FMM and raw GPS counts can disagree, so this safely truncates to the
    smallest valid interval count.
    """
    usable_intervals = min(len(tpath_segments_fid), len(point_times) - 1)

    for i in range(usable_intervals):
        yield i, tpath_segments_fid[i], point_times[i], point_times[i + 1]

def build_rid_and_time_lists_from_tpath(
    traj_id: str,
    tpath_value: Any,
    trip_time_lookup: dict[str, list[pd.Timestamp]],
    fid_to_geo: dict[int, int],
    geo_to_length: dict[int, float],
    min_delta_seconds: float = 1.0,
    stats: InterpolationStats | None = None,
    logger: logging.Logger | None = None,
    ) -> tuple[list[int], list[str]]:
    """
    Build aligned `rid_list` and `time_list` for one FMM trajectory.

    This function parses FMM `tpath`, aligns each tpath segment to a GPS
    timestamp interval, maps FMM IDs to geo IDs, preserves GPS timestamps as
    anchors, interpolates timestamps for intermediate map-matched edges, and
    resolves duplicate consecutive edge IDs.

    Duplicate policy:
    - GPS anchors beat interpolated timestamps.
    - Interpolated duplicate edges are dropped.
    - Anchor-anchor duplicates are kept only when timestamps differ.
    """
    tpath_segments_fid = parse_tpath(tpath_value)
    point_times = trip_time_lookup.get(str(traj_id), [])

    if len(point_times) < 2 or not tpath_segments_fid:
        return [], []

    stitched_points: list[EdgeTimePoint] = []

    for _, seg_fids, start_time, end_time in _iter_usable_tpath_intervals(
        tpath_segments_fid,
        point_times,
    ):
        if not seg_fids:
            continue

        seg_geo = _map_fids_to_geo_ids(seg_fids, fid_to_geo)

        if not seg_geo:
            continue

        segment_points = interpolate_edge_time_points_by_length(
            start_time=start_time,
            end_time=end_time,
            edge_ids=seg_geo,
            geo_to_length=geo_to_length,
            min_delta_seconds=min_delta_seconds,
            stats=stats,
            logger=logger,
        )

        for point in segment_points:
            _append_with_duplicate_policy(stitched_points, point, stats=stats)

    return _format_time_points(stitched_points)

# build_mm_csvs helpers

def _has_valid_tpath(row: Any) -> bool:
    """
    Return True if an FMM row has a non-empty tpath field.
    """
    tpath_value = getattr(row, "tpath", "")
    return not (pd.isna(tpath_value) or str(tpath_value).strip() == "")

def _has_valid_times(
    traj_id: str,
    trip_time_lookup: dict[str, list[pd.Timestamp]],
    ) -> bool:
    """
    Return True if a trajectory has at least two GPS timestamps.
    """
    return traj_id in trip_time_lookup and len(trip_time_lookup[traj_id]) >= 2

def _make_mm_row(
    traj_id: str,
    rid_list: list[int],
    time_list: list[str],
    ) -> dict[str, str]:
    """
    Create one TS-TrajGen-compatible CSV row.
    """
    return {
        "traj_id": traj_id,
        "rid_list": ",".join(str(edge_id) for edge_id in rid_list),
        "time_list": ",".join(time_list),
    }

def _update_progress_bar(
    progress: tqdm,
    build_stats: BuildCsvStats,
    interp_stats: InterpolationStats,
    ) -> None:
    """
    Update tqdm postfix with live data-quality metrics.
    """
    progress.set_postfix(
        kept=build_stats.kept_rows,
        skipped=build_stats.skipped_total,
        fallback=interp_stats.fallback_segments,
        fb_rate=f"{interp_stats.fallback_rate():.2%}",
        repl=interp_stats.duplicate_previous_replaced,
        drop=interp_stats.duplicate_current_dropped,
    )

def _split_train_test(
    df: pd.DataFrame,
    train_ratio: float,
    random_state: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split trajectory dataframe into train and test sets.
    """
    train = df.sample(frac=train_ratio, random_state=random_state)
    test = df.drop(train.index)

    return train.reset_index(drop=True), test.reset_index(drop=True)


def _print_build_summary(
    build_stats: BuildCsvStats,
    interp_stats: InterpolationStats,
    train_size: int,
    test_size: int,
) -> None:
    """
    Print summary stats for interactive runs.
    """
    print("\nMM CSV BUILD STATS")
    print(f"Total input trajectories: {build_stats.total_rows:,}")
    print(f"Kept trajectories:        {build_stats.kept_rows:,}")
    print(f"Kept ratio:               {build_stats.kept_ratio():.3f}")
    print()
    print("Skipped breakdown:")
    print(f"  Empty tpath:            {build_stats.skipped_empty_tpath:,}")
    print(f"  Missing GPS times:      {build_stats.skipped_missing_times:,}")
    print(f"  Too short final path:   {build_stats.skipped_short_path:,}")
    print()
    print("INTERPOLATION SUMMARY")
    print(f"  Total segments:         {interp_stats.total_segments:,}")
    print(f"  Fallback segments:      {interp_stats.fallback_segments:,}")
    print(f"  Fallback rate:          {interp_stats.fallback_rate():.3%}")
    print(f"  Zero/reversed-time:     {interp_stats.non_positive_time_segments:,}")
    print(f"  Short-time segments:    {interp_stats.short_time_segments:,}")
    print(f"  Invalid-length:         {interp_stats.invalid_length_segments:,}")
    print(f"  Duplicate replaced:     {interp_stats.duplicate_previous_replaced:,}")
    print(f"  Duplicate dropped:      {interp_stats.duplicate_current_dropped:,}")
    print(f"  Train size:             {train_size:,}")
    print(f"  Test size:              {test_size:,}")

def build_mm_csvs(
    fmm_path: Path | str,
    fid_to_geo: dict[int, int],
    geo_to_length: dict[int, float],
    trip_time_lookup: dict[str, list[pd.Timestamp]],
    train_ratio: float,
    random_state: int,
    min_len: int = 2,
    verbose: bool = False,
    fmm_sep: str = ";",
    min_delta_seconds: float = 1.0,
    logger: logging.Logger | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build TS-TrajGen train/test CSVs from FMM map-matching output.

    This function reads FMM output, converts each `tpath` into aligned
    `rid_list` and `time_list` sequences, preserves original GPS timestamps as
    anchors, interpolates timestamps for intermediate road segments, filters
    invalid/short trajectories, and performs a reproducible train/test split.

    Parameters
    ----------
    fmm_path : Path | str
        Path to the FMM output CSV.
    fid_to_geo : dict[int, int]
        Mapping from FMM edge ID (`fid`) to dataset road ID (`geo_id`).
    geo_to_length : dict[int, float]
        Mapping from `geo_id` to road segment length.
    trip_time_lookup : dict[str, list[pd.Timestamp]]
        Mapping from trajectory ID to ordered raw GPS timestamps.
    train_ratio : float
        Fraction of valid trajectories assigned to the training set.
    random_state : int
        Seed used for reproducible train/test splitting.
    min_len : int, optional
        Minimum number of road segments required to keep a trajectory.
    verbose : bool, optional
        If True, prints a summary in addition to logging it.
    fmm_sep : str, optional
        Delimiter used by the FMM output file.
    min_delta_seconds : float, optional
        Minimum spacing used when fallback interpolation is required.
    logger : logging.Logger, optional
        Logger used for progress, debug messages, and summary output.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Train and test DataFrames with columns `traj_id`, `rid_list`, and
        `time_list`.
    """
    logger = logger or logging.getLogger(__name__)

    logger.info(f"Loading FMM file: {fmm_path}")
    fmm = pd.read_csv(fmm_path, sep=fmm_sep, engine="python")
    logger.info(f"Loaded {len(fmm):,} FMM rows")

    build_stats = BuildCsvStats()
    interp_stats = InterpolationStats()
    rows: list[dict[str, str]] = []

    progress = tqdm(
        fmm.itertuples(index=False),
        total=len(fmm),
        desc="Building MM CSVs",
        unit="traj",
    )

    for row_idx, row in enumerate(progress):
        build_stats.total_rows += 1
        traj_id = str(row.id)

        if not _has_valid_tpath(row):
            build_stats.skipped_empty_tpath += 1
            continue

        if not _has_valid_times(traj_id, trip_time_lookup):
            build_stats.skipped_missing_times += 1
            continue

        rid_list, time_list = build_rid_and_time_lists_from_tpath(
            traj_id=traj_id,
            tpath_value=getattr(row, "tpath"),
            trip_time_lookup=trip_time_lookup,
            fid_to_geo=fid_to_geo,
            geo_to_length=geo_to_length,
            min_delta_seconds=min_delta_seconds,
            stats=interp_stats,
            logger=logger,
        )

        if len(rid_list) < min_len:
            build_stats.skipped_short_path += 1
            continue

        rows.append(_make_mm_row(traj_id, rid_list, time_list))
        build_stats.kept_rows += 1

        if row_idx % 100 == 0:
            _update_progress_bar(progress, build_stats, interp_stats)

    progress.close()

    df = pd.DataFrame(rows)

    if df.empty:
        logger.warning("No valid trajectories were kept. Returning empty train/test dataframes.")
        build_stats.log_summary(logger, train_size=0, test_size=0)
        interp_stats.log_summary(logger)
        return df, df.copy()

    train, test = _split_train_test(df, train_ratio, random_state)

    build_stats.log_summary(logger, train_size=len(train), test_size=len(test))
    interp_stats.log_summary(logger)

    if verbose:
        _print_build_summary(
            build_stats=build_stats,
            interp_stats=interp_stats,
            train_size=len(train),
            test_size=len(test),
        )

    return train, test



# Time interpolation helpers
# def _get_edge_lengths(
#     edge_ids: list[int],
#     geo_to_length: dict[int, float],
#     stats: InterpolationStats | None = None,
# ) -> list[float]:
#     """
#     Return non-negative edge lengths.

#     Missing edges default to 1.0 so interpolation can still proceed.
#     If all lengths are invalid, equal weights are used.
#     """
#     lengths = [
#         max(float(geo_to_length.get(edge_id, 1.0)), 0.0)
#         for edge_id in edge_ids
#     ]

#     if sum(lengths) <= 0:
#         if stats is not None:
#             stats.invalid_length_segments += 1

#         # Equal weights are safer than returning zeros because weights need a
#         # positive denominator.
#         return [1.0] * len(edge_ids)

#     return lengths


# def _compute_gap_weights(lengths: list[float]) -> list[float]:
#     """
#     Compute weights for gaps between edge-entry timestamps.

#     For N edge-entry timestamps, there are N-1 gaps.
#     """
#     if len(lengths) <= 1:
#         return []

#     # We return edge-entry timestamps, so the final edge does not need a
#     # following gap.
#     gap_lengths = lengths[:-1]
#     total_gap_length = sum(gap_lengths)

#     if total_gap_length <= 0:
#         return [1.0 / (len(lengths) - 1)] * (len(lengths) - 1)

#     return [length / total_gap_length for length in gap_lengths]

# def _build_weighted_timeline(
#     start_time: pd.Timestamp,
#     gap_weights: list[float],
#     total_seconds: float,
#     min_delta_seconds: float,
# ) -> list[pd.Timestamp]:
#     """
#     Build timestamps from gap weights while enforcing minimum spacing.
#     """
#     times = [start_time]
#     elapsed_seconds = 0.0

#     for weight in gap_weights:
#         # Enforces strictly increasing timestamps even when weighted time is
#         # smaller than timestamp resolution.
#         gap_seconds = max(min_delta_seconds, total_seconds * weight)

#         elapsed_seconds += gap_seconds
#         times.append(start_time + pd.to_timedelta(elapsed_seconds, unit="s"))

#     return times

# def length_weighted_min_step_times(
#     start_time: pd.Timestamp,
#     edge_ids: list[int],
#     geo_to_length: dict[int, float],
#     min_delta_seconds: float = 1.0,
#     stats: InterpolationStats | None = None,
# ) -> list[pd.Timestamp]:
#     """
#     Generate a synthetic strictly increasing timeline using edge-length weights.

#     This is used when the real GPS interval cannot support normal interpolation,
#     such as equal/reversed timestamps or too many inferred edges for a short
#     time interval.
#     """
#     if not edge_ids:
#         return []

#     if len(edge_ids) == 1:
#         return [start_time]

#     # Need N-1 gaps between N edge-entry timestamps.
#     min_total_seconds = (len(edge_ids) - 1) * min_delta_seconds

#     lengths = _get_edge_lengths(edge_ids, geo_to_length, stats=stats)
#     gap_weights = _compute_gap_weights(lengths)

#     return _build_weighted_timeline(
#         start_time=start_time,
#         gap_weights=gap_weights,
#         total_seconds=min_total_seconds,
#         min_delta_seconds=min_delta_seconds,
#     )

# def interpolate_times_by_length(
#     start_time: pd.Timestamp,
#     end_time: pd.Timestamp,
#     edge_ids: list[int],
#     geo_to_length: dict[int, float],
#     min_delta_seconds: float = 1.0,
#     stats: InterpolationStats | None = None,
#     logger: logging.Logger | None = None,
# ) -> list[pd.Timestamp]:
#     """
#     Interpolate timestamps across a sequence of road edges using edge length as weight.

#     For a given GPS interval (start_time -> end_time) and a sequence of edges,
#     this function assigns one timestamp per edge (edge-entry time).

#     Interpolation behavior:
#     - Standard case:
#         Distributes time proportionally based on edge lengths.
#     - Fallback cases:
#         - If timestamps are equal or reversed -> synthetic timeline.
#         - If interval is too short for number of edges -> minimum-step timeline.

#     Parameters
#     ----------
#     start_time : pd.Timestamp
#         Timestamp of the starting GPS point.
#     end_time : pd.Timestamp
#         Timestamp of the ending GPS point.
#     edge_ids : list[int]
#         Sequence of road segment IDs (geo_id).
#     geo_to_length : dict[int, float]
#         Mapping from geo_id to edge length.
#     min_delta_seconds : float, optional
#         Minimum enforced time difference between consecutive edges.
#     stats : InterpolationStats, optional
#         Tracks fallback usage and problematic segments.
#     logger : logging.Logger, optional
#         Logs debug information for fallback conditions.

#     Returns
#     -------
#     list[pd.Timestamp]
#         List of timestamps aligned to each edge (edge-entry times).

#     Notes
#     -----
#     - Edge-entry timestamps represent the time at which traversal of each
#     edge begins.
#     - Fallback logic ensures strictly increasing timestamps and prevents
#     duplicates when GPS resolution is insufficient.
#     - This function is critical for maintaining temporal consistency in
#     downstream trajectory generation models.
#     """
#     if not edge_ids:
#         return []

#     if stats is not None:
#         stats.total_segments += 1

#     start_time = pd.to_datetime(start_time)
#     end_time = pd.to_datetime(end_time)

#     if len(edge_ids) == 1:
#         return [start_time]

#     total_seconds = (end_time - start_time).total_seconds()
#     min_required_seconds = (len(edge_ids) - 1) * min_delta_seconds

#     if total_seconds <= 0:
#         if stats is not None:
#             stats.non_positive_time_segments += 1
#             stats.fallback_segments += 1

#         if logger is not None:
#             logger.debug(
#                 "Fallback: non-positive time segment | "
#                 f"edges={len(edge_ids)}, total_seconds={total_seconds:.3f}, "
#                 f"start={start_time}, end={end_time}"
#             )

#         return length_weighted_min_step_times(
#             start_time=start_time,
#             edge_ids=edge_ids,
#             geo_to_length=geo_to_length,
#             min_delta_seconds=min_delta_seconds,
#             stats=stats,
#         )

#     if total_seconds < min_required_seconds:
#         if stats is not None:
#             stats.short_time_segments += 1
#             stats.fallback_segments += 1

#         if logger is not None:
#             logger.debug(
#                 "Fallback: short-time segment | "
#                 f"edges={len(edge_ids)}, total_seconds={total_seconds:.3f}, "
#                 f"min_required_seconds={min_required_seconds:.3f}, "
#                 f"start={start_time}, end={end_time}"
#             )

#         return length_weighted_min_step_times(
#             start_time=start_time,
#             edge_ids=edge_ids,
#             geo_to_length=geo_to_length,
#             min_delta_seconds=min_delta_seconds,
#             stats=stats,
#         )

#     lengths = _get_edge_lengths(edge_ids, geo_to_length, stats=stats)
#     total_length = sum(lengths)

#     edge_times: list[pd.Timestamp] = []
#     cumulative_length = 0.0

#     for length in lengths:
#         # Edge-entry time depends on distance already traveled before entering
#         # the current edge.
#         fraction = cumulative_length / total_length
#         edge_time = start_time + pd.to_timedelta(total_seconds * fraction, unit="s")

#         edge_times.append(edge_time)
#         cumulative_length += length

#     return edge_times

# # tpath -> rid_list/time_list builder
# def build_rid_and_time_lists_from_tpath(
#     traj_id: str,
#     tpath_value: Any,
#     trip_time_lookup: dict[str, list[pd.Timestamp]],
#     fid_to_geo: dict[int, int],
#     geo_to_length: dict[int, float],
#     min_delta_seconds: float = 1.0,
#     stats: InterpolationStats | None = None,
#     logger: logging.Logger | None = None,
#     ) -> tuple[list[int], list[str]]:
#     """
#     Build aligned rid_list/time_list while preserving GPS anchor timestamps.

#     Duplicate road IDs are resolved by timestamp fidelity:
#     - anchor beats interpolated
#     - interpolated duplicates are dropped
#     - anchor-anchor duplicates are preserved
#     """
#     tpath_segments_fid = parse_tpath(tpath_value)
#     point_times = trip_time_lookup.get(str(traj_id), [])

#     if len(point_times) < 2 or not tpath_segments_fid:
#         return [], []

#     usable_intervals = min(len(tpath_segments_fid), len(point_times) - 1)

#     stitched_points: list[EdgeTimePoint] = []

#     for i in range(usable_intervals):
#         seg_fids = tpath_segments_fid[i]

#         if not seg_fids:
#             continue

#         # Drop unknown FMM edge IDs instead of failing the whole trajectory.
#         seg_geo = [fid_to_geo[fid] for fid in seg_fids if fid in fid_to_geo]

#         if not seg_geo:
#             continue

#         segment_points = interpolate_edge_time_points_by_length(
#             start_time=point_times[i],
#             end_time=point_times[i + 1],
#             edge_ids=seg_geo,
#             geo_to_length=geo_to_length,
#             min_delta_seconds=min_delta_seconds,
#             stats=stats,
#             logger=logger,
#         )

#         for point in segment_points:
#             _append_with_duplicate_policy(stitched_points, point)

#     rid_list = [point.edge_id for point in stitched_points]
#     time_list = [
#         point.timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
#         for point in stitched_points
#     ]

#     return rid_list, time_list


# tpath -> rid_list/time_list builder
# def build_rid_and_time_lists_from_tpath(
#     traj_id: str,
#     tpath_value: Any,
#     trip_time_lookup: dict[str, list[pd.Timestamp]],
#     fid_to_geo: dict[int, int],
#     geo_to_length: dict[int, float],
#     min_delta_seconds: float = 1.0,
#     stats: InterpolationStats | None = None,
#     logger: logging.Logger | None = None,
# ) -> tuple[list[int], list[str]]:
#     """
#     Construct aligned `rid_list` and `time_list` from a single FMM trajectory.

#     This function converts FMM `tpath` output into:
#     - `rid_list`: sequence of geo_ids (road segments)
#     - `time_list`: timestamps aligned to each road segment

#     Processing steps:
#     1. Parse `tpath` into per-interval edge segments.
#     2. Align segments with original GPS timestamps.
#     3. Map FMM edge IDs (fid) to dataset road IDs (geo_id).
#     4. Interpolate timestamps per edge using length-weighted timing.
#     5. Stitch segments into a single trajectory.
#     6. Remove duplicate boundary edges between segments.

#     Parameters
#     ----------
#     traj_id : str
#         Unique trajectory identifier.
#     tpath_value : Any
#         Raw `tpath` field from FMM output.
#     trip_time_lookup : dict[str, list[pd.Timestamp]]
#         Mapping from trajectory ID to ordered GPS timestamps.
#     fid_to_geo : dict[int, int]
#         Mapping from FMM fid to geo_id.
#     geo_to_length : dict[int, float]
#         Mapping from geo_id to edge length.
#     min_delta_seconds : float, optional
#         Minimum enforced time difference between consecutive edges.
#     stats : InterpolationStats, optional
#         Object tracking interpolation quality metrics.
#     logger : logging.Logger, optional
#         Logger for debug output (e.g., fallback segments).

#     Returns
#     -------
#     tuple[list[int], list[str]]
#         - rid_list: list of geo_ids
#         - time_list: list of ISO-formatted timestamps

#     Notes
#     -----
#     - Each `tpath` segment corresponds to a GPS interval:
#     GPS[i] → GPS[i+1].
#     - If segment counts and timestamp counts do not match, the function
#     truncates safely to the minimum valid length.
#     - Duplicate edges at segment boundaries are removed to maintain
#     alignment between `rid_list` and `time_list`.
#     """
#     tpath_segments_fid = parse_tpath(tpath_value)
#     point_times = trip_time_lookup.get(str(traj_id), [])

#     if len(point_times) < 2 or not tpath_segments_fid:
#         return [], []

#     # Truncate safely because FMM tpath segment counts and GPS timestamp counts
#     # can occasionally disagree.
#     usable_intervals = min(len(tpath_segments_fid), len(point_times) - 1)

#     stitched_rids: list[int] = []
#     stitched_times: list[str] = []

#     for i in range(usable_intervals):
#         seg_fids = tpath_segments_fid[i]

#         if not seg_fids:
#             continue

#         # Drop unknown FMM ids instead of failing the whole trajectory.
#         seg_geo = [fid_to_geo[fid] for fid in seg_fids if fid in fid_to_geo]

#         if not seg_geo:
#             continue

#         seg_times = interpolate_times_by_length(
#             start_time=point_times[i],
#             end_time=point_times[i + 1],
#             edge_ids=seg_geo,
#             geo_to_length=geo_to_length,
#             min_delta_seconds=min_delta_seconds,
#             stats=stats,
#             logger=logger,
#         )

#         # Adjacent FMM segments can repeat the boundary edge, so remove the
#         # duplicate to keep rid_list and time_list aligned one-to-one.
#         if stitched_rids and seg_geo and stitched_rids[-1] == seg_geo[0]:
#             seg_geo = seg_geo[1:]
#             seg_times = seg_times[1:]

#         stitched_rids.extend(seg_geo)
#         stitched_times.extend(
#             ts.strftime("%Y-%m-%dT%H:%M:%SZ")
#             for ts in seg_times
#         )

#     return stitched_rids, stitched_times

# Main MM CSV builder
# def build_mm_csvs(
#     fmm_path,
#     fid_to_geo: dict[int, int],
#     geo_to_length: dict[int, float],
#     trip_time_lookup: dict[str, list[pd.Timestamp]],
#     train_ratio: float,
#     random_state: int,
#     min_len: int = 2,
#     verbose: bool = False,
#     fmm_sep: str = ";",
#     min_delta_seconds: float = 1.0,
#     logger: logging.Logger | None = None,
# ) -> tuple[pd.DataFrame, pd.DataFrame]:
#     """
#     Build TS-TrajGen-compatible train/test CSVs from FMM map-matching output.

#     This function processes FMM trajectories (`tpath`) and reconstructs:
#     - `rid_list`: sequence of road segment IDs (geo_id space)
#     - `time_list`: aligned timestamps per road segment

#     The pipeline:
#     1. Loads FMM output.
#     2. Filters invalid trajectories (missing tpath or timestamps).
#     3. Converts `tpath` → `rid_list` using fid→geo mapping.
#     4. Interpolates timestamps per edge using length-weighted timing.
#     5. Filters short trajectories.
#     6. Splits into train/test sets.

#     Additionally, this function:
#     - Tracks preprocessing statistics (skipped trajectories).
#     - Tracks interpolation quality (fallback usage, bad segments).
#     - Logs progress and summary metrics.

#     Parameters
#     ----------
#     fmm_path : Path | str
#         Path to FMM output CSV file.
#     fid_to_geo : dict[int, int]
#         Mapping from FMM edge IDs (fid) to dataset road IDs (geo_id).
#     geo_to_length : dict[int, float]
#         Mapping from geo_id to edge length (used for time interpolation).
#     trip_time_lookup : dict[str, list[pd.Timestamp]]
#         Mapping from trajectory ID to ordered original GPS timestamps.
#     train_ratio : float
#         Fraction of trajectories to assign to the training set.
#     random_state : int
#         Random seed for reproducible train/test split.
#     min_len : int, optional
#         Minimum number of edges required for a valid trajectory.
#     verbose : bool, optional
#         If True, prints summary stats in addition to logging.
#     fmm_sep : str, optional
#         Delimiter used in the FMM CSV file.
#     min_delta_seconds : float, optional
#         Minimum enforced time difference between consecutive edges.
#     logger : logging.Logger, optional
#         Logger for progress, debug, and summary output.

#     Returns
#     -------
#     tuple[pd.DataFrame, pd.DataFrame]
#         Train and test DataFrames with columns:
#         - traj_id
#         - rid_list (comma-separated geo_ids)
#         - time_list (comma-separated ISO timestamps)

#     Notes
#     -----
#     - Interpolation may fall back to synthetic timelines when GPS timestamps
#     are insufficient (e.g., equal timestamps or too many edges).
#     - Logging includes detailed statistics on fallback usage and skipped data.
#     """
#     if logger is None:
#         logger = logging.getLogger(__name__)

#     logger.info(f"Loading FMM file: {fmm_path}")
#     fmm = pd.read_csv(fmm_path, sep=fmm_sep, engine="python")
#     logger.info(f"Loaded {len(fmm):,} FMM rows")

#     rows: list[dict[str, str]] = []

#     total = 0
#     skipped_empty_tpath = 0
#     skipped_short = 0
#     skipped_missing_times = 0

#     interp_stats = InterpolationStats()

#     progress = tqdm(
#         fmm.itertuples(index=False),
#         total=len(fmm),
#         desc="Building MM CSVs",
#         unit="traj",
#     )

#     for i, row in enumerate(progress):
#         total += 1
#         traj_id = str(row.id)

#         tpath_val = getattr(row, "tpath", "")

#         if pd.isna(tpath_val) or str(tpath_val).strip() == "":
#             skipped_empty_tpath += 1
#             continue

#         if traj_id not in trip_time_lookup or len(trip_time_lookup[traj_id]) < 2:
#             skipped_missing_times += 1
#             continue

#         rid_list, time_list = build_rid_and_time_lists_from_tpath(
#             traj_id=traj_id,
#             tpath_value=tpath_val,
#             trip_time_lookup=trip_time_lookup,
#             fid_to_geo=fid_to_geo,
#             geo_to_length=geo_to_length,
#             min_delta_seconds=min_delta_seconds,
#             stats=interp_stats,
#             logger=logger,
#         )

#         if len(rid_list) < min_len:
#             skipped_short += 1
#             continue

#         rows.append(
#             {
#                 "traj_id": traj_id,
#                 "rid_list": ",".join(str(x) for x in rid_list),
#                 "time_list": ",".join(time_list),
#             }
#         )

#         if i % 100 == 0:
#             skipped_total = (
#                 skipped_empty_tpath
#                 + skipped_missing_times
#                 + skipped_short
#             )

#             # set_postfix gives a live mini-dashboard without spamming logs.
#             progress.set_postfix(
#                 kept=len(rows),
#                 skipped=skipped_total,
#                 fallback=interp_stats.fallback_segments,
#                 fb_rate=f"{interp_stats.fallback_rate():.2%}",
#             )

#     progress.close()

#     df = pd.DataFrame(rows)

#     if df.empty:
#         logger.warning("No valid trajectories were kept. Returning empty train/test dataframes.")
#         interp_stats.log_summary(logger)
#         return df, df.copy()

#     train = df.sample(frac=train_ratio, random_state=random_state)
#     test = df.drop(train.index)

#     kept = len(df)
#     skipped_total = skipped_empty_tpath + skipped_missing_times + skipped_short

#     logger.info("=== MM CSV BUILD STATS ===")
#     logger.info(f"Total input trajectories: {total:,}")
#     logger.info(f"Kept trajectories:        {kept:,}")
#     logger.info(f"Kept ratio:               {kept / total:.3f}" if total else "Kept ratio: N/A")
#     logger.info(f"Skipped trajectories:     {skipped_total:,}")
#     logger.info("Skipped breakdown:")
#     logger.info(f"  Empty tpath:            {skipped_empty_tpath:,}")
#     logger.info(f"  Missing GPS times:      {skipped_missing_times:,}")
#     logger.info(f"  Too short final path:   {skipped_short:,}")
#     logger.info(f"Train size:               {len(train):,}")
#     logger.info(f"Test size:                {len(test):,}")

#     interp_stats.log_summary(logger)

#     if verbose:
#         print("\nMM CSV BUILD STATS")
#         print(f"Total input trajectories: {total:,}")
#         print(f"Kept trajectories:        {kept:,}")
#         print(f"Kept ratio:               {kept / total:.3f}" if total else "Kept ratio: N/A")
#         print()
#         print("Skipped breakdown:")
#         print(f"  Empty tpath:            {skipped_empty_tpath:,}")
#         print(f"  Missing GPS times:      {skipped_missing_times:,}")
#         print(f"  Too short final path:   {skipped_short:,}")
#         print()
#         print("INTERPOLATION SUMMARY")
#         print(f"  Total segments:         {interp_stats.total_segments:,}")
#         print(f"  Fallback segments:      {interp_stats.fallback_segments:,}")
#         print(f"  Fallback rate:          {interp_stats.fallback_rate():.3%}")
#         print(f"  Zero/reversed-time:     {interp_stats.non_positive_time_segments:,}")
#         print(f"  Short-time segments:    {interp_stats.short_time_segments:,}")
#         print(f"  Invalid-length:         {interp_stats.invalid_length_segments:,}")

#     return train.reset_index(drop=True), test.reset_index(drop=True)

# def interpolate_times_by_length(
#     start_time: pd.Timestamp,
#     end_time: pd.Timestamp,
#     edge_ids: list[int],
#     geo_to_length: dict[int, float],
# ) -> list[pd.Timestamp]:
#     """
#     Interpolate timestamps across a traversed edge segment using edge lengths as time weight.

#     The returned timestamps are edge-entry times. If the segment contains N edges,
#     this returns N timestamps aligned to those edges.

#     Parameters
#     ----------
#     start_time : pd.Timestamp
#         Timestamp of the starting GPS point.
#     end_time : pd.Timestamp
#         Timestamp of the ending GPS point.
#     edge_ids : list[int]
#         Traversed edge IDs for one GPS interval, already in geo_id space.
#     geo_to_length : dict[int, float]
#         Mapping from geo_id to edge length.

#     Returns
#     -------
#     list[pd.Timestamp]
#         One timestamp per edge in `edge_ids`.
#     """
#     if not edge_ids:
#         return []

#     if len(edge_ids) == 1:
#         return [start_time]

#     total_seconds = (end_time - start_time).total_seconds()

#     # Guard against equal or reversed timestamps.
#     if total_seconds <= 0:
#         return [start_time for _ in edge_ids]

#     lengths = [float(geo_to_length.get(edge_id, 1.0)) for edge_id in edge_ids]
#     total_length = sum(lengths)

#     # Fall back to equal spacing if lengths are invalid.
#     if total_length <= 0:
#         timeline = pd.date_range(start=start_time, end=end_time, periods=len(edge_ids) + 1)
#         return list(timeline[:-1])

#     edge_times: list[pd.Timestamp] = []
#     cumulative_length = 0.0

#     for length in lengths:
#         fraction = cumulative_length / total_length
#         edge_time = start_time + pd.to_timedelta(total_seconds * fraction, unit="s")
#         edge_times.append(edge_time)
#         cumulative_length += length

#     return edge_times

# 3A   + 3B   + 3C   + 3D = 12 = time
# 819, 268, 246, 56 =  lengths
#  0.58963282937365010799136069114471 = A
#  0.19294456443484521238300935925126 = B
#  0.17710583153347732181425485961123 = C
#  0.0403167746580273578113750899928 = D
# total len = 1389


# def build_rid_and_time_lists_from_tpath(
#     traj_id: str,
#     tpath_value: Any,
#     trip_time_lookup: dict[str, list[pd.Timestamp]],
#     fid_to_geo: dict[int, int],
#     geo_to_length: dict[int, float],
# ) -> tuple[list[int], list[str]]:
#     """
#     Build aligned `rid_list` and `time_list` for one trajectory from FMM `tpath`.

#     This function:
#     1. Parses `tpath` into per-GPS-interval edge segments.
#     2. Retrieves original GPS timestamps for the trajectory.
#     3. Maps FMM edge ids (fid) to dataset road ids (geo_id).
#     4. Interpolates one timestamp per edge using length-weighted timing.
#     5. Stitches all segments together while removing duplicated boundary edges.

#     Parameters
#     ----------
#     traj_id : str
#         FMM trajectory id.
#     tpath_value : Any
#         Raw `tpath` field from one FMM row.
#     trip_time_lookup : dict[str, list[pd.Timestamp]]
#         Mapping from trajectory id to ordered original GPS timestamps.
#     fid_to_geo : dict[int, int]
#         Mapping from FMM fid to geo_id.
#     geo_to_length : dict[int, float]
#         Mapping from geo_id to edge length.

#     Returns
#     -------
#     tuple[list[int], list[str]]
#         Final stitched rid_list and ISO-formatted time_list.

#     Notes
#     -----
#     `tpath` has one segment per GPS interval, so for N GPS points we expect
#     approximately N-1 segments. If counts do not align exactly, this function
#     safely truncates to the smaller usable count.
#     """
#     tpath_segments_fid = parse_tpath(tpath_value)
#     point_times = trip_time_lookup.get(str(traj_id), [])

#     # Need at least two GPS timestamps to define one interval.
#     if len(point_times) < 2 or not tpath_segments_fid:
#         return [], []

#     # Each tpath segment corresponds to interval i: GPS[i] -> GPS[i+1]
#     # so align lengths for both tpath_segments_fid with its gps timestamps
#     # if one is longer than other then these get truncated, to keep the 
#     # constraint that each edge gets a timestamp and vice versa 
#     usable_intervals = min(len(tpath_segments_fid), len(point_times) - 1)

#     stitched_rids: list[int] = []
#     stitched_times: list[str] = []

#     for i in range(usable_intervals):
#         seg_fids = tpath_segments_fid[i]
#         if not seg_fids:
#             continue

#         # Map FMM fids to geo_ids and drop any unknowns.
#         seg_geo = [fid_to_geo[fid] for fid in seg_fids if fid in fid_to_geo]
#         if not seg_geo:
#             continue

#         seg_start = point_times[i]
#         seg_end = point_times[i + 1]

#         seg_times = interpolate_times_by_length(
#             start_time=seg_start,
#             end_time=seg_end,
#             edge_ids=seg_geo,
#             geo_to_length=geo_to_length,
#         )

#         # Remove duplicated boundary edge if this segment starts with the same edge
#         # the previous segment ended with.
#         if stitched_rids and seg_geo and stitched_rids[-1] == seg_geo[0]:
#             seg_geo = seg_geo[1:]
#             seg_times = seg_times[1:]

#         stitched_rids.extend(seg_geo)
#         stitched_times.extend(ts.strftime("%Y-%m-%dT%H:%M:%SZ") for ts in seg_times)

#     return stitched_rids, stitched_times

def build_trip_time_lookup(
    parquet_path: Path,
    trip_id_map_csv: Path,
    user_id_col: str = "uid",
    trajectory_id_col: str = "tid",
    datetime_col: str = "datetime"
) -> dict[str, list[pd.Timestamp]]:
    """
    Build a lookup from FMM trajectory id to ordered original GPS timestamps.

    Parameters
    ----------
    parquet_path : str
        Path to original cleaned parquet with datetime, uid, tid.
    trip_id_map_csv : str
        CSV created during FMM input preparation mapping trip_key -> FMM id.

    Returns
    -------
    dict[str, list[pd.Timestamp]]
        Mapping from FMM trajectory id string to timestamp list.
    """
    df = pd.read_parquet(parquet_path, columns=[datetime_col, user_id_col, trajectory_id_col]).copy()
    df["trip_key"] = df[user_id_col].astype(str) + "_" + df[trajectory_id_col].astype(str)

    trip_map = pd.read_csv(trip_id_map_csv).copy()
    trip_map["id"] = trip_map["id"].astype(str)

    df = df.merge(trip_map, on="trip_key", how="left")
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(["id", datetime_col]).reset_index(drop=True)

    lookup: dict[str, list[pd.Timestamp]] = {}
    for traj_id, group in df.groupby("id", sort=False):
        lookup[str(traj_id)] = list(group[datetime_col])

    return lookup

def build_geo_and_length_lookups(network_path: Path):
    """
    Build geo-related lookup structures from the road network file.

    Build `.geo` file and mapping from fid -> geo_id.

    The `.geo` file represents road segments with geometry.

    Parameters
    ----------
    network_path : str
        Path to the road network file used to build `.geo`.

    Returns
    -------
    geo_df : pd.DataFrame
        DataFrame ready to save as `.geo`.
    edges_df : pd.DataFrame
        Original edges with added `geo_id`.
    fid_to_geo : dict[int, int]
        Mapping from FMM edge IDs (fid) to geo_id.
    geo_to_length : dict[int, float]]
        geo_to_length mappings
    """
    # edges = gpd.read_file(network_path).copy()
    # edges = edges.sort_values("fid").reset_index(drop=True)
    # edges["geo_id"] = range(len(edges))

    # fid_to_geo = dict(zip(edges["fid"].astype(int), edges["geo_id"].astype(int)))

    geo_df, edges_df, fid_to_geo = build_geo(network_path)

    if "length" in edges_df.columns:
        geo_to_length = dict(zip(edges_df["geo_id"].astype(int), edges_df["length"].astype(float)))
    else:
        # Fallback if length is missing.
        geo_to_length = {int(geo_id): 1.0 for geo_id in edges_df["geo_id"]}

    # return fid_to_geo, geo_to_length
    return geo_df, edges_df, fid_to_geo, geo_to_length