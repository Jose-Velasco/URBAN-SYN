import ast
import json
from typing import Any
import re

import pandas as pd
import geopandas as gpd
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from dateutil import tz
from collections.abc import Iterable

from dataclass_models import BuildCsvStats, DuplicateAction, EdgeTimePoint, InterpolationStats

from pint import UnitRegistry
from pint.errors import DimensionalityError, UndefinedUnitError

UNIT_REGISTRY = UnitRegistry()

def _parse_single_osm_width(value: Any) -> float | None:
    """
    Parse one OSM width value and return meters.

    Bare numeric OSM width values are interpreted as meters.
    Explicit unit conversion is delegated to Pint.
    """
    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    # OSM bare width values are meters.
    try:
        return float(text)
    except ValueError:
        pass

    # OSM feet/inches syntax, e.g. 25'8"
    match = re.fullmatch(
        r"""(\d+(?:\.\d+)?)'\s*(\d+(?:\.\d+)?)"?""",
        text,
    )

    if match:
        feet = UNIT_REGISTRY.Quantity(
            float(match.group(1)),
            "foot",
        )
        inches = UNIT_REGISTRY.Quantity(
            float(match.group(2)),
            "inch",
        )

        return float(
            (feet + inches).to("meter").magnitude # pyright: ignore[reportAttributeAccessIssue]
        )

    try:
        quantity = UNIT_REGISTRY.Quantity(text)
        return float(
            quantity.to("meter").magnitude
        )
    except (
        ValueError,
        TypeError,
        DimensionalityError,
        UndefinedUnitError,
    ):
        return None

def parse_osm_width(value: Any) -> float | None:
    """
    Convert an OSM width attribute to meters.

    Serialized multi-valued widths are parsed individually and averaged.
    """
    if pd.isna(value):
        return None

    text = str(value).strip()

    values: list[Any]

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)

            if isinstance(parsed, (list, tuple)):
                values = list(parsed)
            else:
                values = [value]

        except (ValueError, SyntaxError):
            values = [value]
    else:
        values = [value]

    parsed_widths = [
        width
        for item in values
        if (width := _parse_single_osm_width(item)) is not None
    ]

    if not parsed_widths:
        return None

    return float(
        sum(parsed_widths) / len(parsed_widths)
    )

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
def build_geo(network_path: Path, feature_columns: Iterable[str]):
    """
    Build the TS-TrajGen `.geo` table and mapping from canonical edge_id -> geo_id mapping.

    The `.geo` file represents road segments with geometry with added  feature_columns from road network road features.

    Parameters
    ----------
    network_path : Path
        Path to shapefile / geopackage containing road edges. (canonical road-network GeoPackage)
    
    feature_columns: Iterable[str]
        iterable of road feature names in the road network to add to the output .geo file

    Returns
    -------
    geo_df : pd.DataFrame
        DataFrame ready to save as `.geo`.
    edges_df : pd.DataFrame
        Original edges with added `geo_id`. Canonical road edges with an added TS-TrajGen `geo_id`.
    edge_id_to_geo_id : dict[int, int]
        Mapping from canonical/FMM `edge_id` values to TS-TrajGen `geo_id`.
    """
    edges = gpd.read_file(network_path).copy()

    required = {"edge_id", "u", "v", "geometry"}
    missing = required - set(edges.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    edges = edges.sort_values("edge_id").reset_index(drop=True)
    edges["geo_id"] = range(len(edges))

    edge_id_to_geo_id = dict(zip(edges["edge_id"].astype(int), edges["geo_id"].astype(int)))

    geo_df = pd.DataFrame({
        "geo_id": edges["geo_id"].astype(int),
        "type": "LineString",
        "coordinates": edges["geometry"].apply(geometry_to_coordinate_string),
    })

    for column in feature_columns:
        if column not in edges.columns:
            raise ValueError(
                f"Configured .geo feature column is missing from road network: {column}"
            )

        geo_df[column] = edges[column]

    return geo_df, edges, edge_id_to_geo_id


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
    geo_ids: list[int],
    geo_to_length: dict[int, float],
    stats: InterpolationStats | None = None,
    ) -> list[float]:
    """
    Return non-negative edge lengths with fallback defaults.

    Missing lengths default to 1.0 so interpolation can still run.
    """
    lengths = [
        max(float(geo_to_length.get(geo_id, 1.0)), 0.0)
        for geo_id in geo_ids
    ]

    if sum(lengths) <= 0:
        if stats is not None:
            stats.invalid_length_segments += 1

        # Equal weights avoid divide-by-zero while preserving ordering.
        return [1.0] * len(geo_ids)

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

def _interpolate_middle_edge_points(
    start_time: pd.Timestamp,
    total_seconds: float,
    geo_ids: list[int],
    lengths: list[float],
    ) -> list[EdgeTimePoint]:
    """
    Interpolate timestamps for middle edges only.

    First and last edges are handled separately because they preserve GPS
    anchor timestamps.
    """
    if len(geo_ids) <= 2:
        return []

    total_length = sum(lengths)
    
    # Start cumulative distance at the first edge so interpolation for middle
    # edges reflects distance traveled *before entering* each edge.
    cumulative_length = lengths[0]
    middle_points: list[EdgeTimePoint] = []

    for edge_id, length in zip(geo_ids[1:-1], lengths[1:-1]):
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
                geo_id=edge_id,
                timestamp=timestamp,
                is_anchor=False,
            )
        )
        cumulative_length += length

    return middle_points

def interpolate_edge_time_points_by_length(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    geo_ids: list[int],
    geo_to_length: dict[int, float],
    min_delta_seconds: float = 1.0,
    stats: InterpolationStats | None = None,
    logger: logging.Logger | None = None,
    ) -> list[EdgeTimePoint]:
    """
    Assign timestamps to road segments while preserving GPS timestamps as anchors.

    The first road segment receives `start_time` as a GPS anchor. The last segment
    receives `end_time` as a GPS anchor when the segment contains two or more roads.
    Middle segments receive length-weighted interpolated timestamps.

    If the GPS interval is non-positive or shorter than the diagnostic
    `min_delta_seconds` threshold, the original GPS bounds are preserved and the
    condition is recorded in `stats`.
    """
    if not geo_ids:
        return []

    if stats is not None:
        stats.total_segments += 1

    start_time = _safe_timestamp(start_time)
    end_time = _safe_timestamp(end_time)

    if len(geo_ids) == 1:
        return [EdgeTimePoint(geo_ids[0], start_time, is_anchor=True)]

    end_time, total_seconds = _use_fallback_interval(
        start_time=start_time,
        end_time=end_time,
        edge_count=len(geo_ids),
        min_delta_seconds=min_delta_seconds,
        stats=stats,
        logger=logger,
    )

    lengths = _get_edge_lengths(geo_ids, geo_to_length, stats=stats)

    return [
        EdgeTimePoint(geo_ids[0], start_time, is_anchor=True),
        *_interpolate_middle_edge_points(
            start_time=start_time,
            total_seconds=total_seconds,
            geo_ids=geo_ids,
            lengths=lengths,
        ),
        EdgeTimePoint(geo_ids[-1], end_time, is_anchor=True),
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
    if prev.geo_id != curr.geo_id:
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

def _map_edge_ids_to_geo_ids(
    edge_ids: list[int],
    edge_id_to_geo_id: dict[int, int],
) -> list[int]:
    """
    Convert canonical/FMM edge IDs to TS-TrajGen geo_id values.

    Unknown edge IDs are skipped so one missing edge does not discard the
    entire trajectory.
    """
    return [
        edge_id_to_geo_id[edge_id]
        for edge_id in edge_ids
        if edge_id in edge_id_to_geo_id
    ]

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
    rid_list = [point.geo_id for point in points]
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
    tpath_segments_edge_id: list[list[int]],
    point_times: list[pd.Timestamp],
    ):
    """
    Yield tpath segments aligned to GPS timestamp intervals.

    FMM and raw GPS counts can disagree, so this safely truncates to the
    smallest valid interval count.
    """
    usable_intervals = min(len(tpath_segments_edge_id), len(point_times) - 1)

    for i in range(usable_intervals):
        yield i, tpath_segments_edge_id[i], point_times[i], point_times[i + 1]

def build_rid_and_time_lists_from_tpath(
    traj_id: str,
    tpath_value: Any,
    trip_time_lookup: dict[str, list[pd.Timestamp]],
    edge_id_to_geo_id: dict[int, int],
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
    tpath_segments_edge_id = parse_tpath(tpath_value)
    point_times = trip_time_lookup.get(str(traj_id), [])

    if len(point_times) < 2 or not tpath_segments_edge_id:
        return [], []

    stitched_points: list[EdgeTimePoint] = []

    for _, seg_edge_ids, start_time, end_time in _iter_usable_tpath_intervals(
        tpath_segments_edge_id,
        point_times,
    ):
        if not seg_edge_ids:
            continue

        seg_geo_ids = _map_edge_ids_to_geo_ids(seg_edge_ids, edge_id_to_geo_id)

        if not seg_geo_ids:
            continue

        segment_points = interpolate_edge_time_points_by_length(
            start_time=start_time,
            end_time=end_time,
            geo_ids=seg_geo_ids,
            geo_to_length=geo_to_length,
            min_delta_seconds=min_delta_seconds,
            stats=stats,
            logger=logger,
        )

        for point in segment_points:
            _append_with_duplicate_policy(stitched_points, point, stats=stats)

    return _format_time_points(stitched_points)

def build_rid_and_time_lists_from_opath(
    traj_id: str,
    opath_value: Any,
    trip_time_lookup: dict[str, list[pd.Timestamp]],
    edge_id_to_geo_id: dict[int, int],
    logger: logging.Logger | None = None,
) -> tuple[list[int], list[str]]:
    """
    Build anchor-only road/time sequences from FMM `opath`.

    Each FMM `opath` road corresponds directly to one GPS observation.
    Consecutive duplicate roads are intentionally preserved to maintain
    one-to-one alignment with GPS timestamps and future point metadata.
    """
    edge_ids = parse_cpath_like(opath_value)
    point_times = trip_time_lookup.get(str(traj_id), [])

    if not edge_ids or not point_times:
        return [], []

    # For anchor-only mode we require exact alignment. Silently truncating
    # would break future GPS-feature alignment.
    if len(edge_ids) != len(point_times):
        if logger is not None:
            logger.warning(
                "Trajectory %s has opath/GPS length mismatch: "
                "opath=%d, gps_points=%d",
                traj_id,
                len(edge_ids),
                len(point_times),
            )

        return [], []

    missing_edge_ids = [
        edge_id
        for edge_id in edge_ids
        if edge_id not in edge_id_to_geo_id
    ]

    if missing_edge_ids:
        if logger is not None:
            logger.warning(
                "Trajectory %s contains %d opath edges missing from canonical network",
                traj_id,
                len(missing_edge_ids),
            )

        return [], []

    geo_ids = [
        edge_id_to_geo_id[edge_id]
        for edge_id in edge_ids
    ]

    time_list = [
        _format_timestamp(_safe_timestamp(timestamp))
        for timestamp in point_times
    ]

    return geo_ids, time_list

# build_mm_csvs helpers

def _has_valid_times(
    traj_id: str,
    trip_time_lookup: dict[str, list[pd.Timestamp]],
    ) -> bool:
    """
    Return True if a trajectory has at least two GPS timestamps.
    """
    return traj_id in trip_time_lookup and len(trip_time_lookup[traj_id]) >= 2

def _has_valid_path(row: Any, path_column: str) -> bool:
    """Return True if the selected FMM path field is non-empty."""
    value = getattr(row, path_column, "")
    return not (pd.isna(value) or not str(value).strip())

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
        "rid_list": ",".join(str(geo_id) for geo_id in rid_list),
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
    print(f"  Empty path:            {build_stats.skipped_empty_path:,}")
    print(f"  Single-point trajectories:      {build_stats.skipped_missing_times:,}")
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

def _limit_trajectories(
    df: pd.DataFrame,
    max_trajectories: int | None,
    random_state: int,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Deterministically limit the number of trajectories in a split."""
    if max_trajectories is None:
        return df.reset_index(drop=True)

    if max_trajectories <= 0:
        raise ValueError("max_trajectories must be greater than 0.")

    if len(df) <= max_trajectories:
        return df.reset_index(drop=True)

    if logger:
        logger.info(f"Applied trajectory limits: {max_trajectories = }")
    return (
        df.sample(
            n=max_trajectories,
            random_state=random_state,
        )
        .reset_index(drop=True)
    )
def build_mm_csvs(
    fmm_path: Path | str,
    edge_id_to_geo_id: dict[int, int],
    geo_to_length: dict[int, float],
    trip_time_lookup: dict[str, list[pd.Timestamp]],
    train_ratio: float,
    random_state: int,
    interpolate_intermediate_edges: bool,
    min_len: int = 2,
    verbose: bool = False,
    fmm_sep: str = ";",
    min_delta_seconds: float = 1.0,
    logger: logging.Logger | None = None,
    max_train_trajectories: int | None = None,
    max_test_trajectories: int | None = None
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
    edge_id_to_geo_id : dict[int, int]
        Mapping from canonical/FMM `edge_id` to TS-TrajGen dataset road ID `geo_id`.
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

        path_column = (
            "tpath"
            if interpolate_intermediate_edges
            else "opath"
        )

        if not _has_valid_path(row, path_column):
            build_stats.skipped_empty_path += 1
            continue

        if not _has_valid_times(traj_id, trip_time_lookup):
            build_stats.skipped_missing_times += 1
            continue

        if interpolate_intermediate_edges:
            rid_list, time_list = build_rid_and_time_lists_from_tpath(
                traj_id=traj_id,
                tpath_value=getattr(row, "tpath"),
                trip_time_lookup=trip_time_lookup,
                edge_id_to_geo_id=edge_id_to_geo_id,
                geo_to_length=geo_to_length,
                min_delta_seconds=min_delta_seconds,
                stats=interp_stats,
                logger=logger,
            )
        else:
            rid_list, time_list = build_rid_and_time_lists_from_opath(
                traj_id=traj_id,
                opath_value=getattr(row, "opath"),
                trip_time_lookup=trip_time_lookup,
                edge_id_to_geo_id=edge_id_to_geo_id,
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

    train = _limit_trajectories(
        train,
        max_train_trajectories,
        random_state,
    )

    test = _limit_trajectories(
        test,
        max_test_trajectories,
        random_state,
    )

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

def build_geo_and_length_lookups(network_path: Path, feature_columns: Iterable[str]):
    """
    Build geo-related lookup structures from the canonical road-network GeoPackage.

    Build `.geo` file and mapping from edge_id -> geo_id.

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
        Canonical road-network edges with added `geo_id`.
    edge_id_to_geo_id : dict[int, int]
        Mapping from project/FMM `edge_id` to TS-TrajGen `geo_id`.
    geo_to_length : dict[int, float]]
        geo_to_length mappings. Mapping from TS-TrajGen `geo_id` to road length.
    """
    geo_df, edges_df, edge_id_to_geo_id = build_geo(network_path, feature_columns)

    if "length" in edges_df.columns:
        geo_to_length = dict(zip(edges_df["geo_id"].astype(int), edges_df["length"].astype(float)))
    else:
        # Fallback if length is missing.
        geo_to_length = {int(geo_id): 1.0 for geo_id in edges_df["geo_id"]}

    # return edge_id_to_geo, geo_to_length
    return geo_df, edges_df, edge_id_to_geo_id, geo_to_length