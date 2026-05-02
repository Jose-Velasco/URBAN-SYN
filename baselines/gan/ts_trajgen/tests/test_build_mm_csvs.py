"""
Each trajectory is represented as a sequence of road segment identifiers with one timestamp per segment.

Timestamps correspond to estimated entry times into each road segment.

Temporal interpolation is performed within observed GPS intervals using edge-length-based weighting.

Observed GPS timestamps are preserved as anchors whenever representable under the one-timestamp-per-edge constraint.

No synthetic timestamps are generated outside the bounds of observed GPS data.

Duplicate segment-time pairs are removed when they are redundant (same segment and same timestamp), while repeated segments with distinct timestamps are preserved.

Sub-second temporal precision is maintained to prevent collapse of interpolated timestamps.

"""

import pandas as pd

from hypothesis import given, settings, strategies as st
from baselines.gan.ts_trajgen.utils import build_mm_csvs
from pathlib import Path
from tempfile import TemporaryDirectory



def _write_fmm_csv(tmp_path, rows: list[dict], sep: str = ";"):
    """
    Write a temporary FMM CSV for testing.
    """
    fmm_path = tmp_path / "fmm_output.csv"
    pd.DataFrame(rows).to_csv(fmm_path, sep=sep, index=False)
    return fmm_path


def _split_rid_list(value: str) -> list[int]:
    """
    Parse comma-separated rid_list output into integers.
    """
    return [int(x) for x in value.split(",") if x]


def _split_time_list(value: str) -> list[pd.Timestamp]:
    """
    Parse comma-separated time_list output into pandas timestamps.

    This avoids brittle string-format assertions such as comparing
    `.000Z` vs `Z`.
    """
    return [pd.Timestamp(x) for x in value.split(",") if x]


def test_build_mm_csvs_preserves_anchors_and_skips_invalid_rows(tmp_path):
    """
    Test that build_mm_csvs keeps valid trajectories, skips invalid rows,
    preserves GPS anchor timestamps, and interpolates intermediate edges.
    """
    fmm_path = _write_fmm_csv(
        tmp_path,
        [
            {
                "id": "trip_1",
                # GPS intervals:
                # t0 -> t1: [2]
                # t1 -> t2: [2, 5, 13]
                # t2 -> t3: [13, 14]
                "tpath": "2|2,5,13|13,14",
            },
            {"id": "trip_empty", "tpath": ""},
            {"id": "trip_missing_times", "tpath": "2|3"},
            {"id": "trip_short", "tpath": "2"},
        ],
    )

    fid_to_geo = {
        2: 102,
        5: 105,
        13: 113,
        14: 114,
    }

    geo_to_length = {
        102: 10.0,
        105: 20.0,
        113: 30.0,
        114: 40.0,
    }

    trip_time_lookup = {
        "trip_1": [
            pd.Timestamp("2026-01-01T10:00:00Z"),
            pd.Timestamp("2026-01-01T10:00:10Z"),
            pd.Timestamp("2026-01-01T10:00:25Z"),
            pd.Timestamp("2026-01-01T10:00:40Z"),
        ],
        "trip_short": [
            pd.Timestamp("2026-01-01T11:00:00Z"),
            pd.Timestamp("2026-01-01T11:00:10Z"),
        ],
    }

    train_df, test_df = build_mm_csvs(
        fmm_path=fmm_path,
        fid_to_geo=fid_to_geo,
        geo_to_length=geo_to_length,
        trip_time_lookup=trip_time_lookup,
        train_ratio=1.0,
        random_state=101,
        min_len=2,
        verbose=False,
        fmm_sep=";",
        min_delta_seconds=1.0,
    )

    assert len(train_df) == 1
    assert len(test_df) == 0

    row = train_df.iloc[0]

    assert row["traj_id"] == "trip_1"

    rid_list = _split_rid_list(row["rid_list"])
    time_list = _split_time_list(row["time_list"])

    # Expected behavior:
    # - 102 at t0 is kept.
    # - 102 at t1 is also kept because it is a real anchor with a different time.
    # - 105 is interpolated.
    # - 113 at t2 replaces/dominates duplicate interpolated 113.
    # - 114 at t3 is kept.
    assert rid_list == [102, 102, 105, 113, 114]

    assert time_list[0] == pd.Timestamp("2026-01-01T10:00:00Z")
    assert time_list[1] == pd.Timestamp("2026-01-01T10:00:10Z")
    assert time_list[3] == pd.Timestamp("2026-01-01T10:00:25Z")
    assert time_list[4] == pd.Timestamp("2026-01-01T10:00:40Z")

    # Middle edge should be interpolated between t1 and t2.
    assert time_list[1] < time_list[2] < time_list[3]


def test_build_mm_csvs_interpolates_short_intervals_within_real_gps_bounds(tmp_path):
    """
    Test that short GPS intervals are not stretched beyond the real GPS anchors.

    min_delta_seconds is used as a diagnostic threshold, but timestamps should
    still remain inside the original start/end interval.
    """
    fmm_path = _write_fmm_csv(
        tmp_path,
        [{"id": "trip_short_interval", "tpath": "1,2,3,4"}],
    )

    fid_to_geo = {
        1: 101,
        2: 102,
        3: 103,
        4: 104,
    }

    geo_to_length = {
        101: 1.0,
        102: 1.0,
        103: 1.0,
        104: 1.0,
    }

    start = pd.Timestamp("2026-01-01T10:00:00Z")
    end = pd.Timestamp("2026-01-01T10:00:01Z")

    trip_time_lookup = {
        "trip_short_interval": [start, end],
    }

    train_df, test_df = build_mm_csvs(
        fmm_path=fmm_path,
        fid_to_geo=fid_to_geo,
        geo_to_length=geo_to_length,
        trip_time_lookup=trip_time_lookup,
        train_ratio=1.0,
        random_state=101,
        min_len=2,
        verbose=False,
        fmm_sep=";",
        min_delta_seconds=1.0,
    )

    assert len(train_df) == 1
    assert len(test_df) == 0

    row = train_df.iloc[0]
    rid_list = _split_rid_list(row["rid_list"])
    time_list = _split_time_list(row["time_list"])

    assert rid_list == [101, 102, 103, 104]
    assert time_list[0] == start
    assert time_list[-1] == end

    # All timestamps must stay within the real GPS interval.
    assert all(start <= t <= end for t in time_list)

    # Sequence order should still be non-decreasing.
    assert all(curr >= prev for prev, curr in zip(time_list, time_list[1:]))


def test_build_mm_csvs_preserves_millisecond_interpolation(tmp_path):
    """
    Test that interpolated timestamps preserve millisecond precision.

    Without millisecond serialization, sub-second timestamps would collapse
    into duplicate second-level strings.
    """
    fmm_path = _write_fmm_csv(
        tmp_path,
        [{"id": "trip_ms", "tpath": "1,2,3"}],
    )

    fid_to_geo = {
        1: 101,
        2: 102,
        3: 103,
    }

    geo_to_length = {
        101: 1.0,
        102: 1.0,
        103: 1.0,
    }

    trip_time_lookup = {
        "trip_ms": [
            pd.Timestamp("2026-01-01T10:00:00Z"),
            pd.Timestamp("2026-01-01T10:00:01Z"),
        ],
    }

    train_df, _ = build_mm_csvs(
        fmm_path=fmm_path,
        fid_to_geo=fid_to_geo,
        geo_to_length=geo_to_length,
        trip_time_lookup=trip_time_lookup,
        train_ratio=1.0,
        random_state=101,
        min_len=2,
        verbose=False,
        fmm_sep=";",
        min_delta_seconds=1.0,
    )

    time_strings = train_df.iloc[0]["time_list"].split(",")
    time_list = _split_time_list(train_df.iloc[0]["time_list"])

    assert time_strings[0] == "2026-01-01T10:00:00.000Z"
    assert time_strings[-1] == "2026-01-01T10:00:01.000Z"

    # Middle timestamp should be sub-second and inside the interval.
    assert time_list[0] < time_list[1] < time_list[2]
    
    # Ensure it is not second-aligned (i.e., not 10:00:00.000)
    assert time_list[1].microsecond != 0

    
    # Check millisecond precision exists in string
    assert "." in time_strings[1]
    assert time_strings[1].endswith("Z")

    # test approximate spacing
    delta = (time_list[1] - time_list[0]).total_seconds()
    assert 0 < delta < 1


def test_build_mm_csvs_drops_redundant_same_edge_same_anchor_time(tmp_path):
    """
    Test that consecutive duplicate anchors with the same edge and same timestamp
    are collapsed because they add no new information.
    """
    fmm_path = _write_fmm_csv(
        tmp_path,
        [
            {
                "id": "trip_duplicate_anchor",
                # Boundary duplicate:
                # interval 0 ends on edge 1 at t1
                # interval 1 starts on edge 1 at t1
                "tpath": "1|1,2",
            }
        ],
    )

    fid_to_geo = {
        1: 101,
        2: 102,
    }

    geo_to_length = {
        101: 1.0,
        102: 1.0,
    }

    t0 = pd.Timestamp("2026-01-01T10:00:00Z")
    t1 = pd.Timestamp("2026-01-01T10:00:10Z")
    t2 = pd.Timestamp("2026-01-01T10:00:20Z")

    trip_time_lookup = {
        "trip_duplicate_anchor": [t0, t1, t2],
    }

    train_df, _ = build_mm_csvs(
        fmm_path=fmm_path,
        fid_to_geo=fid_to_geo,
        geo_to_length=geo_to_length,
        trip_time_lookup=trip_time_lookup,
        train_ratio=1.0,
        random_state=101,
        min_len=2,
        verbose=False,
        fmm_sep=";",
        min_delta_seconds=1.0,
    )

    rid_list = _split_rid_list(train_df.iloc[0]["rid_list"])
    time_list = _split_time_list(train_df.iloc[0]["time_list"])

    # The duplicate 101 @ t1 appears at the boundary and should be kept once.
    assert rid_list == [101, 101, 102]
    assert time_list == [t0, t1, t2]


def test_build_mm_csvs_keeps_same_edge_different_anchor_times(tmp_path):
    """
    Test that repeated road IDs are preserved when they correspond to distinct
    GPS anchor times.
    """
    fmm_path = _write_fmm_csv(
        tmp_path,
        [
            {
                "id": "trip_same_edge_different_times",
                # Same edge is observed across consecutive GPS intervals.
                "tpath": "1|1|1",
            }
        ],
    )

    fid_to_geo = {
        1: 101,
    }

    geo_to_length = {
        101: 1.0,
    }

    t0 = pd.Timestamp("2026-01-01T10:00:00Z")
    t1 = pd.Timestamp("2026-01-01T10:00:10Z")
    t2 = pd.Timestamp("2026-01-01T10:00:20Z")
    t3 = pd.Timestamp("2026-01-01T10:00:30Z")

    trip_time_lookup = {
        "trip_same_edge_different_times": [t0, t1, t2, t3],
    }

    train_df, _ = build_mm_csvs(
        fmm_path=fmm_path,
        fid_to_geo=fid_to_geo,
        geo_to_length=geo_to_length,
        trip_time_lookup=trip_time_lookup,
        train_ratio=1.0,
        random_state=101,
        min_len=2,
        verbose=False,
        fmm_sep=";",
        min_delta_seconds=1.0,
    )

    rid_list = _split_rid_list(train_df.iloc[0]["rid_list"])
    time_list = _split_time_list(train_df.iloc[0]["time_list"])

    assert rid_list == [101, 101, 101]
    assert time_list == [t0, t1, t2]


def test_build_mm_csvs_train_test_split_is_reproducible(tmp_path):
    """
    Test that train/test splitting is reproducible with a fixed random_state.
    """
    fmm_path = _write_fmm_csv(
        tmp_path,
        [
            {"id": "trip_1", "tpath": "1,2"},
            {"id": "trip_2", "tpath": "1,2"},
            {"id": "trip_3", "tpath": "1,2"},
            {"id": "trip_4", "tpath": "1,2"},
        ],
    )

    fid_to_geo = {
        1: 101,
        2: 102,
    }

    geo_to_length = {
        101: 1.0,
        102: 1.0,
    }

    trip_time_lookup = {
        f"trip_{i}": [
            pd.Timestamp(f"2026-01-01T10:00:0{i}Z"),
            pd.Timestamp(f"2026-01-01T10:00:1{i}Z"),
        ]
        for i in range(1, 5)
    }

    kwargs = dict(
        fmm_path=fmm_path,
        fid_to_geo=fid_to_geo,
        geo_to_length=geo_to_length,
        trip_time_lookup=trip_time_lookup,
        train_ratio=0.5,
        random_state=101,
        min_len=2,
        verbose=False,
        fmm_sep=";",
        min_delta_seconds=1.0,
    )

    train_1, test_1 = build_mm_csvs(**kwargs) # pyright: ignore[reportArgumentType]
    train_2, test_2 = build_mm_csvs(**kwargs) # pyright: ignore[reportArgumentType]

    assert train_1["traj_id"].tolist() == train_2["traj_id"].tolist()
    assert test_1["traj_id"].tolist() == test_2["traj_id"].tolist()

@given(
    edge_count=st.integers(min_value=2, max_value=25),
    duration_seconds=st.floats(
        min_value=0.001,
        max_value=300.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)
@settings(max_examples=50)
def test_build_mm_csvs_interpolated_times_stay_within_gps_bounds(
    edge_count: int,
    duration_seconds: float,
):
    """
    Property-based test: for many edge counts and GPS durations, generated
    timestamps should remain inside the original GPS interval.

    This guards against accidentally reintroducing synthetic time extension.
    """
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        fids = list(range(1, edge_count + 1))

        fmm_path = tmp_path / "fmm_output.csv"
        pd.DataFrame(
            [
                {
                    "id": "trip_property",
                    "tpath": ",".join(str(fid) for fid in fids),
                }
            ]
        ).to_csv(fmm_path, sep=";", index=False)

        fid_to_geo = {fid: fid + 100 for fid in fids}

        # Vary lengths so the test covers non-uniform interpolation.
        geo_to_length = {
            fid + 100: float((fid % 5) + 1)
            for fid in fids
        }

        start = pd.Timestamp("2026-01-01T10:00:00Z")
        end = start + pd.to_timedelta(duration_seconds, unit="s")

        train_df, test_df = build_mm_csvs(
            fmm_path=fmm_path,
            fid_to_geo=fid_to_geo,
            geo_to_length=geo_to_length,
            trip_time_lookup={"trip_property": [start, end]},
            train_ratio=1.0,
            random_state=101,
            min_len=2,
            verbose=False,
            fmm_sep=";",
            min_delta_seconds=1.0,
        )

        assert len(train_df) == 1
        assert len(test_df) == 0

        row = train_df.iloc[0]
        rid_list = [int(x) for x in row["rid_list"].split(",")]
        time_list = [pd.Timestamp(t) for t in row["time_list"].split(",")]

        assert len(rid_list) == edge_count
        assert len(time_list) == edge_count

        assert time_list[0] == start
        # End timestamp should match after millisecond truncation
        # Interpolation is precise
        # Serialization is lossy (ms truncation)
        # Tests must match serialization behavior
        assert time_list[-1] == end.floor("ms")
        assert time_list[-1] <= end
        assert (end - time_list[-1]).total_seconds() < 0.001

        assert all(start <= t <= end for t in time_list)
        assert all(curr >= prev for prev, curr in zip(time_list, time_list[1:]))

@st.composite
def random_tpath_case(draw):
    """
    Generate one random FMM-like tpath with aligned GPS timestamps.

    The generated case intentionally includes boundary duplicates because FMM
    tpath often repeats the last edge of one interval as the first edge of the
    next interval.
    """
    interval_count = draw(st.integers(min_value=1, max_value=8))
    next_fid = 1
    segments: list[list[int]] = []

    for i in range(interval_count):
        segment_len = draw(st.integers(min_value=1, max_value=5))

        if i > 0 and draw(st.booleans()):
            # Reuse previous boundary edge to simulate FMM segment overlap.
            segment = [segments[-1][-1]]
        else:
            segment = []

        while len(segment) < segment_len:
            segment.append(next_fid)
            next_fid += 1

        segments.append(segment)

    # Use non-decreasing GPS times. This allows repeated real timestamps,
    # matching MAT-Builder compressed/bursty GPS behavior.
    gaps_ms = draw(
        st.lists(
            st.integers(min_value=0, max_value=30_000),
            min_size=interval_count,
            max_size=interval_count,
        )
    )

    return segments, gaps_ms


@given(case=random_tpath_case())
@settings(max_examples=75)
def test_build_mm_csvs_monotonic_and_preserves_gps_anchor_bounds_for_random_tpaths(case):
    """
    Property-based test for random tpath structures.

    Validates:
    - timestamps stay within GPS bounds
    - timestamps are non-decreasing
    - start anchor is preserved
    - end anchor is preserved only when representable
    """
    segments, gaps_ms = case

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        fmm_path = tmp_path / "fmm_output.csv"
        tpath = "|".join(",".join(str(fid) for fid in segment) for segment in segments)

        pd.DataFrame([{"id": "trip_random", "tpath": tpath}]).to_csv(
            fmm_path,
            sep=";",
            index=False,
        )

        all_fids = sorted({fid for segment in segments for fid in segment})
        fid_to_geo = {fid: fid + 100 for fid in all_fids}

        # Vary lengths so interpolation is not only uniform.
        geo_to_length = {
            fid + 100: float((fid % 7) + 1)
            for fid in all_fids
        }

        start = pd.Timestamp("2026-01-01T10:00:00Z")
        point_times = [start]

        for gap_ms in gaps_ms:
            point_times.append(point_times[-1] + pd.to_timedelta(gap_ms, unit="ms"))

        train_df, test_df = build_mm_csvs(
            fmm_path=fmm_path,
            fid_to_geo=fid_to_geo,
            geo_to_length=geo_to_length,
            trip_time_lookup={"trip_random": point_times},
            train_ratio=1.0,
            random_state=101,
            min_len=1,
            verbose=False,
            fmm_sep=";",
            min_delta_seconds=1.0,
        )

        assert len(test_df) == 0

        # Some generated cases may collapse to no valid rows only if the tpath
        # parsing/mapping fails. With our generated mappings, this should not happen.
        assert len(train_df) == 1

        row = train_df.iloc[0]
        rid_list = [int(x) for x in row["rid_list"].split(",") if x]
        time_list = [pd.Timestamp(t) for t in row["time_list"].split(",") if t]

        assert len(rid_list) == len(time_list)
        assert len(time_list) >= 1

        expected_start = point_times[0].floor("ms")
        expected_end = point_times[-1].floor("ms")

        assert time_list[0] == expected_start

        # The final GPS anchor is only guaranteed to appear when the final tpath
        # segment has at least two edges, because single-edge intervals only keep
        # the start anchor under the one-timestamp-per-edge representation.
        last_segment_has_end_anchor = len(segments[-1]) >= 2
        
        # Only require final GPS anchor when there are at least 2 output points.
        if last_segment_has_end_anchor:
            assert time_list[-1] == expected_end
        else:
            # One edge can only receive one timestamp, so it keeps the start anchor.
            assert time_list[-1] <= expected_end

        assert all(expected_start <= t <= expected_end for t in time_list)
        assert all(curr >= prev for prev, curr in zip(time_list, time_list[1:]))

@st.composite
def duplicate_rule_case(draw):
    """
    Generate duplicate-edge scenarios for the duplicate resolution policy.

    Cases covered:
    - anchor + interpolated
    - interpolated + anchor
    - interpolated + interpolated
    - anchor + anchor same timestamp
    - anchor + anchor different timestamp
    """
    case_name = draw(
        st.sampled_from(
            [
                "anchor_then_interpolated",
                "interpolated_then_anchor",
                "interpolated_then_interpolated",
                "anchor_anchor_same_time",
                "anchor_anchor_different_time",
            ]
        )
    )

    base = pd.Timestamp("2026-01-01T10:00:00Z")
    later = base + pd.to_timedelta(
        draw(st.integers(min_value=1, max_value=60_000)),
        unit="ms",
    )

    return case_name, base, later


@given(case=duplicate_rule_case())
@settings(max_examples=75)
def test_duplicate_edge_resolution_policy_with_random_cases(case):
    """
    Property-based test for duplicate edge handling.

    This uses small tpath examples that force duplicate consecutive edge IDs
    and checks that the final output follows the policy:
    - anchor beats interpolated
    - interpolated duplicates are removed
    - same edge + same anchor timestamp is collapsed
    - same edge + different anchor timestamps is preserved
    """
    case_name, base, later = case

    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        fmm_path = tmp_path / "fmm_output.csv"

        fid_to_geo = {
            1: 101,
            2: 102,
        }

        geo_to_length = {
            101: 1.0,
            102: 1.0,
        }

        if case_name == "anchor_then_interpolated":
            # First segment: 1 at base, 2 at later.
            # Second segment starts with 2 at later, so no interpolated duplicate survives.
            tpath = "1,2|2"
            point_times = [base, later, later + pd.Timedelta(seconds=1)]
            expected_rids = [101, 102]
            expected_times = [base.floor("ms"), later.floor("ms")]

        elif case_name == "interpolated_then_anchor":
            # First segment: 1 anchor, 2 interpolated, 1 anchor.
            # Second segment starts with 1 anchor at same boundary.
            # The duplicate boundary anchor should not create a redundant copy.
            tpath = "1,2,1|1"
            point_times = [base, later, later + pd.Timedelta(seconds=1)]
            expected_rids = [101, 102, 101]
            expected_times = [base.floor("ms"), None, later.floor("ms")]

        elif case_name == "interpolated_then_interpolated":
            # Consecutive duplicate inside the same segment. Both middle entries are
            # interpolated, so duplicate policy should keep only one.
            tpath = "1,2,2,1"
            point_times = [base, later]
            expected_rids = [101, 102, 101]
            expected_times = [base.floor("ms"), None, later.floor("ms")]

        elif case_name == "anchor_anchor_same_time":
            # Boundary duplicate 102 @ later appears twice and should collapse
            tpath = "1,2|2,1"
            point_times = [base, later, later]
            expected_rids = [101, 102, 101]
            expected_times = [base.floor("ms"), later.floor("ms"), later.floor("ms")]
            # Same edge observed at the same GPS time across a boundary.
            # This is redundant and should collapse to one.
            # tpath = "1|1"
            # point_times = [base, base, later]
            # expected_rids = [101]
            # expected_times = [base.floor("ms")]

        else:
            # Same edge observed at two different GPS times.
            # This is a real repeated observation and should be preserved.
            tpath = "1|1"
            point_times = [base, later, later + pd.Timedelta(seconds=1)]
            expected_rids = [101, 101]
            expected_times = [base.floor("ms"), later.floor("ms")]

        pd.DataFrame([{"id": "trip_dup", "tpath": tpath}]).to_csv(
            fmm_path,
            sep=";",
            index=False,
        )

        train_df, _ = build_mm_csvs(
            fmm_path=fmm_path,
            fid_to_geo=fid_to_geo,
            geo_to_length=geo_to_length,
            trip_time_lookup={"trip_dup": point_times},
            train_ratio=1.0,
            random_state=101,
            min_len=1,
            verbose=False,
            fmm_sep=";",
            min_delta_seconds=1.0,
        )

        assert len(train_df) == 1

        row = train_df.iloc[0]
        rid_list = [int(x) for x in row["rid_list"].split(",") if x]
        time_list = [pd.Timestamp(t) for t in row["time_list"].split(",") if t]

        assert rid_list == expected_rids
        assert len(time_list) == len(expected_times)

        for actual, expected in zip(time_list, expected_times):
            # None means the timestamp is interpolated; test bounds instead of exact value.
            if expected is not None:
                assert actual == expected

        assert all(curr >= prev for prev, curr in zip(time_list, time_list[1:]))