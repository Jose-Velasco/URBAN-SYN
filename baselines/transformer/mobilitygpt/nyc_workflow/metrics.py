#!/usr/bin/env python3
"""CPU-only, single-file MobilityGPT adapter and notebook-compatible metrics.

No notebook execution, training imports, or historical evaluator imports occur.
See python -m nyc_workflow metrics --help and README.md. Input sequences are never shortened.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import csv
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
from scipy.stats import entropy, rankdata

EARTH_RADIUS_KM = 6371.0
GRID_CELL_DEG = 0.005
GRID_LON_MIN, GRID_LAT_MIN = -74.2963164, 40.4983251
GRID_WIDTH, GRID_HEIGHT = 87, 90
GRID_LON_MAX, GRID_LAT_MAX = -73.868091, 40.9410935  # plotting only
DISTANCE_BINS, DISTANCE_MAX_KM = 1000, 40.71
RADIUS_BINS, RADIUS_MAX_KM = 100, 10.6
EDR_EPS_M, FASTDTW_RADIUS = 100.0, 1
MAX_PAIRWISE_TRAJECTORIES, RANDOM_SEED = 5000, 42
PRIVACY_POINTS, PRIVACY_SAMPLE_SIZE, PRIVACY_BLOCK_SIZE = 32, 2000, 32
UNIQUENESS_THRESHOLD_M, OFF_NETWORK_THRESHOLD_M = 50.0, 30.0
OFF_NETWORK_CHUNK_SIZE = 100_000
METRICS = ("global_geographic_jsd", "od_jsd", "trip_distance_jsd", "radius_jsd",
           "hausdorff_km", "normalized_dtw_km", "edr", "jaccard", "mia_auc",
           "uniqueness", "off_network_rate", "rsc")
NOTEBOOK = "trajectory_evaluation_ts_lstm_all_metrics_parallel.ipynb"
NOTEBOOK_PROVENANCE = Path(__file__).with_name('notebook_provenance.json')
from .runtime import require_output


def require(condition, message):
    if not condition:
        raise ValueError(message)


def haversine_km_points(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    lat1, lon1 = np.radians(a[..., 0]), np.radians(a[..., 1])
    lat2, lon2 = np.radians(b[..., 0]), np.radians(b[..., 1])
    h = np.sin((lat2-lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def haversine_km_pair(a, b):
    return float(haversine_km_points(a, b))


def pairwise_haversine_matrix_km(a, b):
    return haversine_km_points(np.asarray(a)[:, None, :], np.asarray(b)[None, :, :])


def js_divergence(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    require(p.shape == q.shape, "JSD shape mismatch")
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")
    # Deliberately pass COUNTS to entropy, exactly as the notebook does.
    m = (p+q)/2
    return float(0.5*entropy(p, m) + 0.5*entropy(q, m))


def point_to_grid(lat, lon):
    x = math.floor((lon-GRID_LON_MIN)/GRID_CELL_DEG)
    y = math.floor((lat-GRID_LAT_MIN)/GRID_CELL_DEG)
    return x*GRID_HEIGHT+y if 0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT else None


def grid_distribution(lookup):
    counts = np.zeros(GRID_WIDTH*GRID_HEIGHT, dtype=np.int64)
    total = oob = 0
    for trajectory in lookup.values():
        for lat, lon in trajectory:
            total += 1
            cell = point_to_grid(float(lat), float(lon))
            if cell is None:
                oob += 1
            else:
                counts[cell] += 1
    return counts, dict(total_points=total, in_bounds_points=total-oob,
                        oob_points=oob, oob_rate=oob/total if total else float("nan"))


def od_distribution(lookup):
    counts, total, oob, empty = Counter(), 0, 0, 0
    for t in lookup.values():
        if not len(t):
            empty += 1
            continue
        total += 1
        o, d = point_to_grid(*t[0]), point_to_grid(*t[-1])
        if o is None or d is None:
            oob += 1
        else:
            counts[o, d] += 1
    return counts, dict(total_nonempty_trajectories=total, oob_od_trajectories=oob,
                        empty_trajectories=empty)


def sparse_counter_jsd(a, b):
    keys = sorted(set(a) | set(b), key=str)
    return js_divergence([a[k] for k in keys], [b[k] for k in keys])


def histogram_edges(max_value, core_bins):
    return np.r_[np.linspace(0, max_value, core_bins+1, dtype=float), np.inf]


DISTANCE_EDGES = histogram_edges(DISTANCE_MAX_KM, DISTANCE_BINS)
RADIUS_EDGES = histogram_edges(RADIUS_MAX_KM, RADIUS_BINS)


def histogram_jsd(real, generated, edges):
    return js_divergence(np.histogram(real, bins=edges)[0], np.histogram(generated, bins=edges)[0])


def trip_distance_km(t):
    return float(haversine_km_points(t[:-1], t[1:]).sum()) if len(t) >= 2 else 0.0


def ts_style_radius_km(t):
    if not len(t):
        return 0.0
    center = np.asarray([t[:, 0].mean(), t[:, 1].mean()])
    return float(haversine_km_points(t, np.repeat(center[None, :], len(t), axis=0)).mean())


def distribution_summary(lookup):
    grid, gm = grid_distribution(lookup)
    od, om = od_distribution(lookup)
    nonempty = [t for t in lookup.values() if len(t)]
    distances = np.asarray([trip_distance_km(t) for t in nonempty])
    radii = np.asarray([ts_style_radius_km(t) for t in nonempty])
    return dict(grid=grid, od=od, distances=distances, radii=radii), dict(
        grid=gm, od=om, trip_distance_overflow=int((distances >= DISTANCE_MAX_KM).sum()),
        radius_overflow=int((radii >= RADIUS_MAX_KM).sum()))


def distribution_metrics(real, generated):
    return dict(global_geographic_jsd=js_divergence(real['grid'], generated['grid']),
                od_jsd=sparse_counter_jsd(real['od'], generated['od']),
                trip_distance_jsd=histogram_jsd(real['distances'], generated['distances'], DISTANCE_EDGES),
                radius_jsd=histogram_jsd(real['radii'], generated['radii'], RADIUS_EDGES))


def hausdorff_km(real, generated):
    matrix = pairwise_haversine_matrix_km(real, generated)
    return float(max(matrix.min(axis=1).max(), matrix.min(axis=0).max()))


def normalized_dtw_km(real, generated):
    from fastdtw import fastdtw
    cost, path = fastdtw(np.asarray(real, dtype=float), np.asarray(generated, dtype=float),
                         radius=FASTDTW_RADIUS, dist=haversine_km_pair)
    return float(cost)/len(path) if path else float("nan")


def edr_distance(real, generated, eps_m=EDR_EPS_M):
    n, m = len(real), len(generated)
    if not n or not m:
        return float("nan")
    previous = np.arange(m+1, dtype=float)
    for i in range(1, n+1):
        current = np.empty(m+1, dtype=float)
        current[0] = i
        # Vectorizing only this independent row preserves the scalar recurrence.
        distance_m = haversine_km_points(real[i-1], generated)*1000
        for j in range(1, m+1):
            substitution = 0.0 if distance_m[j-1] < eps_m else 1.0
            current[j] = min(current[j-1]+1, previous[j]+1, previous[j-1]+substitution)
        previous = current
    return float(previous[m]/max(n, m))


def visited_grid_cells(t):
    return {cell for lat, lon in t if (cell := point_to_grid(float(lat), float(lon))) is not None}


def grid_jaccard(real, generated):
    a, b = visited_grid_cells(real), visited_grid_cells(generated)
    return len(a & b)/len(a | b) if a | b else float("nan")


def compute_single_pair_metrics(real, generated):
    if not len(real) or not len(generated):
        return None
    return (hausdorff_km(real, generated), normalized_dtw_km(real, generated),
            edr_distance(real, generated), grid_jaccard(real, generated))


def choose_pairs(pairs, max_pairs=MAX_PAIRWISE_TRAJECTORIES, seed=RANDOM_SEED):
    if max_pairs is None or len(pairs) <= max_pairs:
        return pairs
    return [pairs[int(i)] for i in np.random.default_rng(seed).choice(len(pairs), max_pairs, replace=False)]


@contextmanager
def worker_thread_limits():
    """Spawn one native numerical thread per process, restoring the caller's env."""
    names = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
    previous = {name: os.environ.get(name) for name in names}
    try:
        os.environ.update({name: '1' for name in names})
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def pairwise_metrics(test, generated, parents, jobs):
    # Sorted explicit section IDs replace notebook's sorted direct IDs. No OD fallback.
    pairs = choose_pairs([(parents[s], s) for s in sorted(generated)])
    workers = min(jobs, len(pairs)) if pairs else 1
    if workers > 1:
        # Bound pending work to 2*workers; retain input order for nanmean parity.
        with worker_thread_limits(), ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            pending, results, iterator = deque(), [], iter(pairs)
            for _ in range(min(len(pairs), 2*workers)):
                r, g = next(iterator)
                pending.append(pool.submit(compute_single_pair_metrics, test[r], generated[g]))
            while pending:
                results.append(pending.popleft().result())
                pair = next(iterator, None)
                if pair is not None:
                    r, g = pair
                    pending.append(pool.submit(compute_single_pair_metrics, test[r], generated[g]))
    else:
        results = [compute_single_pair_metrics(test[r], generated[g]) for r, g in pairs]
    valid = [r for r in results if r is not None]
    values = np.asarray(valid, dtype=float).reshape(-1, 4)
    metrics = {name: float(np.nanmean(values[:, i])) if len(values) and np.isfinite(values[:, i]).any()
               else float('nan') for i, name in enumerate(METRICS[4:8])}
    return metrics, dict(pairing_method='explicit_provenance_full_raw_parent',
                         selected_pairs=pairs, pair_count=len(valid), skipped_empty=len(results)-len(valid),
                         finite_counts={name: int(np.isfinite(values[:, i]).sum()) for i, name in enumerate(METRICS[4:8])},
                         workers=workers)


def resample_trajectory(traj, n_points=PRIVACY_POINTS):
    traj = np.asarray(traj, dtype=float)
    if not len(traj):
        return np.empty((0, 2), dtype=float)
    if len(traj) == 1:
        return np.repeat(traj[:1], n_points, axis=0)
    step = np.hypot(np.diff(traj[:, 0]), np.diff(traj[:, 1])*np.cos(np.radians(traj[0, 0])))
    cumulative = np.r_[0.0, np.cumsum(step)]
    if cumulative[-1] == 0:
        return np.repeat(traj[:1], n_points, axis=0)
    target = np.linspace(0, cumulative[-1], n_points)
    return np.c_[np.interp(target, cumulative, traj[:, 0]), np.interp(target, cumulative, traj[:, 1])]


def sample_lookup(lookup, sample_size=PRIVACY_SAMPLE_SIZE, seed=43):
    ids = np.asarray(list(lookup), dtype=object)
    if sample_size is not None and len(ids) > sample_size:
        ids = np.random.default_rng(seed).choice(ids, sample_size, replace=False)
    return {str(tid): np.asarray(lookup[str(tid)], dtype=float) for tid in ids}


def matched_point_trajectory_distance_matrix(queries, pool, block_size=PRIVACY_BLOCK_SIZE):
    queries, pool = np.asarray(queries, dtype=float), np.asarray(pool, dtype=float)
    require(queries.shape[1:] == pool.shape[1:], 'Privacy trajectory shapes differ')
    pool_rad = np.radians(pool)
    output = np.empty((len(queries), len(pool)), dtype=np.float32)
    for start in range(0, len(queries), block_size):
        q = np.radians(queries[start:start+block_size])
        q_lat, q_lon = q[:, None, :, 0], q[:, None, :, 1]
        p_lat, p_lon = pool_rad[None, :, :, 0], pool_rad[None, :, :, 1]
        h = np.sin((p_lat-q_lat)/2)**2 + np.cos(q_lat)*np.cos(p_lat)*np.sin((p_lon-q_lon)/2)**2
        distances = EARTH_RADIUS_KM*2*np.arcsin(np.sqrt(np.clip(h, 0, 1)))
        output[start:start+block_size] = distances.mean(axis=2).astype(np.float32)
    return output


def rank_auc(member_scores, nonmember_scores):
    if not len(member_scores) or not len(nonmember_scores):
        return float('nan')
    ranks = rankdata(np.r_[member_scores, nonmember_scores], method='average')
    p, n = len(member_scores), len(nonmember_scores)
    return float((ranks[:p].sum()-p*(p+1)/2)/(p*n))


def privacy_metrics(train, test, generated, parents):
    samples = [sample_lookup(pool, PRIVACY_SAMPLE_SIZE, seed) for pool, seed in
               zip((train, test, generated), (43, 44, 45))]
    ids = [list(s) for s in samples]
    diag = dict(train_ids=ids[0], test_ids=ids[1], generated_ids=ids[2], seeds=[43, 44, 45],
                resampled=True, points=32, block_size=32, distance_storage='float32', excluded_comparisons=0)
    require(not any(not len(t) for s in samples for t in s.values()),
            'Privacy metric inputs contain empty trajectories')
    if any(not s for s in samples):
        diag['unavailable_reason'] = 'Empty privacy pool'
        return dict(mia_auc=float('nan'), uniqueness=float('nan')), diag
    train_a, test_a, gen_a = [np.stack([resample_trajectory(t) for t in s.values()]) for s in samples]
    train_d = matched_point_trajectory_distance_matrix(train_a, gen_a)
    test_d = matched_point_trajectory_distance_matrix(test_a, gen_a)
    exclusions = []
    for row, tid in enumerate(ids[1]):
        cols = [i for i, gid in enumerate(ids[2]) if parents[gid] == tid]
        test_d[row, cols] = np.inf
        if cols:
            exclusions.append(dict(test_id=tid, generated_ids=[ids[2][c] for c in cols]))
    diag['excluded_comparisons'] = sum(len(x['generated_ids']) for x in exclusions)
    diag['exclusions'] = exclusions
    member, nonmember = -train_d.min(axis=1), -test_d.min(axis=1)
    unavailable = [tid for tid, score in zip(ids[1], nonmember) if not np.isfinite(score)]
    diag['unavailable_test_score_ids'] = unavailable
    diag['member_scores'] = member.tolist()
    diag['nonmember_scores'] = [float(s) if np.isfinite(s) else None for s in nonmember]
    if unavailable:
        diag['mia_auc_reason'] = 'Parent exclusions leave no candidate for one or more sampled test trips'
    # Keep float32 multiplication and strict > threshold, matching the notebook.
    uniqueness = float(np.mean(train_d.min(axis=0)*1000.0 > UNIQUENESS_THRESHOLD_M))
    return dict(mia_auc=float('nan') if unavailable else rank_auc(member, nonmember), uniqueness=uniqueness), diag


def compute_rsc_from_geo_rel(lookup, adjacency):
    valid = total = 0
    for seq in lookup.values():
        for a, b in zip(seq, seq[1:]):
            total += 1
            valid += a == b or (a, b) in adjacency
    return (valid/total if total else float('nan'), dict(valid_transitions=valid, total_transitions=total))


def load_road_spatial_index(path, layer='roads', mode_column=None):
    import geopandas as gpd
    from shapely import STRtree
    from pyproj import Transformer
    roads = gpd.read_file(path, **({'layer': layer} if layer else {}))
    require(roads.crs is not None, 'Canonical road network has no CRS')
    source_count = len(roads)
    roads = roads.loc[roads.geometry.notna() & ~roads.geometry.is_empty].copy()
    if mode_column:
        import pandas as pd
        require(mode_column in roads.columns, f'Missing network mode column {mode_column}')
        s = roads[mode_column]
        if pd.api.types.is_bool_dtype(s):
            mask = s.fillna(False)
        elif pd.api.types.is_numeric_dtype(s):
            mask = s.fillna(0).astype(float) != 0
        else:
            mask = s.astype(str).str.strip().str.lower().isin({'1', 'true', 't', 'yes', 'y'})
        roads = roads.loc[mask].copy()
    require(len(roads) > 0, 'Canonical network has no usable roads')
    require(roads.geom_type.isin(['LineString', 'MultiLineString']).all() and roads.is_valid.all(),
            'Canonical network has invalid or non-road geometry')
    metric_crs = roads.estimate_utm_crs()
    require(metric_crs is not None, 'Cannot estimate canonical network UTM CRS')
    projected = roads.to_crs(metric_crs)
    require(np.isfinite(projected.total_bounds).all(), 'Nonfinite projected network coordinates')
    return STRtree(projected.geometry.to_numpy()), Transformer.from_crs('EPSG:4326', metric_crs, always_xy=True), dict(
        source_road_count=source_count, road_count=len(projected), source_crs=str(roads.crs),
        metric_crs=str(metric_crs), mode_column=mode_column, geographic_bounds=roads.to_crs(4326).total_bounds.tolist())


def compute_off_network_rate(generated, tree, transformer):
    from shapely import points
    arrays = [t for t in generated.values() if len(t)]
    latlon = np.vstack(arrays) if arrays else np.empty((0, 2))
    distances = np.full(len(latlon), np.nan)
    for start in range(0, len(latlon), OFF_NETWORK_CHUNK_SIZE):
        chunk = latlon[start:start+OFF_NETWORK_CHUNK_SIZE]
        x, y = transformer.transform(chunk[:, 1], chunk[:, 0])
        require(np.isfinite(x).all() and np.isfinite(y).all(), 'Nonfinite projected representative points')
        indices, d = tree.query_nearest(points(x, y), return_distance=True, all_matches=False)
        distances[start+indices[0]] = d
    valid = np.isfinite(distances)
    return (float(np.mean(distances[valid] > OFF_NETWORK_THRESHOLD_M)) if valid.any() else float('nan'),
            dict(generated_point_count=len(distances), valid_nearest_distance_count=int(valid.sum()),
                 off_network_point_count=int((distances[valid] > OFF_NETWORK_THRESHOLD_M).sum()),
                 nearest_road_mean_m=float(distances[valid].mean()) if valid.any() else None,
                 nearest_road_p95_m=float(np.percentile(distances[valid], 95)) if valid.any() else None,
                 threshold_m=OFF_NETWORK_THRESHOLD_M))


def read_csv(path):
    with Path(path).open(newline='') as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames), f'Invalid CSV header: {path}')
        for row in reader:
            require(None not in row and all(v is not None for v in row.values()), f'Malformed CSV row: {path}')
            yield row


def read_json(path):
    def unique(pairs):
        result = {}
        for k, v in pairs:
            require(k not in result, f'Duplicate JSON key {k} in {path}')
            result[k] = v
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def jsonl(path):
    with Path(path).open() as f:
        for line in f:
            require(bool(line.strip()), f'Blank JSONL record in {path}')
            def unique(pairs):
                d = {}
                for k, v in pairs:
                    require(k not in d, f'Duplicate JSON key {k} in {path}')
                    d[k] = v
                return d
            yield json.loads(line, object_pairs_hook=unique)


def strict_int(value):
    require(not isinstance(value, (float, bool)), f'Non-integer road ID: {value!r}')
    return int(value)


def parse_int_sequence(value):
    if isinstance(value, list):
        return [strict_int(x) for x in value]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith('[') or text.startswith('('):
        return [strict_int(x) for x in ast.literal_eval(text)]
    return [strict_int(x.strip()) for x in text.split(',')]


def load_geo_id_to_latlon(path):
    mapping = {}
    for row in read_csv(path):
        rid = strict_int(row['geo_id'])
        require(rid not in mapping, f'Duplicate geometry road ID {rid}')
        coords = np.asarray(ast.literal_eval(row['coordinates']), dtype=float)
        require(coords.ndim == 2 and coords.shape[1] == 2 and len(coords) >= 2 and np.isfinite(coords).all(),
                f'Malformed/nonfinite coordinates for road {rid}')
        require((np.abs(coords[:, 0]) <= 180).all() and (np.abs(coords[:, 1]) <= 90).all(),
                f'Out-of-range coordinates for road {rid}')
        mapping[rid] = ((coords[0, 1]+coords[-1, 1])/2, (coords[0, 0]+coords[-1, 0])/2)
    require(bool(mapping), 'Empty raw geometry')
    return mapping


def load_road_lookup(path):
    result = {}
    for row in read_csv(path):
        tid = row['traj_id']
        require(tid and tid not in result, f'Duplicate/empty trajectory ID {tid!r} in {path}')
        result[tid] = parse_int_sequence(row['rid_list'])
    require(bool(result), f'Empty trajectory file {path}')
    return result


def road_lookup_to_gps(lookup, geo):
    result = {}
    for tid, seq in lookup.items():
        require(all(r in geo for r in seq), f'Unknown original road ID in trajectory {tid}')
        result[tid] = np.asarray([geo[r] for r in seq], dtype=float).reshape(-1, 2)
    return result


def load_rel_adjacency(path, geo):
    adjacency, keys = set(), set()
    for row in read_csv(path):
        key = row['rel_id']
        require(key not in keys, f'Duplicate relation record key {key}')
        keys.add(key)
        a, b = strict_int(row['origin_id']), strict_int(row['destination_id'])
        require(a in geo and b in geo, 'Unknown original road ID in adjacency')
        adjacency.add((a, b))
    return adjacency


def load_sections(path, provenance, reverse, raw_test, geo=None):
    rows = {}
    for row in read_csv(path):
        tid = row['traj_id']
        require(tid not in rows and tid, f'Duplicate prepared section {tid}')
        seq = parse_int_sequence(row['rid_list'])
        require(all(r in reverse for r in seq), f'Unknown mapped road ID in section {tid}')
        original = [reverse[r] for r in seq]
        require(original == parse_int_sequence(row['original_rid_list']), f'Inconsistent road mapping for {tid}')
        require(row['parent_traj_id'] in raw_test, f'Missing raw parent for {tid}')
        rows[tid] = dict(row=row, roads=seq, original=original, parent=row['parent_traj_id'])
    found, parents = set(), set()
    for trip in jsonl(provenance):
        parent = trip['parent_traj_id']
        require(trip['split'] == 'test' and parent not in parents, 'Duplicate/wrong-split provenance parent')
        require(parent in raw_test, f'Missing known raw parent {parent}')
        parents.add(parent)
        raw = raw_test[parent]
        require(trip['original_observations'] == len(raw), f'Raw parent observation count mismatch {parent}')
        cursor = 0
        for node in trip['nodes']:
            if geo is not None:
                require(node['road_id'] in geo, f'Unknown original road ID in provenance for {parent}')
            span = node['observation_range']
            if span is not None:
                a, b = span
                require(a == cursor and a < b <= len(raw) and all(r == node['road_id'] for r in raw[a:b]),
                        f'Raw parent/provenance observation mismatch {parent}')
                cursor = b
        require(cursor == len(raw), f'Incomplete parent provenance {parent}')
        node_cursor = 0
        section_indices = set()
        for section in trip['sections']:
            require(section['section_index'] not in section_indices, 'Duplicate provenance section index')
            section_indices.add(section['section_index'])
            a, b = section['node_range']
            require(a == node_cursor and a < b <= len(trip['nodes']), 'Invalid provenance section range')
            node_cursor = b
            sid = section['export_traj_id']
            if section['status'] != 'retained':
                require(sid is None, 'Discarded section has exported identity')
                continue
            require(sid in rows and sid not in found, f'Missing/duplicate exported section {sid}')
            found.add(sid)
            info, nodes = rows[sid], trip['nodes'][a:b]
            require(info['parent'] == parent and int(info['row']['section_index']) == section['section_index'],
                    f'Inconsistent section parent/index {sid}')
            require(info['original'] == [n['road_id'] for n in nodes], f'CSV/provenance roads mismatch {sid}')
            require(json.loads(info['row']['observation_ranges']) == [n['observation_range'] for n in nodes] and
                    info['row']['time_list'] == ','.join(n['timestamp'] for n in nodes), f'CSV/provenance observations mismatch {sid}')
        require(node_cursor == len(trip['nodes']), f'Incomplete section coverage {parent}')
    require(found == set(rows), 'Prepared sections lack explicit provenance')
    return rows


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return dict(sha256=h.hexdigest(), bytes=Path(path).stat().st_size)


def hashes(paths):
    return {str(p): fingerprint(p) for p in paths}


def dependencies(network):
    names = ['numpy', 'scipy', 'fastdtw'] + (['geopandas', 'shapely', 'pyproj', 'pandas', 'pyogrio'] if network else [])
    versions = {}
    for name in names:
        try:
            __import__(name)
            versions[name] = importlib.metadata.version(name)
        except ImportError as exc:
            raise ValueError(f'Missing dependency {name}; install with .venv/bin/python -m pip install {" ".join(names)}') from exc
    if network:
        from shapely import STRtree, points  # require Shapely 2 interface
        require(hasattr(STRtree, 'query_nearest'), 'Shapely >=2 required')
    return dict(python=platform.python_version(), **versions)


def load_inputs(args):
    run, raw = args.run_dir.resolve(), args.raw_data_dir.resolve()
    prepared = run/'prepared/NYCAll-Taxi'
    paths = dict(generated=run/'evaluation/generated_trajectories.jsonl', config=run/'pipeline_config.json',
                 mapping=prepared/'id_mapping.csv', test=prepared/'NYCAll_Taxi_trajectory_test.csv',
                 provenance=prepared/'test_provenance.jsonl', metadata=prepared/'processing_metadata.json',
                 prepared_geo=prepared/'roadmap.geo', geo=raw/'nyc.geo', rel=raw/'nyc.rel',
                 raw_train=raw/'nyc_mm_train.csv', raw_test=raw/'nyc_mm_test.csv')
    for p in paths.values():
        require(p.is_file(), f'Missing input file {p}')
    # The original notebook is archived. Its recorded identity suffices for
    # provenance; metric calculations have never executed notebook code.
    tracked = list(paths.values())+[Path(__file__).resolve(), NOTEBOOK_PROVENANCE]
    provenance = read_json(NOTEBOOK_PROVENANCE)
    if args.notebook is not None:
        notebook = args.notebook.resolve()
        require(notebook.is_file(), f'Missing optional notebook: {notebook}')
        require(fingerprint(notebook)['sha256'] == provenance['sha256'], 'Notebook provenance hash mismatch')
        tracked.append(notebook)
    if args.canonical_network and args.canonical_network.is_file():
        tracked.append(args.canonical_network.resolve())
    before = hashes(tracked)
    metadata, config = read_json(paths['metadata']), read_json(paths['config'])
    for name, key in [('geography', 'geo'), ('relations', 'rel'), ('train', 'raw_train'), ('test', 'raw_test')]:
        require(before[str(paths[key])]['sha256'] == metadata['source_files'][name]['sha256'],
                f'Raw input does not match prepared provenance hash: {name}')
    geo = load_geo_id_to_latlon(paths['geo'])
    reverse, forward = {}, {}
    for row in read_csv(paths['mapping']):
        a, b = strict_int(row['original_id']), strict_int(row['new_id'])
        require(a in geo and a not in forward and b not in reverse, 'Unknown/duplicate/inconsistent road mapping')
        forward[a], reverse[b] = b, a
    prepared_geo = load_geo_id_to_latlon(paths['prepared_geo'])
    require(set(prepared_geo) == set(reverse) and all(prepared_geo[b] == geo[a] for b, a in reverse.items()),
            'Prepared geometry and original ID mapping disagree')
    train_roads, test_roads = load_road_lookup(paths['raw_train']), load_road_lookup(paths['raw_test'])
    train, test = road_lookup_to_gps(train_roads, geo), road_lookup_to_gps(test_roads, geo)
    sections = load_sections(paths['test'], paths['provenance'], reverse, test_roads, geo)
    section_positions = {sid: i for i, sid in enumerate(sections)}
    adjacency = load_rel_adjacency(paths['rel'], geo)
    groups, seen = {}, set()
    for record in jsonl(paths['generated']):
        cp, p, seed, sid = record['checkpoint'], strict_int(record['prompt_length']), strict_int(record['seed']), record['trajectory_id']
        key = (cp, p, seed, sid)
        require(key not in seen, f'Duplicate generated record key {key}')
        seen.add(key)
        require(sid in sections, f'Generated section lacks explicit provenance {sid}')
        info = sections[sid]
        require(record['parent_id'] == info['parent'] and info['parent'] in test, f'Missing/inconsistent known parent {sid}')
        seq, prompt = parse_int_sequence(record['roads']), parse_int_sequence(record['prompt_roads'])
        require(len(info['roads']) > p > 0 and prompt == info['roads'][:p] and seq[:p] == prompt,
                f'Invalid supplied prompt for {key}')
        require(all(r in reverse for r in seq), f'Unknown generated road ID {key}')
        require(record['trajectory_index'] == section_positions[sid], f'Inconsistent trajectory index {key}')
        require(record['road_count'] == len(seq) and record['immediate_stop'] == (record['termination'] == 'eos' and len(seq) == p),
                f'Inconsistent generated length/termination metadata {key}')
        group = groups.setdefault((cp, p, seed), dict(roads={}, parents={}, records=[]))
        group['roads'][sid] = [reverse[r] for r in seq]  # reverse exactly once
        group['parents'][sid] = info['parent']
        group['records'].append(record)
    prompts, seeds = config['evaluation']['prompt_roads'], config['evaluation']['seeds']
    checkpoints = ['pretrained', f"selected_dpo_{config['dpo']['selected_update']}"]
    expected = {(cp, p, s) for cp in checkpoints for p in prompts for s in seeds}
    require(len(expected) == 12 and set(groups) == expected, 'Expected two checkpoints × two prompts × three seeds (12 groups)')
    for (cp, p, seed), group in groups.items():
        eligible = [sid for sid, info in sections.items() if len(info['roads']) > p]
        require(set(group['roads']) == set(eligible), f'Incomplete generated group {(cp,p,seed)}')
        # Common prepared source order guarantees identical sampled identities across checkpoints.
        group['roads'] = {sid: group['roads'][sid] for sid in eligible}
        group['gps'] = road_lookup_to_gps(group['roads'], geo)
    train_routes = set(map(tuple, train_roads.values()))
    test_routes = set(map(tuple, test_roads.values()))
    for group in groups.values():
        routes = list(map(tuple, group['roads'].values()))
        group['overlap'] = dict(samples_equal_to_full_raw_training_route=sum(t in train_routes for t in routes),
                                samples_equal_to_full_raw_test_route=sum(t in test_routes for t in routes),
                                duplicate_generated_route_samples=len(routes)-len(set(routes)))
    diagnostics = dict(raw_training_trips=len(train), raw_test_trips=len(test), prepared_sections=len(sections),
                       raw_training_points=sum(map(len, train.values())), raw_test_points=sum(map(len, test.values())),
                       consecutive_raw_training_repetitions=sum(sum(a == b for a,b in zip(t,t[1:])) for t in train_roads.values()),
                       consecutive_raw_test_repetitions=sum(sum(a == b for a,b in zip(t,t[1:])) for t in test_roads.values()),
                       raw_test_routes_equal_to_training=sum(tuple(t) in train_routes for t in test_roads.values()),
                       raw_train_test_parent_id_overlap=len(set(train) & set(test)),
                       raw_parent_ids_with_prepared_sections=len({s['parent'] for s in sections.values()}),
                       mapping_roads=len(reverse), original_roads=len(geo), records=len(seen))
    return train, test, groups, adjacency, diagnostics, before, config


def benchmark_pairs(test, groups):
    """Time three full pairs spanning the point-product range in one group."""
    key = min(groups, key=lambda k: (k[0], -k[1], k[2]))
    group = groups[key]
    order = sorted(group['gps'], key=lambda sid: (len(test[group['parents'][sid]])*len(group['gps'][sid]), sid))
    selected = list(dict.fromkeys([order[0], order[len(order)//2], order[-1]]))
    rows = []
    for sid in selected:
        raw, gen = test[group['parents'][sid]], group['gps'][sid]
        started = time.perf_counter()
        values = compute_single_pair_metrics(raw, gen)
        rows.append(dict(section_id=sid, parent_id=group['parents'][sid], raw_parent_points=len(raw),
                         generated_points=len(gen), seconds=time.perf_counter()-started,
                         metrics=dict(zip(METRICS[4:8], values)) if values else None))
        print(f'Benchmark full pair {sid}: {len(raw)} × {len(gen)} points, {rows[-1]["seconds"]:.3f}s', flush=True)
    return dict(checkpoint=key[0], prompt_length=key[1], seed=key[2], pairs=rows,
                selection='Minimum, median, maximum raw-length × generated-length in this group; ties by section ID',
                limitation='Small deterministic fixture timings are not an unbiased estimate of full-run cost')


def strict_value(value):
    if isinstance(value, dict):
        return {str(k): strict_value(v) for k,v in value.items()}
    if isinstance(value, (tuple, list)):
        return [strict_value(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(strict_value(value), indent=2, allow_nan=False)+'\n')


def table(rows):
    columns = ['checkpoint', 'prompt_length', 'seed', *METRICS]
    lines = [' | '.join(columns), ' | '.join(['---']*len(columns))]
    for row in rows:
        lines.append(' | '.join('unavailable' if row.get(k) is None else f'{row[k]:.6g}'
                                if isinstance(row[k], float) else str(row[k]) for k in columns))
    return '\n'.join(lines)


DEFINITIONS = '''| Metric | Definition and units | Direction |
| --- | --- | --- |
| Global geographic JSD | In-bounds point-visit counts on fixed grid; natural-log entropy divergence | Lower |
| OD JSD | Joint first/last grid-cell counts; natural-log entropy divergence | Lower |
| Trip distance JSD | Sum consecutive Haversine km; 1,000 bins to 40.71 km plus overflow | Lower |
| Radius JSD | Mean Haversine km to arithmetic coordinate mean; 100 bins to 10.6 km plus overflow | Lower |
| Hausdorff | Symmetric maximum nearest-point Haversine distance, km | Lower |
| Normalized DTW | FastDTW radius 1 Haversine cost / path length, km | Lower |
| EDR | Edit distance, strict <100 m match / maximum sequence length | Lower |
| Grid Jaccard | Unique in-bounds cell intersection / union; empty union undefined | Higher |
| MIA AUC | Average-rank AUC of negative nearest-generated mean matched-point distances | Near 0.5 suggests weak attack; not a guarantee |
| Uniqueness | Fraction >50 m from nearest sampled training trajectory | Higher novelty under this distance only |
| Off-network rate | Representative points strictly >30 m from projected canonical roads | Lower |
| RSC | Valid original directed .rel transitions, including identical consecutive IDs | Higher |

Earth radius is 6371 km. Grid origin is (lon -74.2963164, lat 40.4983251),
cell size 0.005 degrees, width 87, height 90, index x*90+y. Floor-and-index
bounds exclude and report out-of-grid points; plotting maxima (-73.868091,
40.9410935) do not bound this computation. Entropy receives raw count vectors
and their arithmetic mixture, exactly as in the notebook (unequal totals matter).
Pairwise metrics use up to 5,000 sorted explicit pairs, seed 42, and per-metric
nanmean. Privacy independently samples up to 2,000 train/test/generated trips
with seeds 43/44/45 in stable source order, always interpolates 32 points, uses
blocks of 32 queries, and stores distances in float32.

Raw observation repetitions and full trips are retained. Generated road order,
supplied prompts, repeated roads, immediate stops, and capped routes are retained.
Generated IDs are reversed once; GPS is the average of geometry endpoints in
[latitude, longitude] order. All original test trips supply distributions and
privacy nonmembers; all original training trips supply the member pool.
Pairwise references are full raw parents, while generated samples are sections.
Several sections can share a parent and are dependent samples; their identities
and parent counts are recorded. No seed pooling or composite ranking is used.

Generation is conditional on supplied reference prefixes, and the test split
was previously examined. Training/test route overlap is retained, including
identical routes with different parent identities. The nearest-neighbor privacy
attack measures geometric proximity under interpolation, not individual privacy
or a membership guarantee. Every sampled generated section of the queried test
parent is excluded, even when several sections share it. Identical routes from
other parents remain candidates. If no candidate remains, the score and AUC
are unavailable; no test query is silently dropped.

Canonical road coverage, road-mode filtering, representative-point approximation,
and missing or distant network geometry limit off-network interpretation. A
missing canonical network explicitly makes this evaluation incomplete. Prepared
geometry never substitutes for it. These JSD and distance values are not
interchangeable with historical MobilityGPT reports, which use other reference
populations, point weighting, pairing, and definitions.
'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--raw-data-dir', type=Path, required=True)
    parser.add_argument('--canonical-network', type=Path)
    parser.add_argument('--network-layer', default='roads')
    parser.add_argument('--network-mode-column')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 1)//2))
    parser.add_argument('--allow-missing-network', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--validate-only', action='store_true')
    mode.add_argument('--benchmark-only', action='store_true', help='Validate, time three full raw-parent pairs, and stop without primary scores')
    parser.add_argument('--notebook', type=Path, help='Optional original notebook hash verification; never executed or required')
    args = parser.parse_args(argv)
    started = time.perf_counter()
    run, raw = args.run_dir.resolve(), args.raw_data_dir.resolve()
    protected = [raw] + ([args.canonical_network] if args.canonical_network is not None else [])
    out = require_output(args.output_dir, protected)
    # Only this fresh, explicitly run-owned metrics directory may be nested in
    # an input run. All existing canonical results and other run inputs stay immutable.
    if out != run/'canonical_metrics':
        require_output(out, [run])
    require(not out.exists(), f'Refusing existing output directory: {out}')
    require(args.jobs > 0, '--jobs must be positive')
    args.jobs = min(args.jobs, os.cpu_count() or 1)
    network_present = args.canonical_network is not None
    if network_present:
        require(args.canonical_network.is_file(), f'Supplied canonical network is not a file: {args.canonical_network}')
    require(network_present or args.allow_missing_network, 'Canonical network required; supply --canonical-network or explicitly --allow-missing-network')
    versions = dependencies(network_present)
    print('Validating and hashing all inputs...', flush=True)
    train, test, groups, adjacency, coverage, before, config = load_inputs(args)
    tree = transformer = None
    network = dict(unavailable_reason='Canonical network not supplied or file missing')
    if network_present:
        tree, transformer, network = load_road_spatial_index(args.canonical_network, args.network_layer, args.network_mode_column)
    validation_seconds = time.perf_counter()-started
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(status='validated' if args.validate_only else 'running', complete=False,
                    notebook_provenance=read_json(NOTEBOOK_PROVENANCE),
                    configuration={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                    source_configuration=config, dependencies=versions, input_hashes_before=before,
                    coverage=coverage, network=network, timings=dict(validation_seconds=validation_seconds))
    write_json(out/'manifest.json', manifest)
    if args.validate_only:
        after = hashes([Path(p) for p in before])
        require(before == after, 'Input hashes changed during validation')
        manifest.update(input_hashes_after=after, input_hashes_preserved=True)
        write_json(out/'manifest.json', manifest)
        print(f'Validated {len(groups)} groups. No metrics calculated. Manifest: {out}/manifest.json')
        return 0
    manifest['benchmark'] = benchmark_pairs(test, groups)
    write_json(out/'manifest.json', manifest)
    if args.benchmark_only:
        after = hashes([Path(p) for p in before])
        require(before == after, 'Input hashes changed during benchmark')
        manifest.update(status='benchmark_only', input_hashes_after=after, input_hashes_preserved=True)
        manifest['timings']['total_seconds'] = time.perf_counter()-started
        write_json(out/'manifest.json', manifest)
        print(f'Benchmark complete; no primary scores calculated. Manifest: {out}/manifest.json')
        return 0
    print('Computing full raw-test distribution statistics...', flush=True)
    t = time.perf_counter()
    real, real_diag = distribution_summary(test)
    manifest['timings']['raw_test_distribution_seconds'] = time.perf_counter()-t
    rows, diagnostics = [], []
    for key, group in sorted(groups.items()):
        cp, prompt, seed = key
        print(f'Evaluating {cp}, prompt={prompt}, seed={seed}, samples={len(group["gps"])}', flush=True)
        row, diag, timings = dict(checkpoint=cp, prompt_length=prompt, seed=seed), {}, {}
        t = time.perf_counter()
        gen_summary, gen_diag = distribution_summary(group['gps'])
        row.update(distribution_metrics(real, gen_summary))
        diag['distribution'] = dict(real=real_diag, generated=gen_diag)
        timings['distribution_seconds'] = time.perf_counter()-t
        t = time.perf_counter()
        metrics, diag['pairwise'] = pairwise_metrics(test, group['gps'], group['parents'], args.jobs)
        row.update(metrics)
        timings['pairwise_seconds'] = time.perf_counter()-t
        t = time.perf_counter()
        metrics, diag['privacy'] = privacy_metrics(train, test, group['gps'], group['parents'])
        row.update(metrics)
        timings['privacy_seconds'] = time.perf_counter()-t
        t = time.perf_counter()
        if tree is None:
            row['off_network_rate'], diag['off_network'] = float('nan'), network
        else:
            row['off_network_rate'], diag['off_network'] = compute_off_network_rate(group['gps'], tree, transformer)
        row['rsc'], diag['rsc'] = compute_rsc_from_geo_rel(group['roads'], adjacency)
        timings['network_and_rsc_seconds'] = time.perf_counter()-t
        counts = Counter(group['parents'].values())
        diag['pairing_and_generation'] = dict(generated_samples=len(group['gps']), distinct_parents=len(counts),
            parents_with_multiple_sections={p:n for p,n in counts.items() if n > 1},
            supplied_prompt_points=sum(len(r['prompt_roads']) for r in group['records']),
            immediate_stops=sum(r['immediate_stop'] for r in group['records']),
            terminations=dict(Counter(r['termination'] for r in group['records'])),
            repeated_occurrences=sum(len(seq)-len(set(seq)) for seq in group['roads'].values()))
        diag['route_overlap'] = group['overlap']
        reasons = {}
        for name in METRICS:
            if not math.isfinite(row[name]):
                reasons[name] = (network['unavailable_reason'] if name == 'off_network_rate' and tree is None else
                    diag['privacy'].get('mia_auc_reason', diag['privacy'].get('unavailable_reason', 'Undefined for empty pool')) if name in ('mia_auc', 'uniqueness') else
                    'No transitions' if name == 'rsc' else 'No eligible nonempty observations, pairs, or in-bounds support')
        row = strict_value(row)
        row['unavailable_reasons'] = reasons
        rows.append(row)
        diagnostics.append(dict(checkpoint=cp, prompt_length=prompt, seed=seed, diagnostics=diag, timings=timings))
        write_json(out/'progress.json', dict(completed_groups=len(rows), rows=rows, diagnostics=diagnostics))
    after = hashes([Path(p) for p in before])
    require(before == after, 'Input hashes changed during evaluation')
    manifest.update(status='complete' if tree is not None else 'incomplete',
                    complete=tree is not None, input_hashes_after=after, input_hashes_preserved=True)
    manifest['timings']['total_seconds'] = time.perf_counter()-started
    write_json(out/'manifest.json', manifest)
    write_json(out/'results.json', dict(**manifest, results=rows, diagnostics=diagnostics))
    write_json(out/'sample_identities.json', [dict(checkpoint=d['checkpoint'], prompt_length=d['prompt_length'], seed=d['seed'],
        pairs=d['diagnostics']['pairwise']['selected_pairs'],
        privacy={k:v for k,v in d['diagnostics']['privacy'].items() if k.endswith('_ids') or k == 'exclusions'}) for d in diagnostics])
    with (out/'metrics.csv').open('w', newline='') as f:
        fields = ['checkpoint', 'prompt_length', 'seed', *METRICS, *[m+'_reason' for m in METRICS]]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k:v for k,v in row.items() if k != 'unavailable_reasons'},
                             **{m+'_reason': row['unavailable_reasons'].get(m, '') for m in METRICS}})
    report = f'# MobilityGPT notebook-compatible evaluation\n\nStatus: {manifest["status"]}.\n\n'+table(rows)+'\n\n'+DEFINITIONS
    report += '\nCoverage and dependence diagnostics:\n\n```json\n'+json.dumps(coverage, indent=2)+'\n```\n'
    report += '\nFull per-group exclusions, denominator counts, timings, network coverage, hashes, and dependency versions are in results.json; sampled identities are in sample_identities.json.\n'
    (out/'report.md').write_text(report)
    print(table(rows))
    print(f'Outputs: {out}')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, ImportError) as exc:
        print(f'Validation/evaluation failed: {exc}', file=sys.stderr)
        raise SystemExit(2)
