"""Frozen pure notebook definitions for independent parity fixtures only.

Extracted from cells 4,7,11,13,16,18; loading, plotting and evaluation cells
are excluded. Original notebook SHA256: 791977d10ec570810f8cdc832fa5fba18e9cf85fca7b3f1ca6c047dfab8c6209
The complete notebook is preserved in the precleanup snapshot archive.
This module is never imported by either evaluator.
"""
from collections import Counter, defaultdict
import math
import os
from pathlib import Path
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.stats import entropy
from pyproj import Transformer
from shapely import STRtree, points as shapely_points

def tqdm(values, **kwargs):
    return values

EARTH_RADIUS_KM = 6371.0
GRID_CELL_DEG = 0.005
GRID_LON_MIN = -74.2963164
GRID_LON_MAX = -73.868091
GRID_LAT_MIN = 40.4983251
GRID_LAT_MAX = 40.9410935
GRID_WIDTH = 87
GRID_HEIGHT = 90
DISTANCE_BINS = 1000
DISTANCE_MAX_KM = 40.71
RADIUS_BINS = 100
RADIUS_MAX_KM = 10.6
EDR_EPS_M = 100.0
FASTDTW_RADIUS = 1
MAX_PAIRWISE_TRAJECTORIES = 5000
PRIVACY_POINTS = 32
PRIVACY_SAMPLE_SIZE = 2000
PRIVACY_BLOCK_SIZE = 32
UNIQUENESS_THRESHOLD_M = 50.0
PRIVACY_RESAMPLE_MODE = 'always'
OFF_NETWORK_THRESHOLD_M = 30.0
OFF_NETWORK_CHUNK_SIZE = 100000
RANDOM_SEED = 42

def haversine_km_points(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise Haversine distance between [..., 2] [lat, lon] arrays."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    lat1 = np.radians(a[..., 0])
    lon1 = np.radians(a[..., 1])
    lat2 = np.radians(b[..., 0])
    lon2 = np.radians(b[..., 1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    h = np.clip(h, 0.0, 1.0)
    return EARTH_RADIUS_KM * 2.0 * np.arcsin(np.sqrt(h))

def haversine_km_pair(a: np.ndarray, b: np.ndarray) -> float:
    """Haversine distance in km between two [lat, lon] points."""
    return float(haversine_km_points(np.asarray(a), np.asarray(b)))

def pairwise_haversine_matrix_km(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Haversine matrix between trajectory points."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    lat1 = np.radians(a[:, 0])[:, None]
    lon1 = np.radians(a[:, 1])[:, None]
    lat2 = np.radians(b[:, 0])[None, :]
    lon2 = np.radians(b[:, 1])[None, :]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    h = np.clip(h, 0.0, 1.0)
    return EARTH_RADIUS_KM * 2.0 * np.arcsin(np.sqrt(h))

def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence using scipy.stats.entropy and natural logs."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.shape != q.shape:
        raise ValueError(f'JSD shape mismatch: {p.shape} vs {q.shape}')
    if p.sum() <= 0 or q.sum() <= 0:
        return float('nan')
    m = (p + q) / 2.0
    return float(0.5 * entropy(p, m) + 0.5 * entropy(q, m))

def point_to_grid(lat: float, lon: float) -> int | None:
    """Convert [lat, lon] to flattened fixed-grid index using floor()."""
    x = math.floor((lon - GRID_LON_MIN) / GRID_CELL_DEG)
    y = math.floor((lat - GRID_LAT_MIN) / GRID_CELL_DEG)
    if not (0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT):
        return None
    return x * GRID_HEIGHT + y

def grid_distribution(lookup: dict[str, np.ndarray]) -> tuple[np.ndarray, dict]:
    """Count all in-bounds GPS point visits over the shared 87 x 90 grid."""
    counts = np.zeros(GRID_WIDTH * GRID_HEIGHT, dtype=np.int64)
    total = 0
    oob = 0
    for traj in lookup.values():
        traj = np.asarray(traj, dtype=float)
        for lat, lon in traj:
            total += 1
            cell = point_to_grid(float(lat), float(lon))
            if cell is None:
                oob += 1
            else:
                counts[cell] += 1
    metadata = {'total_points': int(total), 'in_bounds_points': int(total - oob), 'oob_points': int(oob), 'oob_rate': float(oob / total) if total else float('nan')}
    return (counts, metadata)

def od_distribution(lookup: dict[str, np.ndarray]) -> tuple[Counter, dict]:
    """Joint origin-destination grid-cell distribution."""
    counter = Counter()
    total = 0
    skipped_oob = 0
    skipped_empty = 0
    for traj in lookup.values():
        traj = np.asarray(traj, dtype=float)
        if len(traj) == 0:
            skipped_empty += 1
            continue
        total += 1
        o = point_to_grid(float(traj[0, 0]), float(traj[0, 1]))
        d = point_to_grid(float(traj[-1, 0]), float(traj[-1, 1]))
        if o is None or d is None:
            skipped_oob += 1
            continue
        counter[o, d] += 1
    return (counter, {'total_nonempty_trajectories': int(total), 'oob_od_trajectories': int(skipped_oob), 'empty_trajectories': int(skipped_empty)})

def sparse_counter_jsd(a: Counter, b: Counter) -> float:
    """JSD over the union of sparse Counter keys."""
    keys = sorted(set(a) | set(b), key=str)
    if not keys:
        return float('nan')
    pa = np.asarray([a.get(k, 0) for k in keys], dtype=float)
    pb = np.asarray([b.get(k, 0) for k in keys], dtype=float)
    return js_divergence(pa, pb)

def histogram_edges(max_value: float, core_bins: int) -> np.ndarray:
    """Uniform core bins plus one overflow bin [max_value, +inf)."""
    core = np.linspace(0.0, max_value, core_bins + 1, dtype=float)
    return np.r_[core, np.inf]

def histogram_jsd(real_values, generated_values, edges) -> float:
    real_hist, _ = np.histogram(np.asarray(real_values, dtype=float), bins=edges)
    gen_hist, _ = np.histogram(np.asarray(generated_values, dtype=float), bins=edges)
    return js_divergence(real_hist, gen_hist)

def trip_distance_km(traj: np.ndarray) -> float:
    """Sum Haversine distances between consecutive trajectory points."""
    traj = np.asarray(traj, dtype=float)
    if len(traj) < 2:
        return 0.0
    return float(haversine_km_points(traj[:-1], traj[1:]).sum())

def ts_style_radius_km(traj: np.ndarray) -> float:
    """Mean Haversine distance to arithmetic mean [lat, lon] center."""
    traj = np.asarray(traj, dtype=float)
    if len(traj) == 0:
        return 0.0
    center = np.asarray([traj[:, 0].mean(), traj[:, 1].mean()], dtype=float)
    center_repeated = np.repeat(center[None, :], len(traj), axis=0)
    return float(haversine_km_points(traj, center_repeated).mean())

def trajectory_distribution_values(lookup: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    distances = []
    radii = []
    for traj in tqdm(lookup.values(), total=len(lookup), desc='Trajectory distance/radius statistics'):
        traj = np.asarray(traj, dtype=float)
        if len(traj) == 0:
            continue
        distances.append(trip_distance_km(traj))
        radii.append(ts_style_radius_km(traj))
    return (np.asarray(distances), np.asarray(radii))

def compute_distribution_metrics(real_lookup: dict[str, np.ndarray], generated_lookup: dict[str, np.ndarray]) -> tuple[dict, dict]:
    real_grid, real_grid_meta = grid_distribution(real_lookup)
    gen_grid, gen_grid_meta = grid_distribution(generated_lookup)
    real_od, real_od_meta = od_distribution(real_lookup)
    gen_od, gen_od_meta = od_distribution(generated_lookup)
    real_distance, real_radius = trajectory_distribution_values(real_lookup)
    gen_distance, gen_radius = trajectory_distribution_values(generated_lookup)
    metrics = {'global_geographic_jsd': js_divergence(real_grid, gen_grid), 'od_jsd': sparse_counter_jsd(real_od, gen_od), 'trip_distance_jsd': histogram_jsd(real_distance, gen_distance, DISTANCE_EDGES), 'radius_jsd': histogram_jsd(real_radius, gen_radius, RADIUS_EDGES)}
    diagnostics = {'real_grid': real_grid_meta, 'generated_grid': gen_grid_meta, 'real_od': real_od_meta, 'generated_od': gen_od_meta, 'real_trip_distance_max_km': float(real_distance.max()) if len(real_distance) else float('nan'), 'generated_trip_distance_max_km': float(gen_distance.max()) if len(gen_distance) else float('nan'), 'real_radius_max_km': float(real_radius.max()) if len(real_radius) else float('nan'), 'generated_radius_max_km': float(gen_radius.max()) if len(gen_radius) else float('nan')}
    return (metrics, diagnostics)

def trajectory_od_key(traj: np.ndarray):
    traj = np.asarray(traj, dtype=float)
    if len(traj) == 0:
        return None
    o = point_to_grid(float(traj[0, 0]), float(traj[0, 1]))
    d = point_to_grid(float(traj[-1, 0]), float(traj[-1, 1]))
    if o is None or d is None:
        return None
    return (o, d)

def build_trajectory_pairs(real_lookup: dict[str, np.ndarray], generated_lookup: dict[str, np.ndarray], mode: str='direct_then_od') -> tuple[list[tuple[str, str]], str]:
    if mode not in {'direct', 'direct_then_od', 'od'}:
        raise ValueError(f'Unsupported pairing mode: {mode}')
    if mode in {'direct', 'direct_then_od'}:
        shared = sorted(set(real_lookup) & set(generated_lookup))
        if shared:
            return ([(tid, tid) for tid in shared], 'direct_id')
        if mode == 'direct':
            return ([], 'direct_id')
    real_groups = defaultdict(list)
    gen_groups = defaultdict(list)
    for tid, traj in real_lookup.items():
        key = trajectory_od_key(traj)
        if key is not None:
            real_groups[key].append(tid)
    for tid, traj in generated_lookup.items():
        key = trajectory_od_key(traj)
        if key is not None:
            gen_groups[key].append(tid)
    pairs = []
    for key in sorted(set(real_groups) & set(gen_groups)):
        for real_id, gen_id in zip(sorted(real_groups[key]), sorted(gen_groups[key])):
            pairs.append((real_id, gen_id))
    return (pairs, 'od_grid')

def choose_pairs(pairs: list[tuple[str, str]], max_pairs: int | None, seed: int) -> list[tuple[str, str]]:
    if max_pairs is None or len(pairs) <= max_pairs:
        return pairs
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(pairs), size=max_pairs, replace=False)
    return [pairs[int(i)] for i in indices]

def hausdorff_km(real: np.ndarray, generated: np.ndarray) -> float:
    """Symmetric Hausdorff distance in km."""
    matrix = pairwise_haversine_matrix_km(real, generated)
    return float(max(matrix.min(axis=1).max(), matrix.min(axis=0).max()))

def normalized_dtw_km(real: np.ndarray, generated: np.ndarray) -> float:
    """FastDTW Haversine cost divided by warping-path length."""
    raw_distance, path = fastdtw(np.asarray(real, dtype=float), np.asarray(generated, dtype=float), radius=FASTDTW_RADIUS, dist=haversine_km_pair)
    if not path:
        return float('nan')
    return float(raw_distance) / len(path)

def edr_distance(real: np.ndarray, generated: np.ndarray, eps_m: float) -> float:
    """Spatial EDR normalized by max trajectory length."""
    real = np.asarray(real, dtype=float)
    generated = np.asarray(generated, dtype=float)
    n = len(real)
    m = len(generated)
    if n == 0 or m == 0:
        return float('nan')
    previous = np.arange(m + 1, dtype=float)
    for i in range(1, n + 1):
        current = np.empty(m + 1, dtype=float)
        current[0] = i
        for j in range(1, m + 1):
            distance_m = haversine_km_pair(real[i - 1], generated[j - 1]) * 1000.0
            substitution = 0.0 if distance_m < eps_m else 1.0
            current[j] = min(current[j - 1] + 1.0, previous[j] + 1.0, previous[j - 1] + substitution)
        previous = current
    return float(previous[m] / max(n, m))

def visited_grid_cells(traj: np.ndarray) -> set[int]:
    cells = set()
    for lat, lon in np.asarray(traj, dtype=float):
        cell = point_to_grid(float(lat), float(lon))
        if cell is not None:
            cells.add(cell)
    return cells

def grid_jaccard(real: np.ndarray, generated: np.ndarray) -> float:
    """IoU of unique visited grid-cell sets."""
    a = visited_grid_cells(real)
    b = visited_grid_cells(generated)
    union = a | b
    if not union:
        return float('nan')
    return len(a & b) / len(union)

def get_pairwise_worker_count() -> int:
    """
    Resolve process count for pairwise trajectory metrics.

    Auto mode approximates physical cores as half the logical CPU count.
    This is a good default for SMT/Hyper-Threaded CPUs such as the 5900X.
    Set PAIRWISE_N_JOBS explicitly to override it.
    """
    logical = os.cpu_count() or 1
    if PAIRWISE_N_JOBS is not None:
        return max(1, min(int(PAIRWISE_N_JOBS), logical))
    return max(1, logical // 2)

def compute_single_pair_metrics(real: np.ndarray, generated: np.ndarray) -> tuple[float, float, float, float] | None:
    """
    Compute all expensive pairwise metrics for one trajectory pair.

    Keeping Hausdorff, DTW, EDR, and Jaccard in one worker avoids sending the
    same trajectories to separate process pools four different times.
    """
    real = np.asarray(real, dtype=float)
    generated = np.asarray(generated, dtype=float)
    if len(real) == 0 or len(generated) == 0:
        return None
    return (hausdorff_km(real, generated), normalized_dtw_km(real, generated), edr_distance(real, generated, EDR_EPS_M), grid_jaccard(real, generated))

def compute_pairwise_metrics(real_lookup: dict[str, np.ndarray], generated_lookup: dict[str, np.ndarray]) -> tuple[dict, dict]:
    pairs, pairing_method = build_trajectory_pairs(real_lookup, generated_lookup, mode=PAIRING_MODE)
    pairs = choose_pairs(pairs, max_pairs=MAX_PAIRWISE_TRAJECTORIES, seed=RANDOM_SEED)
    if not pairs:
        return ({'hausdorff_km': float('nan'), 'normalized_dtw_km': float('nan'), 'edr': float('nan'), 'jaccard': float('nan')}, {'pairing_method': pairing_method, 'pair_count': 0, 'skipped_empty': 0, 'parallel': False, 'workers': 1})
    workers = get_pairwise_worker_count()
    use_parallel = PAIRWISE_PARALLEL and workers > 1
    print(f"Pairwise evaluation: {len(pairs):,} pairs | {('parallel' if use_parallel else 'sequential')} | {(workers if use_parallel else 1)} worker(s)")
    if use_parallel:
        task_results = Parallel(n_jobs=workers, backend='loky', batch_size=PAIRWISE_BATCH_SIZE, pre_dispatch='2*n_jobs', return_as='generator')((delayed(compute_single_pair_metrics)(real_lookup[real_id], generated_lookup[gen_id]) for real_id, gen_id in pairs))
        result_iter = tqdm(task_results, total=len(pairs), desc=f'Pairwise trajectory metrics ({workers} proc)')
    else:
        result_iter = (compute_single_pair_metrics(real_lookup[real_id], generated_lookup[gen_id]) for real_id, gen_id in pairs)
        result_iter = tqdm(result_iter, total=len(pairs), desc='Pairwise trajectory metrics')
    hausdorff_values = []
    dtw_values = []
    edr_values = []
    jaccard_values = []
    skipped_empty = 0
    for result in result_iter:
        if result is None:
            skipped_empty += 1
            continue
        hausdorff, dtw, edr, jaccard = result
        hausdorff_values.append(hausdorff)
        dtw_values.append(dtw)
        edr_values.append(edr)
        jaccard_values.append(jaccard)

    def safe_mean(values):
        values = np.asarray(values, dtype=float)
        return float(np.nanmean(values)) if len(values) else float('nan')
    return ({'hausdorff_km': safe_mean(hausdorff_values), 'normalized_dtw_km': safe_mean(dtw_values), 'edr': safe_mean(edr_values), 'jaccard': safe_mean(jaccard_values)}, {'pairing_method': pairing_method, 'pair_count': int(len(hausdorff_values)), 'skipped_empty': int(skipped_empty), 'parallel': bool(use_parallel), 'workers': int(workers if use_parallel else 1)})

def resample_trajectory(traj: np.ndarray, n_points: int) -> np.ndarray:
    """Resample to evenly spaced path positions, following privacy_metrics.py."""
    traj = np.asarray(traj, dtype=float)
    if len(traj) == 0:
        return np.empty((0, 2), dtype=float)
    if len(traj) == 1:
        return np.repeat(traj[:1], n_points, axis=0)
    step = np.hypot(np.diff(traj[:, 0]), np.diff(traj[:, 1]) * np.cos(np.radians(traj[0, 0])))
    cumulative = np.r_[0.0, np.cumsum(step)]
    if cumulative[-1] == 0:
        return np.repeat(traj[:1], n_points, axis=0)
    target = np.linspace(0.0, cumulative[-1], n_points)
    return np.c_[np.interp(target, cumulative, traj[:, 0]), np.interp(target, cumulative, traj[:, 1])]

def sample_lookup(lookup: dict[str, np.ndarray], sample_size: int | None, seed: int) -> dict[str, np.ndarray]:
    """Reproducibly sample whole trajectories."""
    ids = np.asarray(list(lookup.keys()), dtype=object)
    if sample_size is None or len(ids) <= sample_size:
        selected = ids
    else:
        rng = np.random.default_rng(seed)
        selected = rng.choice(ids, size=sample_size, replace=False)
    return {str(tid): np.asarray(lookup[str(tid)], dtype=float) for tid in selected}

def prepare_privacy_arrays(train_lookup: dict[str, np.ndarray], test_lookup: dict[str, np.ndarray], generated_lookup: dict[str, np.ndarray]):
    train_sample = sample_lookup(train_lookup, PRIVACY_SAMPLE_SIZE, RANDOM_SEED + 1)
    test_sample = sample_lookup(test_lookup, PRIVACY_SAMPLE_SIZE, RANDOM_SEED + 2)
    gen_sample = sample_lookup(generated_lookup, PRIVACY_SAMPLE_SIZE, RANDOM_SEED + 3)
    combined = [*train_sample.values(), *test_sample.values(), *gen_sample.values()]
    if any((len(t) == 0 for t in combined)):
        raise ValueError('Privacy metric inputs contain empty trajectories.')
    lengths = {len(t) for t in combined}
    if PRIVACY_RESAMPLE_MODE == 'always':
        should_resample = True
    elif PRIVACY_RESAMPLE_MODE == 'if_needed':
        should_resample = len(lengths) != 1
    else:
        raise ValueError("PRIVACY_RESAMPLE_MODE must be 'always' or 'if_needed'.")

    def prepare(sample):
        ids = np.asarray(list(sample.keys()), dtype=object)
        if should_resample:
            arr = np.stack([resample_trajectory(t, PRIVACY_POINTS) for t in sample.values()])
        else:
            arr = np.stack(list(sample.values()))
        return (ids, arr)
    train_ids, train_arr = prepare(train_sample)
    test_ids, test_arr = prepare(test_sample)
    gen_ids, gen_arr = prepare(gen_sample)
    return (train_ids, train_arr, test_ids, test_arr, gen_ids, gen_arr, should_resample)

def matched_point_trajectory_distance_matrix(queries: np.ndarray, pool: np.ndarray, block_size: int=32) -> np.ndarray:
    """Mean matched-point Haversine distance matrix in km."""
    queries = np.asarray(queries, dtype=float)
    pool = np.asarray(pool, dtype=float)
    if queries.shape[1:] != pool.shape[1:]:
        raise ValueError(f'Privacy trajectory shapes differ: {queries.shape} vs {pool.shape}')
    pool_rad = np.radians(pool)
    output = np.empty((len(queries), len(pool)), dtype=np.float32)
    for start in tqdm(range(0, len(queries), block_size), desc='Privacy trajectory-distance blocks'):
        stop = min(start + block_size, len(queries))
        q = np.radians(queries[start:stop])
        q_lat = q[:, None, :, 0]
        q_lon = q[:, None, :, 1]
        p_lat = pool_rad[None, :, :, 0]
        p_lon = pool_rad[None, :, :, 1]
        dlat = p_lat - q_lat
        dlon = p_lon - q_lon
        h = np.sin(dlat / 2.0) ** 2 + np.cos(q_lat) * np.cos(p_lat) * np.sin(dlon / 2.0) ** 2
        h = np.clip(h, 0.0, 1.0)
        distances = EARTH_RADIUS_KM * 2.0 * np.arcsin(np.sqrt(h))
        output[start:stop] = distances.mean(axis=2).astype(np.float32)
    return output

def rank_auc(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> float:
    """Rank-based AUC matching Yun's simplified privacy script."""
    member_scores = np.asarray(member_scores, dtype=float)
    nonmember_scores = np.asarray(nonmember_scores, dtype=float)
    ranks = pd.Series(np.r_[member_scores, nonmember_scores]).rank(method='average').to_numpy()
    n_pos = len(member_scores)
    n_neg = len(nonmember_scores)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

def compute_privacy_metrics(train_lookup: dict[str, np.ndarray], test_lookup: dict[str, np.ndarray], generated_lookup: dict[str, np.ndarray], exclude_same_test_generated_id: bool=True) -> tuple[dict, dict]:
    train_ids, train_arr, test_ids, test_arr, gen_ids, gen_arr, was_resampled = prepare_privacy_arrays(train_lookup, test_lookup, generated_lookup)
    train_to_gen = matched_point_trajectory_distance_matrix(train_arr, gen_arr, block_size=PRIVACY_BLOCK_SIZE)
    test_to_gen = matched_point_trajectory_distance_matrix(test_arr, gen_arr, block_size=PRIVACY_BLOCK_SIZE)
    if exclude_same_test_generated_id:
        gen_col = {str(tid): i for i, tid in enumerate(gen_ids)}
        excluded = 0
        for row, tid in enumerate(test_ids):
            col = gen_col.get(str(tid))
            if col is not None:
                test_to_gen[row, col] = np.inf
                excluded += 1
    else:
        excluded = 0
    member_scores = -train_to_gen.min(axis=1)
    nonmember_scores = -test_to_gen.min(axis=1)
    mia_auc = rank_auc(member_scores, nonmember_scores)
    nearest_training_km = train_to_gen.min(axis=0)
    uniqueness = float(np.mean(nearest_training_km * 1000.0 > UNIQUENESS_THRESHOLD_M))
    return ({'mia_auc': mia_auc, 'uniqueness': uniqueness}, {'privacy_train_count': int(len(train_ids)), 'privacy_test_count': int(len(test_ids)), 'privacy_generated_count': int(len(gen_ids)), 'privacy_points_per_trajectory': int(train_arr.shape[1]), 'privacy_resampled': bool(was_resampled), 'privacy_self_matches_excluded': int(excluded), 'uniqueness_threshold_m': float(UNIQUENESS_THRESHOLD_M)})

def _truthy_series(series: pd.Series) -> pd.Series:
    """Interpret common bool/int/string road-mode flags."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) != 0.0
    return series.astype(str).str.strip().str.lower().isin({'1', 'true', 't', 'yes', 'y'})

def load_road_spatial_index(network_path: Path, layer: str | None=None, mode_column: str | None=None):
    """Build projected STRtree for nearest-road distance queries."""
    read_kwargs = {}
    if layer is not None:
        read_kwargs['layer'] = layer
    roads = gpd.read_file(network_path, **read_kwargs)
    if roads.crs is None:
        raise ValueError('Canonical road network has no CRS.')
    roads = roads.loc[roads.geometry.notna() & ~roads.geometry.is_empty].copy()
    if mode_column is not None:
        if mode_column not in roads.columns:
            raise ValueError(f'NETWORK_MODE_COLUMN={mode_column!r} not found. Available columns: {list(roads.columns)}')
        roads = roads.loc[_truthy_series(roads[mode_column])].copy()
    metric_crs = roads.estimate_utm_crs()
    roads_metric = roads.to_crs(metric_crs)
    tree = STRtree(roads_metric.geometry.to_numpy())
    transformer = Transformer.from_crs('EPSG:4326', metric_crs, always_xy=True)
    return (tree, transformer, {'road_count': int(len(roads_metric)), 'metric_crs': str(metric_crs), 'mode_column': mode_column})

def flatten_gps_points(lookup: dict[str, np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(traj, dtype=float) for traj in lookup.values() if len(traj) > 0]
    return np.vstack(arrays) if arrays else np.empty((0, 2), dtype=float)

def nearest_road_distances_m(gps_lookup: dict[str, np.ndarray], road_tree: STRtree, transformer: Transformer, chunk_size: int=100000) -> np.ndarray:
    """Nearest-road distance for every generated GPS point, in meters."""
    latlon = flatten_gps_points(gps_lookup)
    if len(latlon) == 0:
        return np.asarray([], dtype=float)
    distances = np.empty(len(latlon), dtype=float)
    for start in tqdm(range(0, len(latlon), chunk_size), desc='Nearest-road distances'):
        stop = min(start + chunk_size, len(latlon))
        chunk = latlon[start:stop]
        x, y = transformer.transform(chunk[:, 1], chunk[:, 0])
        point_geoms = shapely_points(x, y)
        indices, d = road_tree.query_nearest(point_geoms, return_distance=True, all_matches=False)
        local = np.full(len(chunk), np.nan, dtype=float)
        local[indices[0]] = d
        distances[start:stop] = local
    return distances

def compute_off_network_rate(generated_lookup: dict[str, np.ndarray], road_tree, transformer) -> tuple[float, dict]:
    distances_m = nearest_road_distances_m(generated_lookup, road_tree, transformer, chunk_size=OFF_NETWORK_CHUNK_SIZE)
    valid = np.isfinite(distances_m)
    if not valid.any():
        return (float('nan'), {'generated_point_count': int(len(distances_m)), 'valid_nearest_distance_count': 0})
    rate = float(np.mean(distances_m[valid] > OFF_NETWORK_THRESHOLD_M))
    return (rate, {'generated_point_count': int(len(distances_m)), 'valid_nearest_distance_count': int(valid.sum()), 'nearest_road_mean_m': float(np.mean(distances_m[valid])), 'nearest_road_p95_m': float(np.percentile(distances_m[valid], 95)), 'off_network_threshold_m': float(OFF_NETWORK_THRESHOLD_M)})

def load_rel_adjacency(rel_path: Path) -> set[tuple[int, int]]:
    """Load directed geo_id adjacency from a LibCity .rel file."""
    rel = pd.read_csv(rel_path)
    origin_candidates = ['origin_id', 'origin', 'from_id', 'from']
    dest_candidates = ['destination_id', 'destination', 'to_id', 'to']
    origin_col = next((c for c in origin_candidates if c in rel.columns), None)
    dest_col = next((c for c in dest_candidates if c in rel.columns), None)
    if origin_col is None or dest_col is None:
        raise ValueError(f'Could not identify origin/destination columns in {rel_path}. Columns: {list(rel.columns)}')
    return set(zip(rel[origin_col].astype(int), rel[dest_col].astype(int)))

def load_edge_uv(network_path: Path, layer: str | None=None) -> dict[int, tuple[int, int]]:
    """Canonical edge_id -> (u, v)."""
    read_kwargs = {}
    if layer is not None:
        read_kwargs['layer'] = layer
    edges = gpd.read_file(network_path, **read_kwargs)
    required = {'edge_id', 'u', 'v'}
    missing = required - set(edges.columns)
    if missing:
        raise ValueError(f'{network_path} missing required columns: {sorted(missing)}')
    return {int(row.edge_id): (int(row.u), int(row.v)) for row in edges[['edge_id', 'u', 'v']].itertuples(index=False)}

def compute_rsc_from_geo_rel(generated_road_lookup: dict[str, list[int]], adjacency: set[tuple[int, int]]) -> tuple[float, dict]:
    valid = 0
    total = 0
    for seq in generated_road_lookup.values():
        for a, b in zip(seq[:-1], seq[1:]):
            total += 1
            if a == b or (a, b) in adjacency:
                valid += 1
    return (float(valid / total) if total else float('nan'), {'valid_transitions': int(valid), 'total_transitions': int(total)})

def compute_rsc_from_edge_uv(generated_road_lookup: dict[str, list[int]], edge_uv: dict[int, tuple[int, int]]) -> tuple[float, dict]:
    valid = 0
    total = 0
    missing_edges = 0
    for seq in generated_road_lookup.values():
        for a, b in zip(seq[:-1], seq[1:]):
            total += 1
            if a == b:
                valid += 1
                continue
            if a not in edge_uv or b not in edge_uv:
                missing_edges += 1
                continue
            _, v_a = edge_uv[a]
            u_b, _ = edge_uv[b]
            if v_a == u_b:
                valid += 1
    return (float(valid / total) if total else float('nan'), {'valid_transitions': int(valid), 'total_transitions': int(total), 'missing_edge_transitions': int(missing_edges)})

DISTANCE_EDGES = histogram_edges(40.71, 1000)
RADIUS_EDGES = histogram_edges(10.6, 100)
