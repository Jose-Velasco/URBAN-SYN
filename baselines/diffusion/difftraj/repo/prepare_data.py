"""
Build the DiffTraj training / test files from the MAT-Builder parquet (script version of
DiffTraj_DataFormat.ipynb, in the same folder).

One trajectory per MOVE (enriched_moves.parquet, default) or per user track
(traj_cleaned.parquet, --source traj_cleaned). Test trips are the ones in
fmm_scripts/nyc_test.csv (the same list as Diff-RNTraj, read from there, not copied);
all moves of a test (uid, tid) go to the test set, like the Diff-RNTraj converter.
Normalisation statistics come from the TRAIN trips only.

Writes (into --out-dir, default: this folder):
    traj.npy                     train trips, (N, 2, 256), normalised      -> main.py
    head.npy                     train conditions, (N, 8), normalised      -> main.py
    head_test.npy                test conditions                            -> generate.py
    traj_stats.npy               train-only stats                           -> generate.py
    real_trajectories_test.npy   raw test trips, same order as head_test    (for evaluation)
    prepare_data_info.json       settings, grid and train / test counts
  with --extras also:
    real_trajectories_train.npy, traj_test.npy, trip_keys_*.npy, traj_ids_*.npy, modes_*.npy

head columns: [departure slot (5 min), distance m, duration min, #points, avg m per step,
               avg speed m/min, start cell, end cell]   (cells: 16x16 grid over the data)

Usage
    python prepare_data.py                  # data/.../enriched_moves.parquet + fmm_scripts/nyc_test.csv
    python prepare_data.py --test-csv nyc_mm_test.csv --trip-map nyc_gps_points_fmm_trip_id_map.csv
    python prepare_data.py --parquet output/traj_cleaned.parquet      # one trajectory per user track
    (nyc_test.csv already has uid / tid, so it needs no trip map)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

EARTH_R_KM = 6371.0


def as_key_str(s):
    # '3', 3 and 3.0 all become '3', so keys match the uid_tid trip_key of the test list
    return s.astype(str).str.replace(r"\.0$", "", regex=True)


def resample_trajectory(x, length):
    """Same linear interpolation as DiffTraj (utils/utils.py)."""
    len_x = len(x)
    t = np.arange(length) * (len_x - 1) / (length - 1)
    return np.stack([np.interp(t, np.arange(len_x), x[:, k]) for k in range(2)], axis=1)


def haversine_km(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lng2 - lng1) / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


def load_points(args):
    if args.source == "enriched_moves":
        df = pd.read_parquet(args.parquet).reset_index()   # tid, move_id are the index
        if args.modes:
            df = df[df["label"].isin(args.modes)].copy()
        df["orig_tid"] = df["tid"]
        df["tid"] = as_key_str(df["uid"]) + "_" + as_key_str(df["tid"]) + "_" + as_key_str(df["move_id"])
        df["datetime"] = pd.to_datetime(df["datetime"])
    else:
        df = pd.read_parquet(args.parquet)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["orig_tid"] = df["tid"]
        # split whole-day tracks by date; later dates get the id <tid>_d<k>
        date = df["datetime"].dt.date
        first = df.groupby("tid")["datetime"].transform("min").dt.date
        rank = df.assign(_d=date).groupby("tid")["_d"].rank(method="dense").astype(int) - 1
        df["tid"] = np.where(date == first, df["tid"].astype(str),
                             df["tid"].astype(str) + "_d" + rank.astype(str))
    if "label" not in df.columns:
        df["label"] = np.nan
    df["tid"] = df["tid"].astype(str)
    # rows of one trajectory together, in time order (stable: keeps the file order on ties)
    df = df.sort_values(["tid", "datetime"], kind="mergesort").reset_index(drop=True)
    return df


def load_test_keys(args):
    test = pd.read_csv(args.test_csv)
    if {"uid", "tid"} <= set(test.columns):
        test = test.astype({"uid": str, "tid": str})
        test["trip_key"] = as_key_str(test["uid"]) + "_" + as_key_str(test["tid"])
        if "traj_id" not in test.columns:
            test["traj_id"] = np.nan
    else:
        if not args.trip_map:
            raise SystemExit(f"{args.test_csv} has no uid/tid columns: pass --trip-map")
        fmm_map = pd.read_csv(args.trip_map)
        test = test[["traj_id"]].merge(fmm_map, left_on="traj_id", right_on="id", how="left")
        missing = test["trip_key"].isna().sum()
        assert missing == 0, f"{missing} test traj_ids are not in the trip map"
        test[["uid", "tid"]] = test["trip_key"].str.split("_", n=1, expand=True)
    return test[["traj_id", "uid", "tid", "trip_key"]]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # the MAT-Dataset parquet in URBAN-SYN/data (run from baselines/diffusion/difftraj/repo/)
    p.add_argument("--parquet", default="../../../../data/nyc_output_tabular/output/enriched_moves.parquet")
    p.add_argument("--source", choices=["enriched_moves", "traj_cleaned"], default=None,
                   help="default: from the parquet file name")
    # the shared test list in URBAN-SYN/fmm_scripts (also used by Diff-RNTraj)
    p.add_argument("--test-csv", default="../../../../fmm_scripts/nyc_test.csv",
                   help="test trips: a CSV with uid, tid, or with traj_id (+ --trip-map)")
    p.add_argument("--trip-map", default="nyc_gps_points_fmm_trip_id_map.csv")
    p.add_argument("--modes", type=int, nargs="*", default=None,
                   help="keep only these labels, e.g. 2 3 6 (default: all modes)")
    p.add_argument("--min-points", type=int, default=2)
    p.add_argument("--traj-len", type=int, default=256)
    p.add_argument("--grid-size", type=int, default=16)
    p.add_argument("--out-dir", default=".")
    p.add_argument("--extras", action="store_true", help="also save the optional files")
    args = p.parse_args()
    if args.source is None:
        args.source = "traj_cleaned" if "traj_cleaned" in Path(args.parquet).name else "enriched_moves"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---------------- points and test trips -------------------------------------------------
    df = load_points(args)
    n_traj_all = df["tid"].nunique()
    per_user = df.groupby("uid")["tid"].nunique()
    print(f"source: {args.source} ({args.parquet})  modes: {args.modes or 'all'}")
    print(f"points: {len(df):,}  users: {len(per_user):,}  trajectories: {n_traj_all:,} "
          f"({per_user.mean():.2f} per user)")

    test = load_test_keys(args)
    test_keys = set(test["trip_key"])
    df["trip_key"] = as_key_str(df["uid"]) + "_" + as_key_str(df["orig_tid"])
    found = test_keys & set(df["trip_key"])
    print(f"test trips listed: {len(test_keys):,} | found in the data: {len(found):,}")

    # ---------------- grid (16x16 over all points, as in the notebook) ------------------------
    lat_min, lat_max = df["lat"].min(), df["lat"].max()
    lng_min, lng_max = df["lng"].min(), df["lng"].max()
    g = args.grid_size

    def grid_id(lat, lng):
        x = np.clip(((lat - lat_min) / ((lat_max - lat_min) / g)).astype(int), 0, g - 1)
        y = np.clip(((lng - lng_min) / ((lng_max - lng_min) / g)).astype(int), 0, g - 1)
        return x * g + y

    # ---------------- per-trajectory features (vectorised) -------------------------------------
    tid = df["tid"].to_numpy()
    starts = np.flatnonzero(np.r_[True, tid[1:] != tid[:-1]])
    ends = np.r_[starts[1:], len(df)]
    lat, lng = df["lat"].to_numpy(float), df["lng"].to_numpy(float)
    step_km = np.r_[haversine_km(lat[:-1], lng[:-1], lat[1:], lng[1:]), 0.0]
    step_km[ends - 1] = 0.0                                  # no step across two trajectories
    dist_m = np.add.reduceat(step_km, starts) * 1000.0
    n_pts = ends - starts
    t_sec = (df["datetime"] - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()
    dur_min = (t_sec[ends - 1] - t_sec[starts]) / 60.0
    dt0 = df["datetime"].iloc[starts]
    dep_slot = ((dt0.dt.hour * 60 + dt0.dt.minute) // 5).to_numpy()
    start_cell = grid_id(lat[starts], lng[starts])
    end_cell = grid_id(lat[ends - 1], lng[ends - 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        avg_speed = np.where(dur_min > 0, dist_m / dur_min, 0.0)
        avg_dist = np.where(n_pts > 1, dist_m / (n_pts - 1), 0.0)

    keep = n_pts >= args.min_points
    print(f"trajectories with fewer than {args.min_points} points dropped: {(~keep).sum():,}")
    idx = np.flatnonzero(keep)

    head = np.stack([dep_slot, dist_m, dur_min, n_pts, avg_dist, avg_speed, start_cell, end_cell],
                    axis=1)[idx].astype(float)
    raw, traj = [], np.empty((len(idx), 2, args.traj_len))
    latlng = np.stack([lat, lng], axis=1)
    for k, i in enumerate(idx):
        coords = latlng[starts[i]:ends[i]]
        raw.append(coords)
        traj[k] = resample_trajectory(coords, args.traj_len).T
    trip_keys = df["trip_key"].to_numpy()[starts][idx]
    traj_ids = tid[starts][idx]
    modes = df["label"].to_numpy(float)[starts][idx]

    is_test = np.isin(trip_keys, list(test_keys))
    tr, te = np.flatnonzero(~is_test), np.flatnonzero(is_test)
    assert not set(trip_keys[tr]) & set(trip_keys[te])
    dropped = found - set(trip_keys[te])
    print(f"train: {len(tr):,}  test: {len(te):,}"
          + (f"  ({len(dropped)} test trips had < {args.min_points} points)" if dropped else ""))

    # ---------------- normalise with TRAIN statistics -------------------------------------------
    traj_mean = traj[tr].mean(axis=(0, 2))
    traj_std = traj[tr].std(axis=(0, 2))
    traj_std[traj_std == 0] = 1.0
    traj_norm = (traj - traj_mean[None, :, None]) / traj_std[None, :, None]

    hmean = head[tr, 1:6].mean(axis=0)
    hstd = head[tr, 1:6].std(axis=0)
    hstd[hstd == 0] = 1.0
    head_norm = head.copy()
    head_norm[:, 1:6] = (head[:, 1:6] - hmean) / hstd
    len_mean, len_std = float(head[tr, 3].mean()), float(head[tr, 3].std()) or 1.0
    stats = {
        "mean": traj_mean, "std": traj_std,
        "hmean": np.array([0, *hmean.tolist()], dtype=np.float32),
        "hstd": np.array([1, *hstd.tolist()], dtype=np.float32),
        "len_mean": len_mean, "len_std": len_std,
    }

    def obj(arrs):
        a = np.empty(len(arrs), dtype=object)
        for k, x in enumerate(arrs):
            a[k] = x
        return a

    files = {
        "traj.npy": traj_norm[tr],
        "head.npy": head_norm[tr],
        "head_test.npy": head_norm[te],
        "real_trajectories_test.npy": obj([raw[k] for k in te]),
    }
    if args.extras:
        files.update({
            "traj_test.npy": traj_norm[te],
            "real_trajectories_train.npy": obj([raw[k] for k in tr]),
            "trip_keys_train.npy": trip_keys[tr], "trip_keys_test.npy": trip_keys[te],
            "traj_ids_train.npy": traj_ids[tr], "traj_ids_test.npy": traj_ids[te],
            "modes_train.npy": modes[tr], "modes_test.npy": modes[te],
        })
    for name, arr in files.items():
        np.save(out / name, arr, allow_pickle=True)
        print(f"  {name:<30} {arr.shape}")
    np.save(out / "traj_stats.npy", stats, allow_pickle=True)
    print(f"  {'traj_stats.npy':<30} mean={traj_mean.tolist()} len_mean={len_mean:.2f}")
    (out / "prepare_data_info.json").write_text(json.dumps({
        "source": args.source, "parquet": args.parquet, "test_csv": args.test_csv,
        "modes": args.modes, "min_points": args.min_points, "traj_len": args.traj_len,
        "grid": {"size": g, "lat": [float(lat_min), float(lat_max)], "lng": [float(lng_min), float(lng_max)]},
        "train": int(len(tr)), "test": int(len(te)), "seconds": round(time.time() - t0, 1),
    }, indent=2))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
