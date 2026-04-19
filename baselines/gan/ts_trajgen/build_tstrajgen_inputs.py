from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd


# HELPERS
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


def parse_cpath(value) -> list[int]:
    """
    Parse FMM `cpath` field into a list of edge IDs (fid).

    Handles multiple formats:
        - "[1,2,3]"
        - "1,2,3"
        - "1 2 3"
        - NaN / empty

    Parameters
    ----------
    value : Any
        Raw cpath value from FMM output.

    Returns
    -------
    list[int]
        List of edge IDs.
    """
    if pd.isna(value):
        return []

    s = str(value).strip()
    if not s:
        return []

    if s.startswith("[") and s.endswith("]"):
        try:
            return [int(x) for x in ast.literal_eval(s)]
        except Exception:
            pass

    if "," in s:
        return [int(x.strip()) for x in s.split(",") if x.strip()]

    return [int(x.strip()) for x in s.split() if x.strip()]


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

    fid_to_geo = dict(zip(edges["fid"], edges["geo_id"]))

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


def build_mm_csvs(
    fmm_path,
    fid_to_geo,
    train_ratio,
    random_state,
    min_len: int = 2,
    verbose: bool = False,
):
    """
    Build TS-TrajGen train/test CSVs from FMM output.

    Converts:
        cpath (fid list) -> rid_list (geo_id list)

        
    cpath is added by FMM

    geo_id is the column in .geo (we created it ourselves from the road network file)
    
    Parameters
    ----------
    fmm_path : Path
        CSV file from FMM output.
    fid_to_geo : dict
        Mapping from fid -> geo_id.
    train_ratio : float
        Fraction of data for training.
    random_state : int
        Seed for reproducibility.
    min_len : int, optional
        Minimum number of edges required in a trajectory (default=2).
        Sequences shorter than this are discarded.
    verbose : bool, optional
        If True, print detailed processing statistics.

    Returns
    -------
    train_df : pd.DataFrame
    test_df : pd.DataFrame
    """

    fmm = pd.read_csv(fmm_path, sep=";")

    rows = []

    # stats tracking
    total = 0
    skipped_empty = 0
    skipped_short_raw = 0
    skipped_short_mapped = 0
    skipped_unmapped = 0

    for _, row in fmm.iterrows():
        total += 1

        path = parse_cpath(row["cpath"])

        if not path:
            skipped_empty += 1
            continue

        if len(path) < min_len:
            skipped_short_raw += 1
            continue

        # map fid -> geo_id
        geo_path = []
        unmapped_flag = False

        for f in path:
            if f in fid_to_geo:
                geo_path.append(fid_to_geo[f])
            else:
                unmapped_flag = True

        if unmapped_flag:
            skipped_unmapped += 1

        if len(geo_path) < min_len:
            skipped_short_mapped += 1
            continue

        rows.append({
            "traj_id": str(row["id"]),
            "rid_list": str(geo_path)
        })

    df = pd.DataFrame(rows)

    # split
    train = df.sample(frac=train_ratio, random_state=random_state)
    test = df.drop(train.index)

    # verbose stats
    if verbose:
        kept = len(df)
        print("\n MM CSV BUILD STATS")
        print(f"Total input trajectories: {total:,}")
        print(f"Kept trajectories:        {kept:,}")
        print(f"Kept ratio:              {kept/total:.3f}")
        print()
        print("Skipped breakdown:")
        print(f"  Empty paths:           {skipped_empty:,}")
        print(f"  Too short (raw):       {skipped_short_raw:,}")
        print(f"  Too short (mapped):    {skipped_short_mapped:,}")
        print(f"  Had unmapped edges:    {skipped_unmapped:,}")
        print()

    return train.reset_index(drop=True), test.reset_index(drop=True)


# MAIN
def main():
    parser = argparse.ArgumentParser(description="Build TS-TrajGen inputs")

    parser.add_argument(
        "--network_path",
        type=str,
        required=True,
        help="Path to road network file (e.g., .shp/.gpkg) with columns [fid, u, v, geometry]."
    )
    
    parser.add_argument(
        "--fmm_match_path",
        type=str,
        required=True,
        help="Path to FMM output CSV containing 'id' and 'cpath' columns."
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/NYC",
        help="Output directory for generated files (.geo, .rel, *_mm_train/test.csv)."
    )
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="nyc",
        help="Prefix for output files (e.g., 'nyc' -> nyc.geo, nyc.rel, etc.)."
    )
    
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Fraction of trajectories used for training (rest used for test split)."
    )
    
    parser.add_argument(
        "--random_state",
        type=int,
        default=101,
        help="Random seed for reproducible train/test splitting."
    )

    parser.add_argument(
        "--min_len",
        type=int,
        default=2,
        help="Minimum trajectory length (in edges); sequences with fewer edges are skipped."
    )

    args = parser.parse_args()

    network_path = Path(args.network_path)
    fmm_match_path = Path(args.fmm_match_path)
    out_dir = Path(args.out_dir)
    dataset_name = args.dataset_name

    ensure_dir(out_dir)

    print("Building .geo...")
    geo_df, edges_df, fid_to_geo = build_geo(network_path)
    geo_df.to_csv(out_dir / f"{dataset_name}.geo", index=False)

    print("Building .rel...")
    rel_df = build_rel(edges_df)
    rel_df.to_csv(out_dir / f"{dataset_name}.rel", index=False)

    print("Building train/test CSV...")
    train_df, test_df = build_mm_csvs(
        fmm_match_path,
        fid_to_geo,
        args.train_ratio,
        args.random_state,
        min_len=args.min_len,
        verbose=True
    )

    train_df.to_csv(out_dir / f"{dataset_name}_mm_train.csv", index=False)
    test_df.to_csv(out_dir / f"{dataset_name}_mm_test.csv", index=False)

    print("\nDone!")
    print(f".geo rows: {len(geo_df):,}")
    print(f".rel rows: {len(rel_df):,}")
    print(f"train trajs: {len(train_df):,}")
    print(f"test trajs: {len(test_df):,}")


if __name__ == "__main__":
    main()