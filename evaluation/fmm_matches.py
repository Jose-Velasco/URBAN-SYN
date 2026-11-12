"""Read FMM match files back into road sequences, keyed by the original trajectory IDs."""
from __future__ import annotations

import csv

import pandas as pd

csv.field_size_limit(2**31 - 1)  # cpath, opath and mgeom fields of long trajectories exceed csv's default limit


def read_routes(match_path, id_map_path, column: str = "cpath") -> dict[str, list[int]]:
    """Return {original trajectory ID: matched road IDs} from one FMM match file.

    match_path is an output file of FMM's `fmm` program: semicolon-separated, one row per trajectory, with at least the
    column id and the route column (other output fields are ignored). column picks the route:
      cpath  the complete route: the matched roads joined by FMM's shortest paths, so connected by construction
      opath  the road matched to each GPS point (repeats included; the metrics collapse them), gaps not filled
    id_map_path is the trip ID map written by fmm_scripts/prepare_model_fmm_points.py (columns source_traj_id and id),
    which links FMM's integer IDs back to the original trajectory IDs.

    Trajectories FMM could not match have an empty route and are left out; FMM writes some of them with ID 0, so only
    rows with a route are used.
    """
    if column not in ("cpath", "opath"):
        raise ValueError(f"FMM route column must be cpath or opath, not {column!r}")
    id_map = pd.read_csv(id_map_path, dtype={"source_traj_id": str})
    if not {"source_traj_id", "id"} <= set(id_map.columns):
        raise ValueError(f"{id_map_path}: expected the columns source_traj_id and id, as written by "
                         "fmm_scripts/prepare_model_fmm_points.py")
    original = dict(zip(id_map["id"].astype(int), id_map["source_traj_id"]))

    routes = {}
    with open(match_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        if not {"id", column} <= set(reader.fieldnames or []):
            raise ValueError(f"{match_path}: expected FMM output with the columns id and {column} "
                             f"(add <{column}/> to the FMM config's output fields)")
        for row in reader:
            route = [int(x) for x in row[column].split(",") if x]
            if not route:
                continue
            traj_id = original[int(row["id"])]
            if traj_id in routes:
                raise ValueError(f"{match_path}: trajectory {traj_id} has two matched routes")
            routes[traj_id] = route
    return routes
