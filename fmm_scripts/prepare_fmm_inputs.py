"""
Creates a directed edge layer with columns fid, u, v, and geometry, which matches the common FMM pattern

FMM accepts a CSV point file where each row is one observation with trajectory id,
longitude, latitude, and optional timestamp; the file must already be sorted by id and timestamp

"""
import osmnx as ox
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Literal

def build_fmm_network_from_place(place: str, out_shp: str, mode: Literal["drive", "walk", "bike", "all"]) -> None:
    """ Build the road network with OSMnx.

        the exported network needs fid, u, and v columns to be compatible with FMM
    """

    # Download drivable network
    G = ox.graph_from_place(place, network_type=mode)

    # Convert to GeoDataFrames
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)

    # Flatten (u, v, key) index into columns
    edges_gdf = edges_gdf.reset_index()

    # FMM-compatible columns
    edges_gdf["fid"] = np.arange(len(edges_gdf), dtype="int64")

    # Keep required columns first
    keep = ["fid", "u", "v", "geometry"]
    optional = ["osmid", "length", "highway", "name", "oneway"]
    for col in optional:
        if col not in edges_gdf.columns:
            edges_gdf[col] = None

    edges_gdf = edges_gdf[keep + optional].copy()

    # shp_path = os.path.join(out_dir, "edges.shp")
    # shp_path = output_dir_path / "edges.shp"
    edges_gdf.to_file(out_shp)



def build_fmm_points_csv(
    parquet_path: str,
    out_csv: str,
    trip_id_map_csv: str | None = None
) -> pd.DataFrame:
    """
    CSV point file: a CSV file with a header row and columns separated by ;. 
    Each row stores a single observation containing id(integer), x(longitude), y(latitude), timestamp(optional, integer). 
    
    The file must be sorted already by id and timestamp (trajectory will be passed sequentially). The id, x, y and timestamp column names will be specified by the user.

    Since our tid is only unique within a user, build a unique trip key from uid + tid. FMM's docs say id is an integer, so remap each unique trip key to an integer trajectory id.
    """
    df = pd.read_parquet(parquet_path).copy()

    # Unique trip key
    df["trip_key"] = df["uid"].astype(str) + "_" + df["tid"].astype(str)

    # Integer trajectory ids for FMM
    trip_keys = pd.Index(df["trip_key"].unique())
    trip_id_map = pd.DataFrame({
        "trip_key": trip_keys,
        "id": range(len(trip_keys))
    })

    df = df.merge(trip_id_map, on="trip_key", how="left")

    # Integer unix timestamp in seconds
    df["timestamp"] = pd.to_datetime(df["datetime"]).astype("int64") // 10**9

    # Sort exactly as required
    df = df.sort_values(["id", "timestamp"]).reset_index(drop=True)

    gps_df = pd.DataFrame({
        "id": df["id"].astype("int64"),
        "x": df["lng"].astype(float),
        "y": df["lat"].astype(float),
        "timestamp": df["timestamp"].astype("int64"),
    })

    gps_df.to_csv(out_csv, sep=";", index=False)

    if trip_id_map_csv is not None:
        trip_id_map.to_csv(trip_id_map_csv, index=False)

    return trip_id_map

if __name__ == '__main__':
    BASE_DATA_DIR = "../data/nyc_output_tabular/output"
    # BASE_OUTPUT_DIR = "../outputs/nyc_output_tabular"
    BASE_OUTPUT_DIR = "./data"

    PLACE = "New York City, New York, USA"
    ROAD_NETWORK_OUT_DIR = f"{BASE_OUTPUT_DIR}/fmm_nyc.shp"
    # {"all", "all_public", "bike", "drive", "drive_service", "walk"} What type of street network to retrieve
    NETWORK_TYPE = "drive"

    road_network_out_dir_path = Path(ROAD_NETWORK_OUT_DIR)
    road_network_out_dir_path.parent.mkdir(parents=True, exist_ok=True)

    GPS_TRAJ_PARQUET_PATH = f"{BASE_DATA_DIR}/traj_cleaned.parquet"
    GPS_TRAJ_FMM_OUTPUT_PATH = f"{BASE_OUTPUT_DIR}/nyc_gps_points_fmm_ready.csv"
    TRIP_ID_MAP_CSV = f"{BASE_OUTPUT_DIR}/nyc_gps_points_fmm_trip_id_map.csv"
    gps_traj_fmm_output_path = Path(GPS_TRAJ_FMM_OUTPUT_PATH)
    gps_traj_fmm_output_path.parent.mkdir(parents=True, exist_ok=True)



    build_fmm_network_from_place(PLACE, ROAD_NETWORK_OUT_DIR, mode=NETWORK_TYPE)

    trip_map = build_fmm_points_csv(
        parquet_path=GPS_TRAJ_PARQUET_PATH,
        out_csv=GPS_TRAJ_FMM_OUTPUT_PATH,
        trip_id_map_csv=TRIP_ID_MAP_CSV,
    )

# TODO: then next stage after matching: turn tpath into nyc.geo, nyc.rel (LibCity formats).