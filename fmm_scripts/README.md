## Docker Container

### Build

From the `fmm_scripts/` directory:

```bash
docker build -t urban-syn-fmm:ubuntu22 .
```

### Run

#### Windows — Git Bash

```bash
docker run -it --rm \
    --mount type=bind,source="$(pwd -W)",target=/workspace \
    urban-syn-fmm:ubuntu22
```

#### Linux / macOS

```bash
docker run -it --rm \
    --mount type=bind,source="${PWD}",target=/workspace \
    urban-syn-fmm:ubuntu22
```

The current `fmm_scripts/` directory is bind-mounted into the container at:

```text
/workspace
```

Changes made to files under `/workspace` inside the container are reflected in the host `fmm_scripts/` directory.



## FMM (Fast Map Matching):

### 1. Generate FMM inputs

    (In devcontainer unless trajectory_parquet is in FMM container)


- Run from the development environment containing the trajectory Parquet file:
- Saving to disk in .shp formate takes some time for large graphs


```bash
uv run prepare_fmm_inputs.py \
    --config ./config/nyc.yaml \
    --canonical_road_network_output ./data/road_network/nyc.gpkg \
    --canonical_road_network_layer roads \
    --fmm_road_network_output ./data/fmm/fmm_nyc.shp \
    --trajectory_parquet ../data/nyc_output_tabular/output/traj_cleaned.parquet \
    --gps_output ./data/nyc_gps_points_fmm_ready.csv \
    --trip_id_map_output ./data/nyc_gps_points_fmm_trip_id_map.csv \
    --log_file ./outputs/logs/prepare_fmm_nyc.log
```

This creates two road-network outputs:

- `nyc.gpkg`: canonical road network with OSM metadata for shared project use.
- `fmm_nyc.shp`: minimal FMM-compatible network containing `fid`, `u`, `v`, and `geometry`.

The unified road network is configured in `config/nyc.yaml` and combines the configured drive, bike, and walk OSM filters.

### 2. Generate UBODT ("shortest path cache")

Run inside the urban-syn-fmm:ubuntu22 container:

- A UBODT (Upper-bounded Origin Destination Table) is a precomputed hash table used in the Fast Map Matching (FMM) algorithm to store shortest paths between node pairs within a specific maximum distance (**delta** in `ubodt_config.xml`). It speeds up map matching by avoiding repeated Dijkstra algorithm runs.

- **important**: If mode is changed make sure to to date this in ubodt_config.xml (<mode>all</mode>) one of "drive|walk|bike|all" [based on FMM docs](https://fmm-wiki.github.io/docs/documentation/configuration/#ubodt_gen)

``` bash
ubodt_gen ./config/ubodt_config.xml
```

UBODT (Upper-Bounded Origin Destination Table) precomputes shortest-path information used by FMM.

The maximum precomputed path distance is controlled by `delta` in:

`config/ubodt_config.xml`

Current NYC configuration uses:

`<delta>0.01</delta>`

The UBODT configuration should use the same `fid`, `u`, and `v` fields as the generated FMM road network.


### 3.Perform Map Matching

Run inside the FMM container:

``` bash
fmm ./config/fmm_config_csv_point.xml
```

Current map-matching parameters:

`<k>8</k>`

`<r>0.003</r>`

`<gps_error>0.0005</gps_error>`

The road network and GPS coordinates use EPSG:4326, so FMM distance parameters such as `r`, `gps_error`, and `delta` are expressed in degrees.


## Input Notes:

FMM accepts a point CSV where each row represents one GPS observation.

Expected columns:

```text
id;x;y;timestamp
```

Where:

- `id`: integer trajectory ID
- `x`: longitude
- `y`: latitude
- `timestamp`: Unix timestamp

The CSV must be sorted by:

```text
id, timestamp
```

The preprocessing script also writes a trip-ID mapping file so the integer FMM IDs can be mapped back to the original trajectory identifiers.

The file must be sorted already by id and timestamp (trajectory will be passed sequentially). 