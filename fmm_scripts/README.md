## Build & Run
```bash
docker build -t fmm:ubuntu22 .
docker build -t fmm-cli .
docker run -it --rm -v "${PWD}:/workspace" fmm:ubuntu22
docker run -it --rm -v "%CD%:/workspace" fmm:ubuntu22
```
So inside the container the files are under:

`/workspace`


## FMM (Fast Map Matching):

1. Generate UBODT ("shortest path cache")

A UBODT (Upper-bounded Origin Destination Table) is a precomputed hash table used in the Fast Map Matching (FMM) algorithm to store shortest paths between node pairs within a specific maximum distance (**delta** in `ubodt_config.xml`). It speeds up map matching by avoiding repeated Dijkstra algorithm runs.

``` bash
ubodt_gen ./config/ubodt_config.xml
```

2. Perform Map Matching

``` bash
fmm ./config/fmm_config_csv_point.xml
```


## Notes:

FMM accepts a CSV point file where each row is one observation with trajectory id, longitude, latitude, and optional timestamp; the file must already be sorted by id and timestamp

CSV point file: a CSV file with a header row and columns separated by ;. Each row stores a single observation containing id(integer), x(longitude), y(latitude), timestamp(optional, integer). The file must be sorted already by id and timestamp (trajectory will be passed sequentially). The id, x, y and timestamp column names will be specified by the user.