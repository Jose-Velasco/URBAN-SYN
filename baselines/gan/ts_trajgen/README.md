# Baseline: TS-TrajGen

This baseline is based on:
https://github.com/WenMellors/TS-TrajGen

## Source: 
- Repo: https://github.com/WenMellors/TS-TrajGen/tree/master
- Source commit/tag: a71502d3a834f0069475ba3c71bb56f851e32a62

## Undocumented preprocessing mismatch

...

## Local baseline path
- `baselines/gan/ts_trajgen/repo/`

## Environment
- N/A





## Build & Run TODO: fix this section code
```bash
# docker build -t fmm-cli .
# docker run -it --rm -v "%CD%:/workspace" fmm:ubuntu22
```

So inside the container the files are under (TODO: once docker container is implemented):

<!-- `/workspace` -->


## Preprocess Data:
There are two major steps:

I. **Pre-preprocesses** data into format TS-TrajGen `preprocess_pretrain_input.py` expects

II. Run  `preprocess_pretrain_input.py` on the *Pre-preprocesses* data


1. **Pre-preprocesses** data
    
    1.1 run b`uild_tstrajgen_inputs.py` on your dataset: example command:
    ``` bash
    uv run build_tstrajgen_inputs.py \
           --network_path ../../../fmm_scripts/data/fmm_nyc.shp \
           --fmm_match_path ../../../fmm_scripts/output/nyc_fmm_match.csv \
           --parquet_path ../../../data/nyc_output_tabular/output/traj_cleaned.parquet \
           --trip_id_map_csv ../../../fmm_scripts/data/nyc_gps_points_fmm_trip_id_map.csv \
           --out_dir ./outputs/nyc \
           --log_dir ./outputs/logs \
           --dataset_name nyc \
           --min_len 2 \
           --min_delta_seconds 0.5 \
           --train_ratio 0.8
    ```
2. Run  `preprocess_pretrain_input.py`


