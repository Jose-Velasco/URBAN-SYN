# Baseline: TS-TrajGen

This baseline is based on:
https://github.com/WenMellors/TS-TrajGen

## Source: 
- Repo: https://github.com/WenMellors/TS-TrajGen/tree/master
- Source commit/tag: 

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
           --out_dir ./outputs/nyc \
           --dataset_name nyc \
           --train_ratio 0.8
    ```
2. Run  `preprocess_pretrain_input.py`


