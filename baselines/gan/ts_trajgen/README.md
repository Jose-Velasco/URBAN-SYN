# Baseline: TS-TrajGen

This directory contains the adapted TS-TrajGen baseline used for the NYC trajectory-generation experiments.

The implementation is based on the original TS-TrajGen repository and keeps the model architecture and overall pipeline as close to the authors' code as practical, while replacing hardcoded Xian-specific assumptions with explicit dataset paths, CLI arguments, and YAML-driven configuration.

## Source

- Original repository: https://github.com/WenMellors/TS-TrajGen
- Source commit/tag used for this baseline: `a71502d3a834f0069475ba3c71bb56f851e32a62`
- Local baseline path: `baselines/gan/ts_trajgen/repo/`

## NYC adaptation notes

The original TS-TrajGen code is heavily tied to its original datasets and filenames. This refactor keeps the model logic intact while making the NYC pipeline explicit and reproducible.

Important project-specific changes include:

- Direct NYC dataset paths instead of pretending NYC files are Xian files.
- A feature-complete `nyc.geo` produced directly by `build_tstrajgen_inputs.py`; the older `ensure_geo_feature_columns.py` compatibility step is no longer needed.
- Config-driven model/training parameters through `configs/ts_trajgen_nyc.yaml`.
- Explicit road-level and region-level train/eval/test files.
- Training-derived helper statistics are built from the training split only where possible to avoid evaluation/test leakage.
- Two generation paths are retained:
  - `our_model_generate.py` follows the original authors' generation behavior and loads the separately pretrained Function G / Function H checkpoints.
  - `our_model_generate_using_gan.py` loads the full road-level and region-level GAN-trained `GeneratorV4` checkpoints.

## Data-split policy

The pipeline uses separate training, validation/evaluation, and test data.

- **Train:** used to fit model parameters and construct training-derived helper statistics such as OD-route histories, time distributions, and region-transition frequencies.
- **Eval/validation:** used during model development/pretraining for validation behavior and checkpoint selection where supported.
- **Test:** kept separate from training-derived statistics and used as the held-out OD/reference input for final generation/evaluation.

The pretraining scripts still accept explicit train/eval/test filenames because that mirrors the TS-TrajGen workflow. Do not use test-set metrics to tune model or hyperparameter decisions.

## Execution locations

There are two execution contexts in this README.

**Project/dev environment:** Step 1 runs from the `baselines/gan/ts_trajgen/` directory, where `build_tstrajgen_inputs.py`, `repo/`, and `datasets/` are available.

**TS-TrajGen container:** Steps 2 onward run from the TS-TrajGen repository working directory, typically `/workspace/repo`.

Expected container mounts are conceptually:

```text
/workspace/repo      # TS-TrajGen repository
/workspace/datasets  # project datasets, including NYC
/workspace/outputs   # generated outputs
```

Scripts under `script/` that import project modules should be run with `python -m script.<name>` when shown below.

## Container

The Docker setup is GPU-enabled through `gpus: all`.

### Build

```bash
docker compose build
```

### Rebuild

```bash
docker compose down
docker compose build
```

### Open a shell

```bash
docker compose run --rm ts-trajgen
```

## Pipeline

The commands below are intentionally explicit. Arguments are shown in full so each step can be reproduced or later wrapped without relying on hidden defaults.

> **Smoke-test note:** the GAN commands currently use `--debug True`, which intentionally reduces epochs/sample counts for a fast pipeline test. Use `--debug False` for a full training run after the pipeline is verified.

For the automated preprocessing, pretraining, GAN training, generation,
Docker, see [PIPELINE.md](./PIPELINE.md).

### 1. Build TS-TrajGen road-network and map-matched trajectory inputs

Run this in the project/dev environment, not from `/workspace/repo` inside the TS-TrajGen container.

This converts the canonical NYC road network and FMM map-matching output into TS-TrajGen-compatible road-level inputs.

```bash
uv run build_tstrajgen_inputs.py \
       --network_path ../../../fmm_scripts/data/road_network/nyc.gpkg \
       --fmm_match_path ../../../fmm_scripts/outputs/nyc_fmm_match.csv \
       --parquet_path ../../../data/nyc_output_tabular/output/traj_cleaned.parquet \
       --trip_id_map_csv ../../../fmm_scripts/data/nyc_gps_points_fmm_trip_id_map.csv \
       --config ./repo/configs/ts_trajgen_nyc.yaml \
       --out_dir ./datasets/nyc \
       --log_dir ./datasets/logs \
       --dataset_name nyc \
       --min_len 2 \
       --min_delta_seconds 0.5 \
       --train_ratio 0.8 \
       --no-interpolate_intermediate_edges
```

**Persistent outputs:**

- `nyc_mm_train.csv`
- `nyc_mm_test.csv`
- `nyc.geo`
- `nyc.rel`

### 2. Build road-level TS-TrajGen preprocessing artifacts

Run inside the TS-TrajGen container.

This creates the road adjacency/GPS lookup artifacts and road-level pretraining examples consumed by the Function G and Function H pretraining stages.

```bash
python -m script.preprocess_pretrain_input \
       --dataset_name nyc \
       --data_root ../datasets/ \
       --dataset_prefix nyc \
       --config ./configs/ts_trajgen_nyc.yaml
```

**Persistent outputs:**

- `adjacent_list.json`
- `rid_gps.json`
- `nyc_pretrain_input_train.csv`
- `nyc_pretrain_input_eval.csv`
- `nyc_pretrain_input_test.csv`

### 3. Pretrain road-level Function H / GAT

This stage learns the road-level graph representation and also produces the road adjacency matrix and node-feature tensor required by later stages.

```bash
python pretrain_gat_fc.py \
    --dataset_name nyc \
    --data_root ../datasets \
    --device cuda:0 \
    --debug False \
    --train True \
    --config ./configs/ts_trajgen_nyc.yaml \
    \
    --geo_path ../datasets/nyc/nyc.geo \
    --rel_filename nyc.rel \
    --map_manager_cache_dir ../datasets/nyc \
    \
    --adjacent_np_filename adjacent_mx.npz \
    --node_feature_filename node_feature.pt \
    --rid_gps_filename rid_gps.json \
    \
    --train_filename nyc_pretrain_input_train.csv \
    --eval_filename nyc_pretrain_input_eval.csv \
    --test_filename nyc_pretrain_input_test.csv \
    \
    --save_dir ./save/nyc \
    --save_file_name gat_fc.pt \
    --temp_dir ./temp/nyc/gat
```

**Persistent outputs:**

- `node_feature.pt`
- `adjacent_mx.npz`
- `gat_fc.pt`
- MapManager bounds cache under the configured cache directory

### 4. Pretrain road-level Function G

This stage pretrains the road-level sequential generator using the encoded road trajectories.

```bash
python pretrain_function_g_fc.py \
    --dataset_name nyc \
    --data_root ../datasets \
    --device cuda:0 \
    --train True \
    --config ./configs/ts_trajgen_nyc.yaml \
    --geo_path ../datasets/nyc/nyc.geo \
    \
    --train_filename nyc_pretrain_input_train.csv \
    --eval_filename nyc_pretrain_input_eval.csv \
    --test_filename nyc_pretrain_input_test.csv \
    \
    --save_dir ./save/nyc \
    --save_file_name function_g_fc.pt \
    --temp_dir ./temp/nyc/function_g
```

**Persistent output:** `function_g_fc.pt`

### 5. Convert the road graph to KaHIP input format

```bash
python ./script/process_kahip_graph_format.py \
       --dataset_name nyc \
       --data_root ../datasets \
       --geo_filename nyc.geo \
       --rel_filename nyc.rel \
       --graph_filename nyc.graph \
       --rid2new_filename rid2new.json \
       --new2rid_filename new2rid.json
```

**Persistent outputs:**

- `nyc.graph`
- `rid2new.json`
- `new2rid.json`

### 6. Partition the road graph with KaHIP

The current NYC configuration partitions the graph into 100 initial KaHIP partitions. Later processing may split disconnected components into additional regions.

```bash
/opt/KaHIP/build/kaffpa ../datasets/nyc/nyc.graph \
                        --k 100 \
                        --preconfiguration=strong \
                        --output_filename ../datasets/nyc/tmppartition100
```

**Persistent output:** `tmppartition100`

### 7. Convert the KaHIP partition into TS-TrajGen regions

This maps roads back from KaHIP IDs to road IDs and produces the final road-to-region / region-to-road mappings used by hierarchical generation.

```bash
python ./script/process_kaffpa_res.py \
    --dataset_name nyc \
    --data_root ../datasets \
    --partition_filename tmppartition100 \
    --new2rid_filename new2rid.json \
    --adjacent_filename adjacent_list.json \
    --region2rid_filename region2rid.json \
    --rid2region_filename rid2region.json
```

**Persistent outputs:**

- `region2rid.json`
- `rid2region.json`

### 8. Build region-level adjacency relationships

This derives the region graph and records boundary roads connecting neighboring regions.

```bash
python ./script/construct_traffic_zone_relation.py \
       --dataset_name nyc \
       --data_root ../datasets \
       --rel_filename nyc.rel \
       --adjacent_filename adjacent_list.json \
       --rid2region_filename rid2region.json \
       --region2rid_filename region2rid.json \
       --region_adj_mx_filename_output region_adj_mx.npz \
       --region_adjacent_filename_output region_adjacent_list.json
```

**Persistent outputs:**

- `region_adj_mx.npz`
- `region_adjacent_list.json`

### 9. Map road-level trajectories to region-level trajectories

This converts the road-level train/test trajectories into region sequences and creates the region-level train/eval/test split used by region pretraining and region GAN training.

```bash
python -m script.map_region_traj \
    --dataset_name nyc \
    --data_root ../datasets \
    --rid2region_filename rid2region.json \
    --train_mm_filename nyc_mm_train.csv \
    --test_mm_filename nyc_mm_test.csv \
    --train_region_filename nyc_mm_region_train.csv \
    --eval_region_filename nyc_mm_region_eval.csv \
    --test_region_filename nyc_mm_region_test.csv \
    --config ./configs/ts_trajgen_nyc.yaml
```

Run this as a module (`python -m script.map_region_traj`) so imports such as `from utils.refactor_utils import load_config` resolve consistently from the repository root.

**Persistent outputs:**

- `nyc_mm_region_train.csv`
- `nyc_mm_region_eval.csv`
- `nyc_mm_region_test.csv`

### 10. Encode region-level trajectories for pretraining

This converts region trajectories into the examples expected by the region-level Function G and Function H pretraining scripts and creates `region_gps.json`.

```bash
python -m script.encode_region_traj \
       --dataset_name nyc \
       --data_root ../datasets \
       --rid2region_filename rid2region.json \
       --region2rid_filename region2rid.json \
       --rid_gps_filename rid_gps.json \
       --region_adjacent_filename region_adjacent_list.json \
       --train_region_filename nyc_mm_region_train.csv \
       --eval_region_filename nyc_mm_region_eval.csv \
       --test_region_filename nyc_mm_region_test.csv \
       --region_gps_output_filename region_gps.json \
       --train_output_filename nyc_region_pretrain_input_train.csv \
       --eval_output_filename nyc_region_pretrain_input_eval.csv \
       --test_output_filename nyc_region_pretrain_input_test.csv \
       --config ./configs/ts_trajgen_nyc.yaml
```

**Persistent outputs:**

- `region_gps.json`
- `nyc_region_pretrain_input_train.csv`
- `nyc_region_pretrain_input_eval.csv`
- `nyc_region_pretrain_input_test.csv`

### 11. Build region-level node features

This aggregates the road-level graph/node representation into region-level features required by the region GAT.

```bash
python prepare_region_feature.py \
       --dataset_name nyc \
       --device cuda:0 \
       --data_root ../datasets \
       --geo_path ../datasets/nyc/nyc.geo \
       --map_manager_cache_dir ../datasets/nyc \
       --save_folder ./save/nyc \
       --save_file_name gat_fc.pt \
       --adjacent_np_filename adjacent_mx.npz \
       --node_feature_filename node_feature.pt \
       --rid2region_filename rid2region.json \
       --region2rid_filename region2rid.json \
       --region_feature_filename region_feature.pt \
       --config ./configs/ts_trajgen_nyc.yaml
```

**Persistent output:** `region_feature.pt`

### 12. Build region-distance and road-length artifacts

This computes region-to-region distance information used by region-level candidate scoring and produces the road-length lookup used by the searcher.

For strict split isolation, the NYC YAML currently uses `data.region_distance.include_test: false`. The test filename remains explicit in the CLI for compatibility, but test trajectories should not contribute to the training-derived region-distance statistic while this setting is disabled.

```bash
python -m script.construct_region_dist \
       --dataset_name nyc \
       --data_root ../datasets \
       --geo_filename nyc.geo \
       --road_length_filename road_length.json \
       --rid2region_filename rid2region.json \
       --region_gps_filename region_gps.json \
       --train_mm_filename nyc_mm_train.csv \
       --test_mm_filename nyc_mm_test.csv \
       --processed_traj_filename nyc_traj_mm_processed.csv \
       --region_dist_filename region_count_dist.npy \
       --config ./configs/ts_trajgen_nyc.yaml
```

**Persistent outputs:**

- `road_length.json`
- `nyc_traj_mm_processed.csv`
- `region_count_dist.npy`

### 13. Pretrain region-level Function G

```bash
python pretrain_region_function_g_fc.py \
       --dataset_name nyc \
       --data_root ../datasets \
       --region2rid_filename region2rid.json \
       --train_filename nyc_region_pretrain_input_train.csv \
       --eval_filename nyc_region_pretrain_input_eval.csv \
       --test_filename nyc_region_pretrain_input_test.csv \
       --save_dir ./save/nyc \
       --save_file_name region_function_g_fc.pt \
       --temp_dir ./temp/nyc/region_function_g \
       --device cuda:0 \
       --config ./configs/ts_trajgen_nyc.yaml \
       --train
```

**Persistent output:** `region_function_g_fc.pt`

### 14. Pretrain region-level Function H / GAT

```bash
python pretrain_region_gat_fc.py \
       --dataset_name nyc \
       --data_root ../datasets \
       --region2rid_filename region2rid.json \
       --adjacent_np_filename region_adj_mx.npz \
       --node_feature_filename region_feature.pt \
       --region_dist_filename region_count_dist.npy \
       --train_filename nyc_region_pretrain_input_train.csv \
       --eval_filename nyc_region_pretrain_input_eval.csv \
       --test_filename nyc_region_pretrain_input_test.csv \
       --save_dir ./save/nyc \
       --save_file_name region_gat_fc.pt \
       --temp_dir ./temp/nyc/region_gat \
       --device cuda:0 \
       --config ./configs/ts_trajgen_nyc.yaml \
       --train
```

**Persistent output:** `region_gat_fc.pt`

### 15. Build road-level historical OD-route helper data

`od_distinct_route.json` is consumed by road GAN rollout/yaw-loss logic. Because it is a training-derived helper statistic, it is built from `nyc_mm_train.csv` only.

```bash
python -m script.generate_od_distinct_route \
       --dataset_name nyc \
       --data_root ../datasets \
       --traj_filename nyc_mm_train.csv \
       --route_column rid_list \
       --gps_filename rid_gps.json \
       --output_filename od_distinct_route.json
```

**Persistent output:** `od_distinct_route.json`

### 16. Build road-level travel-time distribution

`road_time_distribution.npy` stores hourly average road travel-time information used during search. It is derived from the training trajectories only.

```bash
python -m script.generate_time_distribution \
       --dataset_name nyc \
       --data_root ../datasets \
       --traj_filename nyc_mm_train.csv \
       --geo_filename nyc.geo \
       --output_filename road_time_distribution.npy
```

**Persistent output:** `road_time_distribution.npy`

### 17. Train the road-level GAN

This initializes the road generator from the pretrained road Function G and Function H checkpoints, then performs the road-level adversarial/reinforcement-learning stage.

```bash
python train_gan.py \
       --dataset_name nyc \
       --data_root ../datasets  \
       --exp_id 1 \
       --save_dir ./save/nyc/gan \
       --pretrain_g_file ./save/nyc/function_g_fc.pt \
       --pretrain_gat_file ./save/nyc/gat_fc.pt \
       --trajectory_filename nyc_mm_train.csv \
       --node_feature_filename node_feature.pt \
       --adjacent_np_filename adjacent_mx.npz \
       --adjacent_list_filename adjacent_list.json \
       --rid_gps_filename rid_gps.json \
       --road_length_filename road_length.json \
       --od_distinct_route_filename od_distinct_route.json \
       --road_time_dist_filename road_time_distribution.npy \
       --geo_filename nyc.geo \
       --map_manager_cache_dir ../datasets/nyc \
       --device cuda:0 \
       --config ./configs/ts_trajgen_nyc.yaml \
       --pretrain_discriminator True \
       --debug True
```

**Persistent outputs:**

- `adversarial_3_generator_1.pt`
- `adversarial_discriminator.pt`

### 18. Build region-transition frequency helper data

`region_transfer_prob.json` records which road segments are historically used when trajectories cross between neighboring regions. It is derived from the road-level training trajectories only.

```bash
python -m script.count_region_transfer \
       --dataset_name nyc \
       --data_root ../datasets \
       --rid2region_filename rid2region.json \
       --region_adjacent_filename region_adjacent_list.json \
       --output_filename region_transfer_prob.json \
       --traj_filename nyc_mm_train.csv
```

**Persistent output:** `region_transfer_prob.json`

### 19. Build region-level historical OD-route helper data

`region_od_distinct_route.json` is the region-level equivalent of the road OD-route helper and is built from `nyc_mm_region_train.csv` only.

Routes containing fewer than two valid region IDs do not form an OD route. The current helper logic also excludes same-origin/same-destination cases where the resulting route is not useful for the intended yaw-loss comparison.

```bash
python -m script.generate_od_distinct_route \
       --dataset_name nyc \
       --data_root ../datasets \
       --traj_filename nyc_mm_region_train.csv \
       --route_column region_list \
       --gps_filename region_gps.json \
       --output_filename region_od_distinct_route.json
```

**Persistent output:** `region_od_distinct_route.json`

### 20. Build region-level travel-time distribution

`region_time_distribution.npy` stores hourly average region travel-time information and is derived from the region-level training trajectories only.

```bash
python -m script.generate_time_distribution_region \
       --dataset_name nyc \
       --data_root ../datasets \
       --region2rid_filename region2rid.json \
       --train_region_filename nyc_mm_region_train.csv \
       --output_filename region_time_distribution.npy
```

**Persistent output:** `region_time_distribution.npy`

### 21. Train the region-level GAN

This initializes the region generator from the pretrained region Function G and Function H checkpoints, then performs the region-level adversarial/reinforcement-learning stage using the train-derived search/helper artifacts.

```bash
python train_region_gan.py \
       --dataset_name nyc \
       --data_root ../datasets \
       --trajectory_file nyc_mm_region_train.csv \
       --pretrain_region_function_g_file ./save/nyc/region_function_g_fc.pt \
       --pretrain_region_gat_file ./save/nyc/region_gat_fc.pt \
       --save_folder ./save/nyc/gan_region \
       --adjacent_list_file adjacent_list.json \
       --rid_gps_file rid_gps.json \
       --road_length_file road_length.json \
       --region_adjacent_list_file region_adjacent_list.json \
       --region_adj_mx_file region_adj_mx.npz \
       --region_feature_file region_feature.pt \
       --region_dist_file region_count_dist.npy \
       --region_transfer_file region_transfer_prob.json \
       --rid2region_file rid2region.json \
       --region2rid_file region2rid.json \
       --region_gps_file region_gps.json \
       --region_od_file region_od_distinct_route.json \
       --road_time_dist_file road_time_distribution.npy \
       --region_time_dist_file region_time_distribution.npy \
       --device cuda:0 \
       --config ./configs/ts_trajgen_nyc.yaml \
       --debug True \
       --pretrain_discriminator True
```

**Persistent outputs:**

- `adversarial_region_generator.pt`
- `adversarial_region_discriminator.pt`

## Generation modes

Steps 22 and 23 are two different generation/evaluation paths. They intentionally load different model checkpoints.

- **Step 22** stays close to the original authors' `our_model_generate.py` behavior and loads only the separately pretrained road/region Function G and Function H checkpoints.
- **Step 23** uses the added `our_model_generate_using_gan.py` script and loads the complete road and region `GeneratorV4` state dictionaries produced by GAN training.

For the final adversarial TS-TrajGen baseline, Step 23 is the GAN-trained generation path. Step 22 is still useful as a pretrained-only comparison and as a reference to the original repository behavior.

### 22. Generate trajectories with pretrained Function G / Function H checkpoints

The held-out road-level test trajectories provide the generation OD/start-time inputs used by the original search flow.

```bash
python our_model_generate.py \
       --dataset_name nyc \
       --data_root ../datasets \
       --true_traj_file nyc_mm_test.csv \
       --generated_trace_output_file TS_TrajGen_non_gan_generated_output.csv \
       \
       --pretrain_gen_file ./save/nyc/function_g_fc.pt \
       --pretrain_gat_file ./save/nyc/gat_fc.pt \
       --pretrain_region_gen_file ./save/nyc/region_function_g_fc.pt \
       --pretrain_region_gat_file ./save/nyc/region_gat_fc.pt \
       \
       --geo_path ../datasets/nyc/nyc.geo \
       --map_manager_cache_dir ../datasets/nyc \
       \
       --node_feature_file node_feature.pt \
       --adjacent_np_file adjacent_mx.npz \
       --region_adjacent_np_file region_adj_mx.npz \
       --region_feature_file region_feature.pt \
       \
       --region2rid_file region2rid.json \
       --adjacent_list_file adjacent_list.json \
       --rid_gps_file rid_gps.json \
       --road_length_file road_length.json \
       --region_adjacent_list_file region_adjacent_list.json \
       --region_dist_file region_count_dist.npy \
       --region_transfer_file region_transfer_prob.json \
       --rid2region_file rid2region.json \
       \
       --road_time_distribution_file road_time_distribution.npy \
       --region_time_distribution_file region_time_distribution.npy \
       --config ./configs/ts_trajgen_nyc.yaml \
       --device cuda:0
```

**Persistent output:** `TS_TrajGen_non_gan_generated_output.csv`

### 23. Generate trajectories with GAN-trained full-generator checkpoints

This is the adversarially trained generation path. It loads the complete road and region `GeneratorV4` checkpoints saved by Steps 17 and 21.

```bash
python our_model_generate_using_gan.py \
    --dataset_name nyc \
    --data_root ../datasets \
    --true_traj_file nyc_mm_test.csv \
    --generated_trace_output_file TS_TrajGen_GAN_generated_output.csv \
    --road_gan_generator_file ./save/nyc/gan/adversarial_3_generator_1.pt \
    --region_gan_generator_file ./save/nyc/gan_region/adversarial_region_generator.pt \
    --geo_path ../datasets/nyc/nyc.geo \
    --map_manager_cache_dir ../datasets/nyc \
    --node_feature_file node_feature.pt \
    --adjacent_np_file adjacent_mx.npz \
    --region_adjacent_np_file region_adj_mx.npz \
    --region_feature_file region_feature.pt \
    --region2rid_file region2rid.json \
    --rid2region_file rid2region.json \
    --adjacent_list_file adjacent_list.json \
    --rid_gps_file rid_gps.json \
    --road_length_file road_length.json \
    --region_adjacent_list_file region_adjacent_list.json \
    --region_dist_file region_count_dist.npy \
    --region_transfer_file region_transfer_prob.json \
    --road_time_distribution_file road_time_distribution.npy \
    --region_time_distribution_file region_time_distribution.npy \
    --config ./configs/ts_trajgen_nyc.yaml \
    --device cuda:0
```

**Persistent output:** `TS_TrajGen_GAN_generated_output.csv`

## Important reproducibility notes

The road network, map-matching outputs, `.geo` / `.rel` files, road IDs, region partitioning, and all downstream helper/model artifacts form one dependency chain. If the canonical road network or map-matching result changes, regenerate the dependent TS-TrajGen artifacts instead of mixing files from different network versions.

The train-derived helper files must also be regenerated whenever the training split changes. This includes at least:

- `od_distinct_route.json`
- `road_time_distribution.npy`
- `region_transfer_prob.json`
- `region_od_distinct_route.json`
- `region_time_distribution.npy`
- any region-distance artifact whose construction is configured to depend on the training trajectory split

For smoke testing, a reduced train/test subset may be substituted under the same filenames. Before a final experiment, restore the full split and regenerate every downstream artifact that depends on those trajectories.
