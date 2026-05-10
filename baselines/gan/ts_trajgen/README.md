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

## Container
- GPU-enabled via `gpus: all`

### Build
```bash
docker compose build
```

### Rebuild Build
```bash
docker compose down
docker compose build
```

### Open shell
```bash
docker compose run --rm ts-trajgen
```


## Preprocess Data:
There are major steps:

I. **Pre-preprocesses** data into format TS-TrajGen `preprocess_pretrain_input.py` expects

II. Run  `preprocess_pretrain_input.py` on the *Pre-preprocesses* data

III. algin TS-TrajGen `.geo` file to have the expect columns

IIII.  (**INSIDE CONTAINER ts-trajgen**) symlink/compatible file with expected columns (hacky workaround instead of refactoring TS-TrajGen fully)


- symlink you custom dataset EX nyc.geo -> xian.geo
- or one can rename thier files to xian instead

IIIII. run `pretrain_gat_fc.py`


1. **Pre-preprocesses** data
    
    1.1 run b`build_tstrajgen_inputs.py` on your dataset: example command:
    ```bash
    uv run build_tstrajgen_inputs.py \
           --network_path ../../../fmm_scripts/data/fmm_nyc.shp \
           --fmm_match_path ../../../fmm_scripts/output/nyc_fmm_match.csv \
           --parquet_path ../../../data/nyc_output_tabular/output/traj_cleaned.parquet \
           --trip_id_map_csv ../../../fmm_scripts/data/nyc_gps_points_fmm_trip_id_map.csv \
           --out_dir ./data/nyc \
           --log_dir ./data/logs \
           --dataset_name nyc \
           --min_len 2 \
           --min_delta_seconds 0.5 \
           --train_ratio 0.8
    ```
2. (**INSIDE CONTAINER ts-trajgen**) Run  `preprocess_pretrain_input.py`

```bash
python ./script/preprocess_pretrain_input.py \
       --dataset_name nyc \
       --data_root ../data/ \
       --dataset_prefix nyc \
       --train_rate 0.9 \
       --random_encode false
    #    --max_step
```

3. CAN run inside (ts-trajgen use `python`) or outside (dev container `uv run`) `ensure_geo_feature_columns.py` 
```bash
uv run ensure_geo_feature_columns.py \
   --geo_path ./data/nyc/nyc.geo \
   --output_path ./data/nyc/nyc_features_processed.geo
```
- Ensures a TS-TrajGen .geo file has the road feature columns expected 
    by the original Xian preprocessing code (pretrain_gat_fc.py).

4. (**INSIDE CONTAINER ts-trajgen**) symlink you custom dataset EX nyc.geo -> xian.geo
```bash
cd /workspace/repo/data/Xian

ln -sf /workspace/data/nyc/nyc_features_processed.geo xian.geo
ln -sf /workspace/data/nyc/nyc.rel xian.rel
ln -sf /workspace/data/nyc/nyc_mm_train.csv xianshi_partA_mm_train.csv
ln -sf /workspace/data/nyc/nyc_mm_test.csv xianshi_partA_mm_test.csv
ln -sf /workspace/data/nyc/nyc_pretrain_input_train.csv xianshi_partA_pretrain_input_train.csv
ln -sf /workspace/data/nyc/nyc_pretrain_input_eval.csv xianshi_partA_pretrain_input_eval.csv
ln -sf /workspace/data/nyc/nyc_pretrain_input_test.csv xianshi_partA_pretrain_input_test.csv

ln -sf /workspace/data/nyc/rid_gps.json rid_gps.json
ln -sf /workspace/data/nyc/adjacent_list.json adjacent_list.json
```
- to verify it worked
`ls -l`

5. (**INSIDE CONTAINER ts-trajgen**) run `pretrain_gat_fc.py` for function H

```bash
python pretrain_gat_fc.py \
       --local True \
       --dataset_name Xian \
       --device cuda:0 \
       --debug False \
       --geo_path ./data/Xian/xian.geo \
       --map_manger_cache_dir ./data/Xian/
```

6. (**INSIDE CONTAINER ts-trajgen**) run `pretrain_function_g_fc.py` for function G

```bash
python pretrain_function_g_fc.py \
       --dataset_name Xian \
       --device cuda:0 \
       --geo_path ./data/Xian/xian.geo
```

7. (**INSIDE CONTAINER ts-trajgen**) run `process_kahip_graph_format.py` to generate KaHIP's input

```bash
python process_kahip_graph_format.py \
       --dataset_name Xian \
       --data_root ../data \
       --geo_filename xian.geo \
       --rel_filename xian.rel \
       --graph_filename xian.graph
```

8. (**INSIDE CONTAINER ts-trajgen**) run to conduct graph partition

```bash
/opt/KaHIP/build/kaffpa ./data/Xian/xian.graph \
                        --k 100 \
                        --preconfiguration=strong \
                        --output_filename ./data/Xian/tmppartition100
```

9. (**INSIDE CONTAINER ts-trajgen**) to process KaHIP's output and generate regions.

```bash
python process_kaffpa_res.py \
       --dataset_name Xian \
       --data_root ../data \
       --geo_filename xian.geo \
       --rel_filename xian.rel \
       --partition_filename tmppartition100 \
       --adjacent_filename adjacent_list.json \
       --region2rid_filename region2rid.json \
       --rid2region_filename rid2region.json
```

10. (**INSIDE CONTAINER ts-trajgen**) to calculate regions' adjacent relationships.
```bash
python construct_traffic_zone_relation.py \
       --dataset_name Xian \
       --data_root ../data \
       --rel_filename xian.rel \
       --adjacent_filename adjacent_list.json \
       --rid2region_filename rid2region.json \
       --region2rid_filename region2rid.json \
       --region_adj_mx_filename_output region_adj_mx.npz \
       --region_adjacent_filename_output region_adjacent_list.json
```

11. (**INSIDE CONTAINER ts-trajgen**) to map the road-level traj to region level.

```bash
python map_region_traj.py \
       --dataset_name Xian \
       --data_root ../data \
       --rid2region_filename rid2region.json \
       --region_adjacent_filename region_adjacent_list.json \
       --train_mm_filename xianshi_partA_mm_train.csv \
       --test_mm_filename xianshi_partA_mm_test.csv \
       --train_region_filename xianshi_mm_region_train.csv \
       --eval_region_filename xianshi_mm_region_eval.csv \
       --test_region_filename xianshi_mm_region_test.csv \
       --train_rate 0.9
```

12. (**INSIDE CONTAINER ts-trajgen**) to encode the region-level trajectories to pretrain input of models.

```bash
python encode_region_traj.py \
       --dataset_name Xian \
       --data_root ../data \
       --random_encode False \
       --rid2region_filename rid2region.json \
       --region2rid_filename region2rid.json \
       --rid_gps_filename rid_gps.json \
       --region_adjacent_filename region_adjacent_list.json \
       --train_region_filename xianshi_mm_region_train.csv \
       --eval_region_filename xianshi_mm_region_eval.csv \
       --test_region_filename xianshi_mm_region_test.csv \
       --region_gps_output_filename region_gps.json \
       --train_output_filename xianshi_region_pretrain_input_train.csv \
       --eval_output_filename xianshi_region_pretrain_input_eval.csv \
       --test_output_filename xianshi_region_pretrain_input_test.csv
```

13. (**INSIDE CONTAINER ts-trajgen**)  to calculate region GAT node feature based on road-level node

```bash
python prepare_region_feature.py \
       --dataset_name Xian \
       --device cuda:0 \
       --data_root ./data \
       --geo_path ./data/Xian/xian.geo \
       --map_manger_cache_dir ./data/Xian \
       --save_folder ./save/Xian \
       --save_file_name gat_fc.pt \
       --adjacent_np_filename adjacent_mx.npz \
       --node_feature_filename node_feature.pt \
       --rid2region_filename rid2region.json \
       --region2rid_filename region2rid.json \
       --region_feature_filename region_feature.pt
```

14. (**INSIDE CONTAINER ts-trajgen**) to calculate gps distance between regions

-  Since region_count_dist.npy becomes a learned/helper statistic used during training, using train+test can be considered mild test leakage. Maybe just try using xianshi_partA_mm_train.csv  instead of xianshi_partA_traj_mm_processed.

```bash
python construct_region_dist.py \
       --dataset_name Xian \
       --data_root ../data \
       --geo_filename xian.geo \
       --road_length_filename road_length.json \
       --rid2region_filename rid2region.json \
       --region_gps_filename region_gps.json \
       --train_mm_filename xianshi_partA_mm_train.csv \
       --test_mm_filename xianshi_partA_mm_test.csv \
       --processed_traj_filename xianshi_partA_traj_mm_processed.csv \
       --region_dist_filename region_count_dist.npy
```

15. (**INSIDE CONTAINER ts-trajgen**) to pretrain region-level function G.

```bash
python pretrain_region_function_g_fc.py \
       --dataset_name Xian \
       --data_root ./data \
       --region2rid_filename region2rid.json \
       --train_filename xianshi_region_pretrain_input_train.csv \
       --eval_filename xianshi_region_pretrain_input_eval.csv \
       --test_filename xianshi_region_pretrain_input_test.csv \
       --save_dir ./save/Xian \
       --save_file_name region_function_g_fc.pt \
       --device cuda:0 \
       --train
```

16. (**INSIDE CONTAINER ts-trajgen**) to pretrain region-level function H.

```bash
python pretrain_region_gat_fc.py \
       --dataset_name Xian \
       --data_root ./data \
       --region2rid_filename region2rid.json \
       --adjacent_np_filename region_adj_mx.npz \
       --node_feature_filename region_feature.pt \
       --region_dist_filename region_count_dist.npy \
       --train_filename xianshi_region_pretrain_input_train.csv \
       --eval_filename xianshi_region_pretrain_input_eval.csv \
       --test_filename xianshi_region_pretrain_input_test.csv \
       --save_dir ./save/Xian \
       --save_file_name region_gat_fc.pt \
       --device cuda:0 \
       --train
```

18. (**INSIDE CONTAINER ts-trajgen**) to build od_distinct_route.json required for `train_gan.py`

-  Since od_distinct_route.json becomes a learned/helper statistic used during training, using train+test can be considered mild test leakage. Maybe just try using xianshi_partA_mm_train.csv instead of xianshi_partA_traj_mm_processed.

```bash
python generate_od_distinct_route.py \
       --dataset_name Xian \
       --data_root ../data \
       --traj_filename xianshi_partA_traj_mm_processed.csv \
       --rid_gps_filename rid_gps.json \
       --output_filename od_distinct_route.json

```

19. (**INSIDE CONTAINER ts-trajgen**) to build road_time_distribution.npy required for `train_gan.py`

-  Since od_distinct_route.json becomes a learned/helper statistic used during training, using train+test can be considered mild test leakage. Maybe just try using xianshi_partA_mm_train.csv instead of xianshi_partA_traj_mm_processed.

```bash
python generate_time_distribution.py \
       --dataset_name Xian \
       --data_root ../data \
       --traj_filename xianshi_partA_traj_mm_processed.csv \
       --geo_filename xian.geo \
       --output_filename road_time_distribution.npy
```

20. (**INSIDE CONTAINER ts-trajgen**) to adversarial learning (road level?)

```bash
python train_gan.py \
       --dataset_name Xian \
       --data_root ./data \
       --exp_id 1 \
       --save_dir ./save/our_gan \
       --pretrain_g_file ./save/Xian/function_g_fc.pt \
       --pretrain_gat_file ./save/Xian/gat_fc.pt \
       --trajectory_filename xianshi_partA_mm_train.csv \
       --node_feature_filename node_feature.pt \
       --adjacent_np_filename adjacent_mx.npz \
       --adjacent_list_filename adjacent_list.json \
       --rid_gps_filename rid_gps.json \
       --road_length_filename road_length.json \
       --od_distinct_route_filename od_distinct_route.json \
       --road_time_dist_filename road_time_distribution.npy \
       --geo_filename xian.geo \
       --map_manger_cache_dir ./data/Xian/ \
       --device cuda:0 \
       --pretrain_discriminator True \
       --debug True
```

21. (**INSIDE CONTAINER ts-trajgen**) to build region_transfer_prob.json required for `train_region_gan.py`

-  Since region_transfer_prob.json becomes a learned/helper statistic used during training, using train+test can be considered mild test leakage. Maybe just try using xianshi_partA_mm_train.csv instead of xianshi_partA_traj_mm_processed.

```bash
python count_region_transfer.py \
       --dataset_name Xian \
       --data_root ../data \
       --rid2region_filename rid2region.json \
       --region_adjacent_filename region_adjacent_list.json \
       --output_filename region_transfer_prob.json \
       --traj_filename xianshi_partA_traj_mm_processed.csv

       # --traj_filename xianshi_partA_mm_train.csv
```

22. (**INSIDE CONTAINER ts-trajgen**) to build xianshi_region_traj_mm_processed.csv *"required"* for `generate_od_distinct_route.py` (**region version**) and `generate_time_distribution_region`

- *"required"*: Region-level auxiliary statistics and OD route mappings can be constructed using the training split (**xianshi_mm_region_train.csv**) to avoid data leakage instead of xianshi_region_traj_mm_processed.csv but it depends on
down stream usage
- The output file of this script is for region-level utilities that require `region_list`, such as region-level OD route generation and generate_time_distribution_region. It should not replace the road-level `xianshi_partA_traj_mm_processed.csv`, which contains `rid_list`.

```bash
python build_region_processed_traj.py \
       --dataset_name Xian \
       --data_root ../data \
       --processed_filename xianshi_region_traj_mm_processed.csv \
       --train_filename xianshi_mm_region_train.csv \
       --eval_filename xianshi_mm_region_eval.csv \
       --test_filename xianshi_mm_region_test.csv
```

23. (**INSIDE CONTAINER ts-trajgen**) to build region_od_distinct_route.json required for `train_region_gan.py`

-  Since region_od_distinct_route.json becomes a learned/helper statistic used during training, using train+test can be considered mild test leakage. Maybe just try using xianshi_mm_region_train.csv instead of xianshi_region_traj_mm_processed.
- Two cases:
       - 1. route with only one region has no OD pair
       - 2. origin == destination: it may not useful for yaw-loss comparison between different origin/destination routes.
       - currently the script does not allow origin == destination

```bash
python generate_od_distinct_route.py \
       --dataset_name Xian \
       --data_root ../data \
       --traj_filename xianshi_region_traj_mm_processed.csv \
       --route_column region_list \
       --gps_filename region_gps.json \
       --output_filename region_od_distinct_route.json
```

24. (**INSIDE CONTAINER ts-trajgen**) to build region_time_distribution.npy required for `train_region_gan.py`

-  Since region_od_distinct_route.json becomes a learned/helper statistic used during training, using train+test can be considered mild test leakage. Maybe just try using xianshi_mm_region_train.csv instead of xianshi_region_traj_mm_processed.

```bash
python generate_time_distribution_region.py \
       --dataset_name Xian \
       --data_root ../data \
       --region2rid_filename region2rid.json \
       --train_region_filename xianshi_mm_region_train.csv \
       --eval_region_filename xianshi_mm_region_eval.csv \
       --test_region_filename xianshi_mm_region_test.csv \
       --output_filename region_time_distribution.npy \
       --include_eval_test
```

25. (**INSIDE CONTAINER ts-trajgen**) to adversarial learning (region level)

```bash
python train_region_gan.py \
       --dataset_name Xian \
       --data_root ./data \
       --trajectory_file xianshi_mm_region_train.csv \
       --pretrain_region_function_g_file ./save/Xian/region_function_g_fc.pt \
       --pretrain_region_gat_file ./save/Xian/region_gat_fc.pt \
       --save_folder ./save/our_region_gan \
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
       --debug True \
       --pretrain_discriminator True
```

26. (**INSIDE CONTAINER ts-trajgen**) to generate trajectories (using only the pretrained weight checkpoints) based on the OD-input from the test dataset, .e.g, xianshi_mm_test.csv.
```bash
python our_model_generate.py \
       --dataset_name Xian \
       --data_root ./data \
       --true_traj_file xianshi_partA_mm_test.csv \
       --generated_trace_output_file TS_TrajGen_generated_output.csv \
       \
       --pretrain_gen_file ./save/Xian/function_g_fc.pt \
       --pretrain_gat_file ./save/Xian/gat_fc.pt \
       --pretrain_region_gen_file ./save/Xian/region_function_g_fc.pt \
       --pretrain_region_gat_file ./save/Xian/region_gat_fc.pt \
       \
       --geo_path ./data/Xian/xian.geo \
       --map_manager_cache_dir ./data/Xian \
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
       --device cuda:0
```

26. (**INSIDE CONTAINER ts-trajgen**) to generate trajectories (using only the the GAN trained checkpoints) based on the OD-input from the test dataset, .e.g, xianshi_mm_test.csv.

```bash
python our_model_generate_gan.py \
  --dataset_name Xian \
  --data_root ./data \
  --device cuda:0 \
  --model_config ./configs/ts_trajgen_nyc.yaml \
  --true_traj_file xianshi_partA_mm_test.csv \
  --generated_trace_output_file TS_TrajGen_GAN_generate.csv \
  --road_gan_generator_file ./save/our_gan/adversarial_3_generator_1.pt \
  --region_gan_generator_file ./save/our_region_gan/adversarial_region_generator.pt \
  --geo_path ./data/Xian/xian.geo \
  --map_manager_cache_dir ./data/Xian
```