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

IIII. 
___

1. **Pre-preprocesses** data ✅
    
    1.1 run b`build_tstrajgen_inputs.py` (in dev container) on your dataset: example command:
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
    **Outputs:** nyc_mm_test.csv, nyc_mm_train.csv, nyc.geo, nyc.rel
2. (**INSIDE CONTAINER ts-trajgen**) Run  `preprocess_pretrain_input.py` ✅

```bash
python -m script.preprocess_pretrain_input \
       --dataset_name nyc \
       --data_root ../datasets/ \
       --dataset_prefix nyc \
       --config ./configs/ts_trajgen_nyc.yaml
```
**Output:** adjacent_list.json, rid_gps.json, nyc_pretrain_input_eval.csv, nyc_pretrain_input_test.csv, nyc_pretrain_input_train.csv


5. (**INSIDE CONTAINER ts-trajgen**) run `pretrain_gat_fc.py` for function H ✅

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

**Outputs:** node_feature.pt, adjacent_mx.npz, gat_fc.pt, nyc_features_processed.bounds.json

6. (**INSIDE CONTAINER ts-trajgen**) run `pretrain_function_g_fc.py` for function G ✅

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

**Outputs:** function_g_fc.pt

7. (**INSIDE CONTAINER ts-trajgen**) run `process_kahip_graph_format.py` to generate KaHIP's input ✅

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

**Outputs:** nyc.graph, rid2new.json, new2rid.json

8. (**INSIDE CONTAINER ts-trajgen**) run to conduct graph partition ✅

```bash
/opt/KaHIP/build/kaffpa ../datasets/nyc/nyc.graph \
                        --k 100 \
                        --preconfiguration=strong \
                        --output_filename ../datasets/nyc/tmppartition100
```

**Outputs:** tmppartition100

9. (**INSIDE CONTAINER ts-trajgen**) to process KaHIP's output and generate regions. ✅

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

**Outputs:** region2rid.json, rid2region.json

10. (**INSIDE CONTAINER ts-trajgen**) to calculate regions' adjacent relationships.✅
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

**Outputs:** region_adj_mx.npz, region_adjacent_list.json

11. (**INSIDE CONTAINER ts-trajgen**) to map the road-level traj to region level. ✅

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
- python -m script.map_region_traj run it as a module to resolve relative imports like *from utils.refactor_utils import load_config*

**Outputs:** nyc_mm_region_eval.csv, nyc_mm_region_test.csv, nyc_mm_region_train.csv

12. (**INSIDE CONTAINER ts-trajgen**) to encode the region-level trajectories to pretrain input of models. ✅

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

**Output:** region_gps.json, nyc_region_pretrain_input_train.csv, nyc_region_pretrain_input_test.csv, nyc_region_pretrain_input_eval.csv

13. (**INSIDE CONTAINER ts-trajgen**)  to calculate region GAT node feature based on road-level node ✅

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

**Output:** region_feature.pt

14. (**INSIDE CONTAINER ts-trajgen**) to calculate gps distance between regions ✅

-  Since region_count_dist.npy becomes a learned/helper statistic used during training, using train+test can be considered mild test leakage. Maybe just try using xianshi_partA_mm_train.csv  instead of xianshi_partA_traj_mm_processed.

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

**Output:** road_length.json, nyc_traj_mm_processed.csv, region_count_dist.npy

15. (**INSIDE CONTAINER ts-trajgen**) to pretrain region-level function G. ✅

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

**Output:** region_function_g_fc.pt

16. (**INSIDE CONTAINER ts-trajgen**) to pretrain region-level function H. ✅

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

**Output:** region_gat_fc.pt

17. (**INSIDE CONTAINER ts-trajgen**) to build od_distinct_route.json required for `train_gan.py` ✅

-  Since od_distinct_route.json becomes a learned/helper statistic used during training, using train+test can be considered mild test leakage. Maybe just try using xianshi_partA_mm_train.csv instead of xianshi_partA_traj_mm_processed.

```bash
python -m script.generate_od_distinct_route \
       --dataset_name nyc \
       --data_root ../datasets \
       --traj_filename nyc_mm_train.csv \
       --route_column rid_list \
       --gps_filename rid_gps.json \
       --output_filename od_distinct_route.json
```

**Output:** od_distinct_route.json

18. (**INSIDE CONTAINER ts-trajgen**) to build road_time_distribution.npy required for `train_gan.py` ✅

-  Since od_distinct_route.json becomes a learned/helper statistic used during training, using train+test can be considered mild test leakage. Maybe just try using xianshi_partA_mm_train.csv instead of xianshi_partA_traj_mm_processed.

```bash
python -m script.generate_time_distribution \
       --dataset_name nyc \
       --data_root ../datasets \
       --traj_filename nyc_mm_train.csv \
       --geo_filename nyc.geo \
       --output_filename road_time_distribution.npy
```

**Output:** road_time_distribution.npy

19. (**INSIDE CONTAINER ts-trajgen**) to adversarial learning (road level) ✅

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

**Outputs:** adversarial_3_generator_1.pt, adversarial_discriminator.pt

20. (**INSIDE CONTAINER ts-trajgen**) to build region_transfer_prob.json required for `train_region_gan.py` ✅

-  Since region_transfer_prob.json becomes a learned/helper statistic used during training, using train+test can be considered mild test leakage. Maybe just try using xianshi_partA_mm_train.csv instead of xianshi_partA_traj_mm_processed.

```bash
python -m script.count_region_transfer \
       --dataset_name nyc \
       --data_root ../datasets \
       --rid2region_filename rid2region.json \
       --region_adjacent_filename region_adjacent_list.json \
       --output_filename region_transfer_prob.json \
       --traj_filename nyc_mm_train.csv
```

**Outputs:** region_transfer_prob.json

21. (**INSIDE CONTAINER ts-trajgen**) to build region_od_distinct_route.json required for `train_region_gan.py` ✅

-  Since region_od_distinct_route.json becomes a learned/helper statistic used during training, using train+test can be considered mild test leakage. Maybe just try using xianshi_mm_region_train.csv instead of xianshi_region_traj_mm_processed.
- Two cases:
       - 1. route with only one region has no OD pair
       - 2. origin == destination: it may not useful for yaw-loss comparison between different origin/destination routes.
       - currently the script does not allow origin == destination

```bash
python -m script.generate_od_distinct_route \
       --dataset_name nyc \
       --data_root ../datasets \
       --traj_filename nyc_mm_region_train.csv \
       --route_column region_list \
       --gps_filename region_gps.json \
       --output_filename region_od_distinct_route.json
```

**Outputs:** region_od_distinct_route.json

22. (**INSIDE CONTAINER ts-trajgen**) to build region_time_distribution.npy required for `train_region_gan.py` ✅

-  Since region_od_distinct_route.json becomes a learned/helper statistic used during training, using train+test can be considered mild test leakage. Maybe just try using xianshi_mm_region_train.csv instead of xianshi_region_traj_mm_processed.

```bash
python -m script.generate_time_distribution_region \
       --dataset_name nyc \
       --data_root ../datasets \
       --region2rid_filename region2rid.json \
       --train_region_filename nyc_mm_region_train.csv \
       --output_filename region_time_distribution.npy
```

**Outputs:** region_time_distribution.npy

23. (**INSIDE CONTAINER ts-trajgen**) to adversarial learning (region level) ✅

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

**Outputs:** adversarial_region_generator.pt, adversarial_region_discriminator.pt

---

### **NOTE:** use step 24 or 25 and most likely not 24 and 25 because **24** generates trajectories from only the pertained weight not the ones trained in GAN (If im correct based on original authors code). In contrast, **25** generated trajectories using GAN trained model weights. The original GitHub is our_model_generate.py but our_model_generate_gan.py has been added and its flow follows our_model_generate.py except our_model_generate_gan.py uses the GAN trained weights

24. (**INSIDE CONTAINER ts-trajgen**) to generate trajectories (using only the pretrained weight checkpoints) based on the OD-input from the test dataset, .e.g, xianshi_mm_test.csv. ✅
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

**Outputs:** TS_TrajGen_generated_output.csv

25. (**INSIDE CONTAINER ts-trajgen**) to generate trajectories (using only the the GAN trained checkpoints) based on the OD-input from the test dataset, .e.g, xianshi_mm_test.csv.

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
**Output:** TS_TrajGen_GAN_generated_output.csv