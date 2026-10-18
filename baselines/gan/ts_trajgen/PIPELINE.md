# TS-TrajGen Pipeline Wrapper

This document describes how to run the adapted TS-TrajGen baseline through the project-level pipeline wrapper in `baselines/gan/ts_trajgen/`.

The wrapper replaces the need to manually execute the full TS-TrajGen preprocessing, pretraining, adversarial-training, and generation sequence one command at a time. It keeps the individual TS-TrajGen scripts intact and orchestrates them in a fixed, reproducible order.

The implementation is based on the original TS-TrajGen repository:

- Upstream repository: <https://github.com/WenMellors/TS-TrajGen>
- Source commit used by this baseline: `a71502d3a834f0069475ba3c71bb56f851e32a62`
- Adapted upstream source: `repo/`
- Wrapper entry point: `run_pipeline.py`
- Wrapper configuration: `pipeline_configs/nyc.yaml`
- TS-TrajGen model/training configuration: `repo/configs/ts_trajgen_nyc.yaml`

## 1. Why there are two YAML files

The project intentionally separates orchestration settings from model settings.

`pipeline_configs/nyc.yaml` controls how the pipeline is run. It contains dataset paths, runtime device, input-building options, KaHIP partitioning options, GAN runtime flags, and the generation mode.

`repo/configs/ts_trajgen_nyc.yaml` contains the adapted TS-TrajGen model and training configuration. Architecture and model hyperparameters belong there rather than in the wrapper configuration.

This separation allows the same wrapper to be used in Docker locally and in Apptainer on HPC without moving model configuration into the orchestration layer.

## 2. Expected project layout

The wrapper assumes the TS-TrajGen baseline remains inside the URBAN-SYN repository:

```text
URBAN-SYN/
├── baselines/
│   └── gan/
│       └── ts_trajgen/
│           ├── build_tstrajgen_inputs.py
│           ├── docker-compose.yml
│           ├── Dockerfile
│           ├── pipeline/
│           │   ├── config.py
│           │   ├── runner.py
│           │   ├── stage.py
│           │   └── stages.py
│           ├── pipeline_configs/
│           │   └── nyc.yaml
│           ├── repo/
│           │   ├── configs/
│           │   │   └── ts_trajgen_nyc.yaml
│           │   └── ...
│           ├── datasets/
│           ├── outputs/
│           └── run_pipeline.py
├── fmm_scripts/
│   ├── data/
│   └── outputs/
└── data/
    └── nyc_output_tabular/
        └── output/
```

The Docker/Apptainer environment exposes the baseline directory as `/workspace` and keeps the upstream FMM and MAT-Dataset artifacts under `/workspace/upstream/...`.

## 3. Required upstream NYC inputs

The current NYC wrapper configuration expects these files:

```text
/workspace/upstream/fmm/data/road_network/nyc.gpkg
/workspace/upstream/fmm/data/nyc_gps_points_fmm_trip_id_map.csv
/workspace/upstream/fmm/outputs/nyc_fmm_match.csv
/workspace/upstream/data/nyc_output_tabular/output/traj_cleaned.parquet
```

These correspond to the following host-side project paths when using the provided Docker Compose configuration:

```text
../../../fmm_scripts/data/road_network/nyc.gpkg
../../../fmm_scripts/data/nyc_gps_points_fmm_trip_id_map.csv
../../../fmm_scripts/outputs/nyc_fmm_match.csv
../../../data/nyc_output_tabular/output/traj_cleaned.parquet
```

The road network, FMM output, trip-ID mapping, and trajectory parquet must belong to the same upstream preprocessing run. Do not mix TS-TrajGen artifacts produced from different road-network or map-matching versions.

## 4. Local Docker environment

The provided Docker image is based on:

```text
pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
```

It also builds KaHIP inside the image at:

```text
/opt/KaHIP/build/kaffpa
```

The Compose service mounts the entire TS-TrajGen baseline at `/workspace`, mounts the required upstream inputs read-only, enables all available GPUs, and uses an 8 GB shared-memory allocation.

From:

```text
URBAN-SYN/baselines/gan/ts_trajgen/
```

build the image with:

```bash
docker compose build
```

Open an interactive shell with:

```bash
docker compose run --rm ts-trajgen
```

Inside the container, the wrapper is available at:

```text
/workspace/run_pipeline.py
```

## 5. Wrapper CLI

The general form is:

```bash
python /workspace/run_pipeline.py <target> \
    --config /workspace/pipeline_configs/nyc.yaml
```

Supported targets are:

```text
prepare
pretrain-road
prepare-region
pretrain-region
train
generate
all
```

The full pipeline can therefore be run with:

```bash
python /workspace/run_pipeline.py all \
    --config /workspace/pipeline_configs/nyc.yaml
```

The individual groups can be run independently:

```bash
python /workspace/run_pipeline.py prepare \
    --config /workspace/pipeline_configs/nyc.yaml

python /workspace/run_pipeline.py pretrain-road \
    --config /workspace/pipeline_configs/nyc.yaml

python /workspace/run_pipeline.py prepare-region \
    --config /workspace/pipeline_configs/nyc.yaml

python /workspace/run_pipeline.py pretrain-region \
    --config /workspace/pipeline_configs/nyc.yaml

python /workspace/run_pipeline.py train \
    --config /workspace/pipeline_configs/nyc.yaml

python /workspace/run_pipeline.py generate \
    --config /workspace/pipeline_configs/nyc.yaml
```

### Dry run

Use `--dry-run` to inspect the selected stages, working directories, and exact subprocess commands without executing them:

```bash
python /workspace/run_pipeline.py all \
    --config /workspace/pipeline_configs/nyc.yaml \
    --dry-run
```

A dry run still validates each stage's configured working directory, so it should be executed inside the Docker/Apptainer environment where `/workspace` and `/workspace/repo` exist.

## 6. Pipeline execution order

The wrapper intentionally uses a fixed pipeline order rather than a dependency solver:

```text
prepare
  ↓
pretrain-road
  ↓
prepare-region
  ↓
pretrain-region
  ↓
train
  ↓
generate
```

With `generation.mode: gan`, `all` currently runs 22 stages. With `generation.mode: both`, both generation implementations are run, for 23 total stages.

### `prepare`

The `prepare` group currently runs 14 stages:

| Stage | Purpose |
| --- | --- |
| `build-inputs` | Build TS-TrajGen `.geo`/`.rel` files and road-level train/test map-matched trajectories. |
| `preprocess-road` | Build road adjacency/GPS artifacts and road-level pretraining inputs. |
| `kahip-format` | Convert the road graph to KaHIP format. |
| `kahip-partition` | Partition the road graph with KaHIP. |
| `process-kahip` | Convert KaHIP partitions into connected TS-TrajGen regions. |
| `construct-region-adjacency` | Build region adjacency and region boundary-road lookup data. |
| `map-region-trajectories` | Convert road-level trajectories to region-level train/eval/test trajectories. |
| `encode-region-trajectories` | Encode region trajectories for region-model pretraining and build region GPS lookup data. |
| `construct-region-distance` | Build road-length and train-derived region-distance artifacts. |
| `road-od-routes` | Build train-only road-level OD distinct-route history. |
| `road-time-distribution` | Build the train-only road travel-time distribution. |
| `region-transfer` | Build train-only region-transition probabilities. |
| `region-od-routes` | Build train-only region-level OD distinct-route history. |
| `region-time-distribution` | Build the train-only region travel-time distribution. |

### `pretrain-road`

| Stage | Purpose |
| --- | --- |
| `pretrain-road-gat` | Pretrain road-level Function H / GAT and produce road graph representations. |
| `pretrain-road-function-g` | Pretrain road-level Function G. |

### `prepare-region`

| Stage | Purpose |
| --- | --- |
| `prepare-region-features` | Build region node features from the pretrained road GAT representation. |

This group must run after `pretrain-road` because region feature construction depends on the learned road representation.

### `pretrain-region`

| Stage | Purpose |
| --- | --- |
| `pretrain-region-function-g` | Pretrain region-level Function G. |
| `pretrain-region-gat` | Pretrain region-level Function H / GAT. |

### `train`

| Stage | Purpose |
| --- | --- |
| `train-road-gan` | Run road-level adversarial TS-TrajGen training. |
| `train-region-gan` | Run region-level adversarial TS-TrajGen training. |

### `generate`

Generation stages are selected by `generation.mode` in the wrapper YAML.

| Mode | Stage | Checkpoints used |
| --- | --- | --- |
| `pretrained` | `generate-pretrained` | Separately pretrained road/region Function G and Function H checkpoints. |
| `gan` | `generate-gan` | Full road-level and region-level GAN-trained `GeneratorV4` checkpoints. |
| `both` | both stages | Runs both generation implementations. |

For the final adversarial TS-TrajGen baseline, `gan` is the intended GAN-trained generation path. The pretrained path is retained as an original-style comparison/reference path.

## 7. Wrapper behavior on failures and reruns

The wrapper is deliberately simple and fail-fast:

- stages run sequentially in the order defined in `pipeline/stages.py`;
- each command is executed directly with `shell=False`;
- stdout and stderr are inherited by the current terminal or Slurm log;
- execution stops immediately when a stage returns a non-zero exit code;
- there is currently no automatic cache, resume, dependency solver, or artifact fingerprinting layer.

If a run fails, fix the cause and rerun the failed group or an earlier required group. For example, if region pretraining fails after road pretraining and region feature preparation already succeeded, rerun:

```bash
python /workspace/run_pipeline.py pretrain-region \
    --config /workspace/pipeline_configs/nyc.yaml
```

Use `all` when a clean end-to-end reproduction is desired.

## 8. Smoke test versus full experiment

The checked-in NYC wrapper configuration is currently set up for a small smoke test:

```yaml
build:
  max_train_trajectories: 200
  max_test_trajectories: 4

gan:
  debug: true
```

This is intended to validate the full pipeline quickly.

For a full experiment, change the trajectory limits to `null` and disable GAN debug mode:

```yaml
build:
  max_train_trajectories: null
  max_test_trajectories: null

gan:
  debug: false
```

After changing from a smoke subset to the full dataset, regenerate downstream artifacts rather than reusing helper files or checkpoints produced from the smoke split.

## 9. Important wrapper configuration fields

The current NYC wrapper configuration contains the following orchestration settings:

```yaml
dataset:
  name: nyc

runtime:
  device: cuda:0

build:
  min_len: 2
  min_delta_seconds: 0.5
  train_ratio: 0.8
  interpolate_intermediate_edges: false
  max_train_trajectories: 200
  max_test_trajectories: 4

partition:
  k: 100
  preconfiguration: strong

gan:
  exp_id: 1
  debug: true
  pretrain_discriminator: true

generation:
  mode: gan
```

`build.interpolate_intermediate_edges: false` preserves the current project policy of using the map-matched GPS-anchor road sequence without inserting intermediate road edges between anchors during TS-TrajGen input construction.

The `paths` section uses container paths so that the same wrapper configuration can be used under Docker and Apptainer as long as the host directories are mounted to the same locations.

## 10. Main artifact locations

For dataset `nyc`, preprocessing and generated trajectory artifacts are stored primarily under:

```text
/workspace/datasets/nyc/
```

Important examples include:

```text
nyc.geo
nyc.rel
nyc_mm_train.csv
nyc_mm_test.csv
adjacent_list.json
rid_gps.json
nyc_pretrain_input_train.csv
nyc_pretrain_input_eval.csv
nyc_pretrain_input_test.csv
region2rid.json
rid2region.json
region_adj_mx.npz
region_adjacent_list.json
nyc_mm_region_train.csv
nyc_mm_region_eval.csv
nyc_mm_region_test.csv
nyc_region_pretrain_input_train.csv
nyc_region_pretrain_input_eval.csv
nyc_region_pretrain_input_test.csv
road_length.json
region_count_dist.npy
od_distinct_route.json
road_time_distribution.npy
region_transfer_prob.json
region_od_distinct_route.json
region_time_distribution.npy
TS_TrajGen_non_gan_generated_output.csv
TS_TrajGen_GAN_generated_output.csv
```

Pretraining/model checkpoints are stored under:

```text
/workspace/repo/save/nyc/
```

with road-level GAN checkpoints under:

```text
/workspace/repo/save/nyc/gan/
```

and region-level GAN checkpoints under:

```text
/workspace/repo/save/nyc/region_gan/
```

The wrapper configuration currently defines `/workspace/outputs`, but the current stage implementations do not route the generated trajectory CSVs there; generation writes them into the dataset directory.

## 11. Data-split and leakage policy

The pipeline keeps train, evaluation/validation, and test responsibilities separate.

- **Train** is used for model fitting and for training-derived helper statistics.
- **Eval/validation** is used by the pretraining workflow where supported.
- **Test** is kept out of training-derived statistics and is used as the held-out reference input for generation/evaluation.

The following helper artifacts are intentionally constructed from the training split:

```text
od_distinct_route.json
road_time_distribution.npy
region_transfer_prob.json
region_od_distinct_route.json
region_time_distribution.npy
```

Any train-dependent region-distance/helper artifact must also be regenerated when the training split changes.

Do not use test-set results to select model hyperparameters.

## 12. Generation evaluation caveat

Both current generation implementations read the held-out test trajectories and use their origin/destination context. They also pass the reference road-sequence length as `default_len=len(rid_list)` to the search/generation flow.

Therefore, generated trajectory length is not fully unconditional. This should be treated as an evaluation limitation when comparing TS-TrajGen with baselines that generate trajectory length independently.

## 15. Reproducibility rules

Treat the following as one dependency chain:

```text
canonical road network
    ↓
FMM map matching
    ↓
TS-TrajGen .geo/.rel + trajectory inputs
    ↓
KaHIP/region artifacts
    ↓
train-derived helper statistics
    ↓
pretrained checkpoints
    ↓
GAN checkpoints
    ↓
generated trajectories
```

If the canonical road network, FMM results, train/test split, KaHIP partitioning, or important preprocessing settings change, rebuild the downstream artifacts that depend on them. Avoid mixing artifacts from different runs simply because their filenames match.

For exact low-level commands used by every stage, run the wrapper with `--dry-run`. `pipeline/stages.py` is the canonical implementation of stage construction and ordering.
