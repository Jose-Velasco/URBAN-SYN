# TS-TrajGen Pipeline Wrapper

This document covers the project-level wrapper used to run the adapted TS-TrajGen baseline in `baselines/gan/ts_trajgen/`.

The wrapper keeps the TS-TrajGen scripts largely intact and runs preprocessing, pretraining, GAN training, and generation in a fixed, reproducible order.

- Upstream repository: <https://github.com/WenMellors/TS-TrajGen>
- Source commit used by this baseline: `a71502d3a834f0069475ba3c71bb56f851e32a62`
- Adapted upstream source: `repo/`
- Wrapper entry point: `run_pipeline.py`
- Wrapper config: `pipeline_configs/nyc.yaml`
- Model/training config: `repo/configs/ts_trajgen_nyc.yaml`

## Configuration

The two YAML files have separate responsibilities:

- `pipeline_configs/nyc.yaml` — orchestration: paths, device, input construction, KaHIP settings, GAN runtime flags, resume flags, and generation mode.
- `repo/configs/ts_trajgen_nyc.yaml` — TS-TrajGen architecture and training hyperparameters.

Keep model hyperparameters in the model config rather than duplicating them in the wrapper config.

### Current wrapper settings

The checked-in NYC wrapper config is a **smoke-test configuration**:

```yaml
build:
  min_len: 2
  min_delta_seconds: 0.5
  train_ratio: 0.8
  interpolate_intermediate_edges: false
  max_train_trajectories: 150
  max_test_trajectories: 4

partition:
  k: 100
  preconfiguration: strong

gan:
  exp_id: 1
  road:
    debug: true
    pretrain_discriminator: true
    resume: false
  region:
    debug: true
    pretrain_discriminator: true
    resume: false

generation:
  mode: gan
```

For a full-data run, set both trajectory limits to `null` and set the road/region GAN `debug` flags to `false`. Do not reuse smoke-test helper artifacts or checkpoints for a full run.

`resume: true` is only for continuing GAN training from an existing compatible checkpoint. It does not make the wrapper itself automatically resume failed preprocessing stages.

## Required NYC inputs

Inside the container, the wrapper expects:

```text
/workspace/upstream/fmm/data/road_network/nyc.gpkg
/workspace/upstream/fmm/data/nyc_gps_points_fmm_trip_id_map.csv
/workspace/upstream/fmm/outputs/nyc_fmm_match.csv
/workspace/upstream/data/nyc_output_tabular/output/traj_cleaned.parquet
```

These files must come from the same road-network/map-matching run. Do not mix artifacts produced from different network versions.

### Trajectory IDs and split order

Map matching is performed before the TS-TrajGen train/test split. `build-inputs` reads the completed FMM output and then creates `nyc_mm_train.csv` and `nyc_mm_test.csv` using `build.train_ratio`.

The integer trajectory `id` used in the map-matched files is the FMM ID. `nyc_gps_points_fmm_trip_id_map.csv` maps it back to `trip_key`, where `trip_key` was constructed from the processed MAT dataset's `uid` and `tid` because `tid` is not globally unique by itself.

## Local Docker usage

From `URBAN-SYN/baselines/gan/ts_trajgen/`:

```bash
docker compose build
docker compose run --rm ts-trajgen
```

The image uses PyTorch 2.0.1 / CUDA 11.7 and includes KaHIP at `/opt/KaHIP/build/kaffpa`.

Inside the container, commands use `/workspace` as the baseline root.

## Running the wrapper

General form:

```bash
python /workspace/run_pipeline.py <target> \
    --config /workspace/pipeline_configs/nyc.yaml
```

Supported targets:

| Target | What it runs |
| --- | --- |
| `prepare` | Road/region preprocessing and train-derived helper statistics |
| `pretrain-road` | Road GAT/Function H and Function G pretraining |
| `prepare-region` | Region feature construction from the pretrained road representation |
| `pretrain-region` | Region Function G and GAT/Function H pretraining |
| `setup-gans` | Everything above; prepares all inputs/checkpoints needed to start GAN training |
| `train-road` | Road-level GAN training only |
| `train-region` | Region-level GAN training only |
| `train` | Both GAN training stages |
| `generate` | Generation path selected by `generation.mode` |
| `all` | Complete pipeline from preprocessing through generation |

Examples:

```bash
# Inspect the full pipeline without executing it.
python /workspace/run_pipeline.py all \
    --config /workspace/pipeline_configs/nyc.yaml \
    --dry-run

# Prepare everything needed for GAN training.
python /workspace/run_pipeline.py setup-gans \
    --config /workspace/pipeline_configs/nyc.yaml

# Train road and region GANs.
python /workspace/run_pipeline.py train \
    --config /workspace/pipeline_configs/nyc.yaml

# Generate trajectories from the configured checkpoints.
python /workspace/run_pipeline.py generate \
    --config /workspace/pipeline_configs/nyc.yaml
```

A dry run validates configured working directories and prints the exact subprocess commands, so run it inside Docker/Apptainer where `/workspace` and `/workspace/repo` exist.

## Pipeline order

The wrapper uses a fixed order:

```text
prepare (14 stages)
  ↓
pretrain-road (2)
  ↓
prepare-region (1)
  ↓
pretrain-region (2)
  ↓
train-road (1)
  ↓
train-region (1)
  ↓
generate (1 or 2, depending on generation.mode)
```

With `generation.mode: gan` or `pretrained`, `all` runs 22 stages. With `generation.mode: both`, it runs 23.

The main primitive stages are:

```text
prepare:
  build-inputs
  preprocess-road
  kahip-format
  kahip-partition
  process-kahip
  construct-region-adjacency
  map-region-trajectories
  encode-region-trajectories
  construct-region-distance
  road-od-routes
  road-time-distribution
  region-transfer
  region-od-routes
  region-time-distribution

pretrain-road:
  pretrain-road-gat
  pretrain-road-function-g

prepare-region:
  prepare-region-features

pretrain-region:
  pretrain-region-function-g
  pretrain-region-gat

train-road:
  train-road-gan

train-region:
  train-region-gan
```

`pipeline/stages.py` is the canonical source for stage construction and exact commands.

## Generation modes

`generation.mode` controls the `generate` target:

| Mode | Behavior |
| --- | --- |
| `pretrained` | Uses separately pretrained road/region Function G and Function H checkpoints |
| `gan` | Uses the adversarially trained road and region `GeneratorV4` checkpoints |
| `both` | Runs both generation paths |

For the adversarial TS-TrajGen baseline, use `gan`.

## Main outputs

Dataset/preprocessing artifacts are written primarily to:

```text
/workspace/datasets/nyc/
```

Key files include:

```text
nyc.geo
nyc.rel
nyc_mm_train.csv
nyc_mm_test.csv
nyc_mm_region_train.csv
nyc_mm_region_eval.csv
nyc_mm_region_test.csv
adjacent_list.json
region_adjacent_list.json
region2rid.json
rid2region.json
road_time_distribution.npy
region_time_distribution.npy
region_transfer_prob.json
od_distinct_route.json
region_od_distinct_route.json
TS_TrajGen_non_gan_generated_output.csv
TS_TrajGen_GAN_generated_output.csv
```

Pretraining and model checkpoints are under:

```text
/workspace/repo/save/nyc/
/workspace/repo/save/nyc/gan/
/workspace/repo/save/nyc/region_gan/
```

The configured `/workspace/outputs` path is currently not used by the generation stages; generated CSVs are written to the dataset directory.

## Data-split and leakage policy

- Train is used for model fitting and train-derived helper statistics.
- Eval/validation is used by pretraining where supported.
- Test is held out from train-derived statistics and is used as the reference input for generation/evaluation.

Train-derived artifacts include:

```text
od_distinct_route.json
road_time_distribution.npy
region_transfer_prob.json
region_od_distinct_route.json
region_time_distribution.npy
```

Regenerate these and any train-dependent region-distance artifacts whenever the training split changes. Do not select hyperparameters using the test set.

## Failure, rerun, and resume behavior

The wrapper is fail-fast: stages run sequentially, output goes to the current terminal/Slurm log, and execution stops on the first non-zero exit code. It does not provide automatic caching or dependency-based restart.

After fixing a failure, rerun the smallest appropriate target. For GAN training, the road and region scripts can resume from compatible checkpoints when their corresponding YAML `resume` flag is enabled.

## Generation caveat

Both generation paths read held-out test trajectories for origin/destination context and pass the reference road-sequence length as `default_len=len(rid_list)`. Generated length is therefore conditioned by the reference trajectory rather than being fully unconditional. Keep this limitation in mind when comparing TS-TrajGen with baselines that generate length independently.

## Reproducibility

Treat the artifacts as a dependency chain:

```text
canonical road network
  → FMM map matching
  → TS-TrajGen inputs/split
  → KaHIP and region artifacts
  → train-derived statistics
  → pretrained checkpoints
  → GAN checkpoints
  → generated trajectories
```

If the road network, FMM output, split, KaHIP settings, or important preprocessing options change, rebuild the downstream artifacts that depend on them. Do not combine files from different runs simply because their filenames match.
