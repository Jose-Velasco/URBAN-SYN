# LSTM-TrajGAN Wrapper

This wrapper is intentionally a thin orchestration layer around the project adapter and the upstream LSTM-TrajGAN scripts:

```text
build_lstm_trajgan_inputs.py
        ↓
repo/data/csv2npy.py  (train)
        ↓
repo/data/csv2npy.py  (test)
        ↓
repo/train.py
        ↓
repo/predict.py
```

The wrapper does not reimplement the LSTM-TrajGAN architecture or training objective. Its job is to resolve paths/settings from YAML, enforce stage dependencies, print the concrete commands, and run them in a reproducible order.

## Targets

```text
build
encode-train
encode-test
prepare       = build + encode-train + encode-test
train
generate
all           = build + encode-train + encode-test + train + generate
```

The current configuration file is:

```text
pipeline_configs/nyc.yaml
```

## Current NYC input mode

The current YAML uses an explicit train/test pair:

```yaml
paths:
  train_source: /workspace/upstream/data/preliminary_results_drive_only_10_14_2026/nyc_gps_train.parquet
  test_source: /workspace/upstream/data/preliminary_results_drive_only_10_14_2026/nyc_gps_test.parquet
```

Therefore the wrapper uses **pre-split mode**:

```text
input_mode = pre_split
```

The adapter preserves those memberships and does **not** perform another random train/test split. `build.train_ratio` is only required when `paths.source` is used instead of the explicit `train_source`/`test_source` pair.

## Data locations

This is the main path reference for teammates.

The baseline directory is mounted at `/workspace`, while the project-wide `data/` directory is mounted read-only at `/workspace/upstream/data`.

| Data/artifact | Repository / host location | Container location |
|---|---|---|
| Current train source | `<URBAN-SYN>/data/preliminary_results_drive_only_10_14_2026/nyc_gps_train.parquet` | `/workspace/upstream/data/preliminary_results_drive_only_10_14_2026/nyc_gps_train.parquet` |
| Current test source | `<URBAN-SYN>/data/preliminary_results_drive_only_10_14_2026/nyc_gps_test.parquet` | `/workspace/upstream/data/preliminary_results_drive_only_10_14_2026/nyc_gps_test.parquet` |
| Prepared LSTM-TrajGAN data | `baselines/gan/lstm_trajgan/data/nyc_full/` | `/workspace/data/nyc_full/` |
| Training checkpoints | `baselines/gan/lstm_trajgan/outputs/lstm_trajgan_nyc_full/training_params/` | `/workspace/outputs/lstm_trajgan_nyc_full/training_params/` |
| Final generated trajectories | `baselines/gan/lstm_trajgan/outputs/lstm_trajgan_nyc_full/results/nyc_syn_traj_test.csv` | `/workspace/outputs/lstm_trajgan_nyc_full/results/nyc_syn_traj_test.csv` |
| Build/runtime output logs | `baselines/gan/lstm_trajgan/outputs/lstm_trajgan_nyc_full/logs/` | `/workspace/outputs/lstm_trajgan_nyc_full/logs/` |
| Model architecture JSON | `baselines/gan/lstm_trajgan/params/` | `/workspace/params/` |

The complete prepared-data directory is expected to look like:

```text
data/nyc_full/
├── nyc_train_latlon.csv
├── nyc_test_latlon.csv
├── nyc_dev_train_encoded_final.csv
├── nyc_dev_test_encoded_final.csv
├── final_train.npy
├── final_test.npy
├── nyc_lstm_trajgan_metadata.json
├── nyc_lstm_trajgan_category_mapping.csv
├── nyc_lstm_trajgan_label_mapping.csv
└── nyc_lstm_trajgan_tid_mapping.csv
```

The final model-output tree is approximately:

```text
outputs/lstm_trajgan_nyc_full/
├── logs/
│   └── build_lstm_trajgan_inputs_nyc_<timestamp>.log
├── training_params/
│   ├── metadata.json
│   ├── C_model_100.h5
│   ├── G_model_100.h5
│   ├── D_model_100.h5
│   ├── ...
│   ├── C_model_2000.h5
│   ├── G_model_2000.h5
│   └── D_model_2000.h5
└── results/
    └── nyc_syn_traj_test.csv
```

## Stage 1: `build`

**Script executed**

```text
build_lstm_trajgan_inputs.py
```

**Purpose**

Adapt project GPS trajectory files to the CSV representation expected by LSTM-TrajGAN while preserving the current explicit train/test split.

**Current inputs**

```text
/workspace/upstream/data/preliminary_results_drive_only_10_14_2026/nyc_gps_train.parquet
/workspace/upstream/data/preliminary_results_drive_only_10_14_2026/nyc_gps_test.parquet
```

For the current `category_mode: constant`, each source requires:

```text
uid, tid, lat, lng|lon, datetime
```

**Important transformations**

The adapter:

1. loads both explicit split files;
2. verifies that source `tid` values do not overlap across train/test;
3. validates and cleans `uid`, `tid`, coordinates, and `datetime`;
4. derives `day` (`Monday=0 ... Sunday=6`) and `hour` (`0 ... 23`);
5. assigns category `0` to every point for the current GPS+temporal baseline;
6. maps source user IDs to contiguous integer `label` values;
7. maps source trajectory IDs to contiguous integer `tid` values;
8. preserves the supplied train/test memberships;
9. sorts each split by `tid` and `datetime`;
10. calculates the global latitude/longitude centroid and trajectory-length statistics;
11. creates both absolute-coordinate and centroid-deviation CSV views;
12. writes mapping and metadata files.

**Current command resolved by the wrapper**

```bash
python /workspace/build_lstm_trajgan_inputs.py \
  --train_input_path /workspace/upstream/data/preliminary_results_drive_only_10_14_2026/nyc_gps_train.parquet \
  --test_input_path /workspace/upstream/data/preliminary_results_drive_only_10_14_2026/nyc_gps_test.parquet \
  --out_dir /workspace/data/nyc_full \
  --dataset_name nyc \
  --random_state 101 \
  --category_mode constant \
  --constant_category 0 \
  --log_dir /workspace/outputs/lstm_trajgan_nyc_full/logs
```

`--max_trajectories` is not emitted because the current YAML value is `null`.

**Outputs**

```text
/workspace/data/nyc_full/nyc_train_latlon.csv
/workspace/data/nyc_full/nyc_test_latlon.csv
/workspace/data/nyc_full/nyc_dev_train_encoded_final.csv
/workspace/data/nyc_full/nyc_dev_test_encoded_final.csv
/workspace/data/nyc_full/nyc_lstm_trajgan_metadata.json
/workspace/data/nyc_full/nyc_lstm_trajgan_category_mapping.csv
/workspace/data/nyc_full/nyc_lstm_trajgan_label_mapping.csv
/workspace/data/nyc_full/nyc_lstm_trajgan_tid_mapping.csv
```

The stage also writes a timestamped build log under:

```text
/workspace/outputs/lstm_trajgan_nyc_full/logs/
```

### Build CSV meanings

`nyc_train_latlon.csv` and `nyc_test_latlon.csv` contain:

```text
label, tid, lat, lon, day, hour, category
```

with **absolute** GPS latitude/longitude.

`nyc_dev_train_encoded_final.csv` and `nyc_dev_test_encoded_final.csv` contain the same columns, but:

```text
encoded_lat = raw_lat - global_lat_centroid
encoded_lon = raw_lon - global_lon_centroid
```

These encoded CSVs are what `csv2npy.py` consumes.

## Stage 2: `encode-train`

**Script executed**

```text
repo/data/csv2npy.py
```

**Input**

```text
/workspace/data/nyc_full/nyc_dev_train_encoded_final.csv
```

**Command**

```bash
python /workspace/repo/data/csv2npy.py \
  --load_path /workspace/data/nyc_full/nyc_dev_train_encoded_final.csv \
  --save_path /workspace/data/nyc_full/final_train.npy \
  --tid_col tid
```

**Output**

```text
/workspace/data/nyc_full/final_train.npy
```

The training NPY contains five trajectory-level arrays:

```text
lat_lon
 day
 hour
 category
 mask
```

More precisely, the logical ordering is:

```text
[lat_lon, day, hour, category, mask]
```

where day/hour/category are one-hot encoded and mask is initially one for each real trajectory point. Padding is performed later by the model training code.

## Stage 3: `encode-test`

**Script executed**

```text
repo/data/csv2npy.py
```

**Input**

```text
/workspace/data/nyc_full/nyc_dev_test_encoded_final.csv
```

**Command**

```bash
python /workspace/repo/data/csv2npy.py \
  --load_path /workspace/data/nyc_full/nyc_dev_test_encoded_final.csv \
  --save_path /workspace/data/nyc_full/final_test.npy \
  --tid_col tid \
  --generate_test_npy
```

**Output**

```text
/workspace/data/nyc_full/final_test.npy
```

The test NPY contains seven logical fields:

```text
[lat_lon, day, hour, category, mask, tid, count]
```

The additional `tid` and `count` fields are required by the prediction/reconstruction step. This addresses the undocumented test-format mismatch discussed in upstream GitHub issue #5.

## Convenience target: `prepare`

`prepare` runs the first three stages in order:

```text
build
  ↓
encode-train
  ↓
encode-test
```

Command:

```bash
python run_pipeline.py prepare \
  --config pipeline_configs/nyc.yaml
```

After `prepare`, everything required to start training is contained under:

```text
/workspace/data/nyc_full/
```

## Stage 4: `train`

**Script executed**

```text
repo/train.py
```

**Inputs**

```text
/workspace/data/nyc_full/nyc_train_latlon.csv
/workspace/data/nyc_full/nyc_test_latlon.csv
/workspace/data/nyc_full/final_train.npy
```

The raw-coordinate train/test CSVs are used to reconstruct the spatial metadata needed when building the model. `final_train.npy` provides the encoded training trajectories.

**Current command**

```bash
python /workspace/repo/train.py \
  --train_csv /workspace/data/nyc_full/nyc_train_latlon.csv \
  --test_csv /workspace/data/nyc_full/nyc_test_latlon.csv \
  --train_npy /workspace/data/nyc_full/final_train.npy \
  --output_dir /workspace/outputs/lstm_trajgan_nyc_full/training_params \
  --epochs 2000 \
  --batch_size 32 \
  --save_params_rate 100
```

No `--max_length` argument is emitted because `train.max_length` is currently `null`.

**Sequence length behavior**

When `--max_length` is omitted, the local `train.py` derives the longest trajectory across the **train and test CSVs**. It then pads all encoded training trajectories to that length using zero pre-padding.

For the current full NYC run, training reported:

```text
Derived max_length from dataset: 3113
```

**Training-loop semantics**

The active upstream-style loop performs one randomly sampled batch per counter step. Therefore:

```text
2000 "epochs" = 2000 random-batch GAN training iterations
```

It is not 2000 complete passes over the training set.

**Outputs**

Training writes model weights to:

```text
/workspace/outputs/lstm_trajgan_nyc_full/training_params/
```

With `save_params_rate: 100`:

```text
C_model_100.h5
G_model_100.h5
D_model_100.h5
...
C_model_2000.h5
G_model_2000.h5
D_model_2000.h5
metadata.json
```

`metadata.json` contains the `max_length` used to construct the model.

Instantiating the model also writes architecture JSON files relative to `/workspace`:

```text
/workspace/params/C_model.json
/workspace/params/G_model.json
/workspace/params/D_model.json
```

These JSON files describe architecture; the `.h5` files contain trained weights.

## Stage 5: `generate`

**Script executed**

```text
repo/predict.py
```

**Inputs**

```text
/workspace/data/nyc_full/nyc_train_latlon.csv
/workspace/data/nyc_full/nyc_test_latlon.csv
/workspace/data/nyc_full/final_test.npy
/workspace/data/nyc_full/nyc_dev_test_encoded_final.csv
/workspace/outputs/lstm_trajgan_nyc_full/training_params/G_model_2000.h5
```

Because `generate.checkpoint_epoch` is `null`, the wrapper resolves it to `train.epochs`, which is currently `2000`.

**Current command**

```bash
python /workspace/repo/predict.py \
  --load_checkpoint_epochs 2000 \
  --train_csv /workspace/data/nyc_full/nyc_train_latlon.csv \
  --test_csv /workspace/data/nyc_full/nyc_test_latlon.csv \
  --test_npy /workspace/data/nyc_full/final_test.npy \
  --encoded_test_csv /workspace/data/nyc_full/nyc_dev_test_encoded_final.csv \
  --generator_weights_dir /workspace/outputs/lstm_trajgan_nyc_full/training_params \
  --output_csv /workspace/outputs/lstm_trajgan_nyc_full/results/nyc_syn_traj_test.csv
```

No `--max_length` argument is emitted under the current YAML, so prediction independently derives the same maximum from the train/test CSV files.

**Generation behavior**

`predict.py`:

1. loads `final_test.npy`;
2. pre-pads the test features to `max_length`;
3. generates one random latent-noise vector per actual test trajectory;
4. loads the selected `G_model_<epoch>.h5` checkpoint;
5. generates latitude/longitude, day, hour, category, and mask outputs;
6. reconstructs each generated trajectory using the stored `tid` and trajectory count;
7. converts latitude/longitude deviations back to absolute coordinates by adding the centroid;
8. removes the mask column;
9. writes the final synthetic trajectory CSV.

If an explicit `max_length` is smaller than a real test trajectory, Keras pre-truncation keeps the final `max_length` points. The local prediction code trims the original encoded-test metadata the same way so trajectory IDs remain aligned with generated rows.

**Output**

```text
/workspace/outputs/lstm_trajgan_nyc_full/results/nyc_syn_traj_test.csv
```

Columns:

```text
label, tid, lat, lon, day, hour, category
```

This is the main LSTM-TrajGAN output to pass into the project's common evaluation pipeline.

## Convenience target: `all`

`all` executes the complete stage graph:

```text
build
  ↓
encode-train
  ↓
encode-test
  ↓
train
  ↓
generate
```

Command:

```bash
python run_pipeline.py all \
  --config pipeline_configs/nyc.yaml
```

For HPC, these primitive/convenience targets can also be invoked separately by Slurm jobs so preparation, training, and generation can have different scheduling/resource requirements.

## Dry run

Before executing a changed configuration, use:

```bash
python run_pipeline.py all \
  --config pipeline_configs/nyc.yaml \
  --dry-run
```

The dry run resolves and prints the concrete stages and commands without executing them or modifying data.

## Switching between pre-split and single-source modes

The current run uses pre-split inputs. The wrapper also supports a single combined source.

### Current pre-split mode

```yaml
paths:
  train_source: /workspace/upstream/data/.../train.parquet
  test_source: /workspace/upstream/data/.../test.parquet

build:
  train_ratio: null
```

This produces:

```text
--train_input_path ... --test_input_path ...
```

and preserves the supplied split.

### Single-source mode

Use `paths.source` instead and remove `train_source`/`test_source`:

```yaml
paths:
  source: /workspace/upstream/data/.../source.parquet

build:
  train_ratio: 0.8
```

This produces:

```text
--input_path ... --train_ratio 0.8
```

and the adapter performs a trajectory-level random split using `random_state`.

Do not configure both source modes at the same time.

## Quick teammate handoff

If someone only needs to know where to find the current LSTM-TrajGAN data:

```text
Original train/test inputs:
<URBAN-SYN>/data/preliminary_results_drive_only_10_14_2026/

LSTM-TrajGAN prepared inputs/intermediates:
baselines/gan/lstm_trajgan/data/nyc_full/

Training checkpoints:
baselines/gan/lstm_trajgan/outputs/lstm_trajgan_nyc_full/training_params/

Final synthetic trajectories:
baselines/gan/lstm_trajgan/outputs/lstm_trajgan_nyc_full/results/nyc_syn_traj_test.csv
```
