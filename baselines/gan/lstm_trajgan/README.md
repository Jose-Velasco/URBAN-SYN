# Baseline: LSTM-TrajGAN

This directory contains the project integration of **LSTM-TrajGAN**, based on the public implementation:

- Upstream repository: `https://github.com/GeoDS/LSTM-TrajGAN`
- Source commit/tag used by this project: `c66e1fef1a169b2c21ab22287bae3d3f06932990`
- Local upstream checkout: `baselines/gan/lstm_trajgan/repo/`

The goal of this integration is to run the published baseline on the project's NYC trajectory data while keeping the original model architecture and training procedure as intact as practical. Project-owned code is used to adapt the NYC data into the input representation expected by LSTM-TrajGAN and to orchestrate preprocessing, training, and generation.

For the detailed wrapper stage graph and the exact input/output files for every stage, see [`README_WRAPPER.md`](README_WRAPPER.md).

## Current project adaptation

The current NYC experiment uses GPS trajectories plus temporal information. The upstream model still contains its original 10-way POI/category branch, but this branch is intentionally held constant for this run:

```text
category_mode: constant
constant_category: 0
```

Therefore, this run should be described as a **GPS + temporal LSTM-TrajGAN baseline with the POI/category feature held constant**, rather than as an exact reproduction of the original semantic Foursquare experiment.

The current pipeline also uses an already-created train/test split rather than creating a new random split inside the LSTM-TrajGAN adapter.

## Current NYC configuration

The active configuration is `pipeline_configs/nyc.yaml`:

```yaml
paths:
  data: /workspace/data/nyc_full
  outputs: /workspace/outputs/lstm_trajgan_nyc_full
  train_source: /workspace/upstream/data/preliminary_results_drive_only_10_14_2026/nyc_gps_train.parquet
  test_source: /workspace/upstream/data/preliminary_results_drive_only_10_14_2026/nyc_gps_test.parquet

build:
  train_ratio: null
  random_state: 101
  category_mode: constant
  constant_category: 0
  max_trajectories: null

train:
  epochs: 2000
  batch_size: 32
  save_params_rate: 100
  max_length: null

generate:
  checkpoint_epoch: null
```

Important behavior:

- `train_source` and `test_source` are both configured, so the adapter runs in **pre-split mode**.
- `train_ratio` is not used in pre-split mode.
- `max_trajectories: null` means the full supplied split is used.
- `max_length: null` means `train.py` and `predict.py` derive the maximum trajectory length from the train and test CSV files.
- `checkpoint_epoch: null` resolves to `train.epochs`, so the current generation stage loads `G_model_2000.h5`.
- Although the upstream code calls the training counter an "epoch", its active training loop performs **one randomly sampled batch update per iteration**, not a complete pass over the full dataset.

## Where the data files are

There are two important mount/location conventions.

The entire project data directory is mounted read-only into the container:

```text
Host / repository:
<URBAN-SYN>/data/

Docker / Apptainer:
/workspace/upstream/data/
```

The LSTM-TrajGAN baseline directory itself is mounted as `/workspace`, so generated baseline-specific data and outputs map directly back to this repository directory:

| Purpose | Repository / host path | Container path |
|---|---|---|
| Source project data | `<URBAN-SYN>/data/` | `/workspace/upstream/data/` |
| LSTM-TrajGAN prepared data | `baselines/gan/lstm_trajgan/data/nyc_full/` | `/workspace/data/nyc_full/` |
| Checkpoints/results/logs | `baselines/gan/lstm_trajgan/outputs/lstm_trajgan_nyc_full/` | `/workspace/outputs/lstm_trajgan_nyc_full/` |
| Model architecture JSON files | `baselines/gan/lstm_trajgan/params/` | `/workspace/params/` |

### Current source inputs

The current fixed NYC split is:

```text
<URBAN-SYN>/data/preliminary_results_drive_only_10_14_2026/
├── nyc_gps_train.parquet
└── nyc_gps_test.parquet
```

Inside Docker/Apptainer these are:

```text
/workspace/upstream/data/preliminary_results_drive_only_10_14_2026/
├── nyc_gps_train.parquet
└── nyc_gps_test.parquet
```

For the default `constant` category mode, source files must contain:

```text
uid
tid
lat
lng OR lon
datetime
```

The optional `top10` semantic mode additionally requires a `category` column.

### Prepared data produced by the pipeline

All preparation artifacts are written to:

```text
baselines/gan/lstm_trajgan/data/nyc_full/
```

or, inside the container:

```text
/workspace/data/nyc_full/
```

The directory contains:

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

The four CSVs produced by the adapter use these model-facing columns:

```text
label, tid, lat, lon, day, hour, category
```

`nyc_train_latlon.csv` and `nyc_test_latlon.csv` contain absolute GPS coordinates. The two `*_encoded_final.csv` files use latitude/longitude **deviations from the global centroid** used by this LSTM-TrajGAN adaptation.

The mapping files preserve the relationship between original identifiers/categories and the integer IDs used by the baseline:

```text
nyc_lstm_trajgan_label_mapping.csv
    source_uid -> label

nyc_lstm_trajgan_tid_mapping.csv
    source_tid -> tid

nyc_lstm_trajgan_category_mapping.csv
    category -> category_id (+ row count)
```

The build metadata JSON records the source paths, input mode, split summary, trajectory-length statistics, normalization values, mapping/output locations, and build timings.

### Training outputs

Training artifacts are written to:

```text
baselines/gan/lstm_trajgan/outputs/lstm_trajgan_nyc_full/training_params/
```

or:

```text
/workspace/outputs/lstm_trajgan_nyc_full/training_params/
```

With the current `save_params_rate: 100`, training saves:

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

`metadata.json` stores the `max_length` used to construct the model.

The model constructor also writes architecture JSON files in:

```text
params/
├── C_model.json
├── G_model.json
└── D_model.json
```

These are model architecture descriptions, not trained weights.

### Generated synthetic trajectories

The final generation output is:

```text
baselines/gan/lstm_trajgan/outputs/lstm_trajgan_nyc_full/results/nyc_syn_traj_test.csv
```

or inside the container:

```text
/workspace/outputs/lstm_trajgan_nyc_full/results/nyc_syn_traj_test.csv
```

Its columns are:

```text
label, tid, lat, lon, day, hour, category
```

The generated data preserve the test trajectory IDs/labels used for reconstruction while replacing the generated trajectory attributes with the generator outputs.

### Logs

Builder logs are written under:

```text
outputs/lstm_trajgan_nyc_full/logs/
```

For example:

```text
build_lstm_trajgan_inputs_nyc_<timestamp>.log
```

HPC/Slurm runtime logs are maintained separately by the HPC scripts.

## Pipeline overview

Use the project wrapper instead of manually calling each upstream script:

```text
nyc_gps_train.parquet + nyc_gps_test.parquet
                    ↓
                  build
                    ↓
train/test raw CSVs + encoded CSVs + mappings + metadata
                    ↓
              encode-train
                    ↓
              final_train.npy
                    ↓
               encode-test
                    ↓
               final_test.npy
                    ↓
                   train
                    ↓
        periodic C/G/D model checkpoints
                    ↓
                 generate
                    ↓
          nyc_syn_traj_test.csv
```

The wrapper targets are:

```text
build
encode-train
encode-test
prepare       = build + encode-train + encode-test
train
generate
all           = prepare + train + generate
```

From the container workspace:

```bash
python run_pipeline.py all \
  --config pipeline_configs/nyc.yaml \
  --dry-run
```

A dry run prints the resolved commands without executing them.

To execute all stages locally/in the container:

```bash
python run_pipeline.py all \
  --config pipeline_configs/nyc.yaml
```

The same primitive targets can be invoked separately:

```bash
python run_pipeline.py prepare --config pipeline_configs/nyc.yaml
python run_pipeline.py train --config pipeline_configs/nyc.yaml
python run_pipeline.py generate --config pipeline_configs/nyc.yaml
```

## Manual pipeline execution (without the wrapper)

The wrapper is the recommended way to run the baseline, but each underlying script can also be called directly. This is useful for debugging, reproducing an individual stage, or understanding exactly which files flow between steps.

Run these commands from `/workspace` inside the Docker/Apptainer environment.

### Step 0: Build LSTM-TrajGAN CSV inputs

**Script**

```text
build_lstm_trajgan_inputs.py
```

**Inputs**

```text
/workspace/upstream/data/preliminary_results_drive_only_10_14_2026/nyc_gps_train.parquet
/workspace/upstream/data/preliminary_results_drive_only_10_14_2026/nyc_gps_test.parquet
```

**Command**

```bash
python build_lstm_trajgan_inputs.py \
  --train_input_path /workspace/upstream/data/preliminary_results_drive_only_10_14_2026/nyc_gps_train.parquet \
  --test_input_path /workspace/upstream/data/preliminary_results_drive_only_10_14_2026/nyc_gps_test.parquet \
  --out_dir /workspace/data/nyc_full \
  --dataset_name nyc \
  --random_state 101 \
  --category_mode constant \
  --constant_category 0 \
  --log_dir /workspace/outputs/lstm_trajgan_nyc_full/logs
```

Because the current experiment uses explicit train/test source files, no `--train_ratio` is supplied. `--max_trajectories` is also omitted for the full run.

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
/workspace/outputs/lstm_trajgan_nyc_full/logs/build_lstm_trajgan_inputs_nyc_<timestamp>.log
```

The `*_latlon.csv` files contain absolute GPS coordinates. The `*_encoded_final.csv` files contain latitude/longitude deviations from the global centroid and are the files passed to `csv2npy.py`.

### Step 1: Encode the training CSV to `final_train.npy`

**Script**

```text
repo/data/csv2npy.py
```

**Input**

```text
/workspace/data/nyc_full/nyc_dev_train_encoded_final.csv
```

**Command**

```bash
python repo/data/csv2npy.py \
  --load_path /workspace/data/nyc_full/nyc_dev_train_encoded_final.csv \
  --save_path /workspace/data/nyc_full/final_train.npy \
  --tid_col tid
```

**Output**

```text
/workspace/data/nyc_full/final_train.npy
```

The training NPY contains the logical fields:

```text
[lat_lon, day, hour, category, mask]
```

### Step 2: Encode the test CSV to `final_test.npy`

**Script**

```text
repo/data/csv2npy.py
```

**Input**

```text
/workspace/data/nyc_full/nyc_dev_test_encoded_final.csv
```

**Command**

```bash
python repo/data/csv2npy.py \
  --load_path /workspace/data/nyc_full/nyc_dev_test_encoded_final.csv \
  --save_path /workspace/data/nyc_full/final_test.npy \
  --tid_col tid \
  --generate_test_npy
```

**Output**

```text
/workspace/data/nyc_full/final_test.npy
```

The test NPY contains the additional trajectory reconstruction fields required by `predict.py`:

```text
[lat_lon, day, hour, category, mask, tid, count]
```

### Step 3: Train LSTM-TrajGAN

**Script**

```text
repo/train.py
```

**Inputs**

```text
/workspace/data/nyc_full/nyc_train_latlon.csv
/workspace/data/nyc_full/nyc_test_latlon.csv
/workspace/data/nyc_full/final_train.npy
```

**Command**

```bash
python repo/train.py \
  --train_csv /workspace/data/nyc_full/nyc_train_latlon.csv \
  --test_csv /workspace/data/nyc_full/nyc_test_latlon.csv \
  --train_npy /workspace/data/nyc_full/final_train.npy \
  --output_dir /workspace/outputs/lstm_trajgan_nyc_full/training_params \
  --epochs 2000 \
  --batch_size 32 \
  --save_params_rate 100
```

No `--max_length` is supplied for the current full run, so the local training code derives it from the train/test CSV files. The current run derives `max_length=3113`.

**Outputs**

```text
/workspace/outputs/lstm_trajgan_nyc_full/training_params/
├── C_model_100.h5
├── G_model_100.h5
├── D_model_100.h5
├── ...
├── C_model_2000.h5
├── G_model_2000.h5
├── D_model_2000.h5
└── metadata.json
```

The model also writes architecture descriptions to:

```text
/workspace/params/C_model.json
/workspace/params/G_model.json
/workspace/params/D_model.json
```

### Step 4: Generate synthetic trajectories

**Script**

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

**Command**

```bash
python repo/predict.py \
  --load_checkpoint_epochs 2000 \
  --train_csv /workspace/data/nyc_full/nyc_train_latlon.csv \
  --test_csv /workspace/data/nyc_full/nyc_test_latlon.csv \
  --test_npy /workspace/data/nyc_full/final_test.npy \
  --encoded_test_csv /workspace/data/nyc_full/nyc_dev_test_encoded_final.csv \
  --generator_weights_dir /workspace/outputs/lstm_trajgan_nyc_full/training_params \
  --output_csv /workspace/outputs/lstm_trajgan_nyc_full/results/nyc_syn_traj_test.csv
```

No `--max_length` is supplied under the current configuration, so prediction derives the sequence length from the same train/test CSV inputs.

**Output**

```text
/workspace/outputs/lstm_trajgan_nyc_full/results/nyc_syn_traj_test.csv
```

Final columns:

```text
label, tid, lat, lon, day, hour, category
```

This CSV is the main synthetic trajectory output used by the project's shared evaluation pipeline.

## Container

The project container uses:

```dockerfile
FROM tensorflow/tensorflow:1.13.1-gpu-py3
```

The Dockerfile additionally installs:

```text
Keras==2.2.4
geohash2==1.1
pyarrow==0.12.0
pandas            (unpinned in the project container)
scikit-learn      (unpinned in the project container)
```

The exact versions originally documented by the upstream repository should therefore be distinguished from the versions actually installed in this project container.

Build the container:

```bash
docker compose build
```

Rebuild:

```bash
docker compose down
docker compose build
```

Open a shell:

```bash
docker compose run --rm lstm_trajgan
```

The compose file mounts the entire project `data/` directory read-only at `/workspace/upstream/data`, while the baseline directory itself is mounted read/write at `/workspace`.

## Important upstream preprocessing issue and local patches

The upstream repository does not clearly document how `final_test.npy` was created. GitHub issue #5 identifies that prediction requires two additional test fields:

```text
tid
count
```

The local `repo/data/csv2npy.py` therefore supports:

```bash
--generate_test_npy
```

which creates the seven-field test representation required by `predict.py`.

Other project compatibility/adaptation changes currently include:

- configurable CLI paths for training and prediction;
- configurable checkpoint/output directories;
- optional `max_length`, with dynamic derivation when omitted;
- dynamic prediction noise size based on the actual number of test trajectories rather than the original hard-coded test-set size;
- prediction/reconstruction alignment when a configured `max_length` truncates trajectories;
- Python 3.5-compatible path/JSON handling in the legacy model code;
- creation of output directories before writing artifacts;
- project-owned `build_lstm_trajgan_inputs.py` for adapting project trajectory data without replacing the upstream `csv2npy.py`, `train.py`, or `predict.py` stages.

## Sequence length behavior

LSTM-TrajGAN pads all trajectories to one common maximum sequence length using Keras `pad_sequences(..., padding='pre')`.

When `train.max_length` is `null`, the current local `train.py` and `predict.py` derive the maximum from **both** `nyc_train_latlon.csv` and `nyc_test_latlon.csv`.

If a manual `max_length` is supplied and a trajectory is longer than it, Keras uses its default `truncating='pre'`, so the final `max_length` points are kept. The local prediction code applies the same effective-length rule to the encoded test metadata so generated rows stay aligned with their trajectory IDs.

For the current uncapped NYC run, the derived maximum observed during training is **3113**, so the test trajectories fit without the earlier 602-point experimental cap.
