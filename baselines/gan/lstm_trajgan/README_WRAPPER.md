# LSTM-TrajGAN Wrapper

This is intentionally a thin orchestration layer around the scripts that have
already been smoke-tested manually:

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

The wrapper does not reimplement LSTM-TrajGAN preprocessing, training, or
generation.

## Targets

```text
build
encode-train
encode-test
prepare       = build + encode-train + encode-test
train
generate
all           = prepare + train + generate
```

## First wrapper smoke test

From `/workspace`:

```bash
python run_pipeline.py all \
  --config pipeline_configs/nyc.yaml \
  --dry-run
```

Then:

```bash
python run_pipeline.py all \
  --config pipeline_configs/nyc.yaml
```

The provided YAML starts with the same smoke settings that already worked
manually: 100 trajectories, 2 epochs, batch size 32, and checkpoint every
epoch.

## Switching to the full NYC run

In `pipeline_configs/nyc.yaml`, change:

```yaml
build:
  max_trajectories: null

train:
  epochs: 2000
  batch_size: 256
  save_params_rate: 100
```

Before launching the full training job, run only:

```bash
python run_pipeline.py prepare --config pipeline_configs/nyc.yaml
```

and inspect the builder's full-data `max_length`, p95, and p99 trajectory
lengths. LSTM-TrajGAN pads every sequence to the global maximum, so this is the
one full-data property worth checking before submitting the expensive GPU job.

After that:

```bash
python run_pipeline.py train --config pipeline_configs/nyc.yaml
python run_pipeline.py generate --config pipeline_configs/nyc.yaml
```

These same targets are intended to be called by the Slurm `.sbatch` file later.
