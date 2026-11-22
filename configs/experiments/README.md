# Experiment YAML Guide

An experiment YAML says **what to train, how to train it, and what to save**. Keep the road network's features and GNN architecture in the separate [graph YAML](../graph/README.md). A [study YAML](../../hpc/urban_syn/README_SLURM.md) links multiple experiment configs and can automatically run GNN pretraining before joint diffusion on HPC.

Start by copying an existing experiment YAML. **Only change the sections you need**—the examples below are not complete replacement configurations. Paths are relative to the repository root.

## 1. Run an experiment

```bash
# Check input paths and configuration, without training.
uv run python -m urban_syn.experiment all \
  --config configs/experiments/nyc_joint.yaml --validate-only

# Train, then generate if enabled.
uv run python -m urban_syn.experiment all \
  --config configs/experiments/nyc_joint.yaml
```

On HPC, use [submit.sh and Slurm](../../hpc/urban_syn/README_SLURM.md), not a long-running Python command on the login node. Available joint stages are `validate`, `prepare-graph`, `train`, `generate`, `export-eval`, and `all`.

## 2. Pick the road encoder strategy

The road GNN supplies an embedding for each road segment to the diffusion model.

| Goal | `model.road_encoder.initialization` | `trainable` | `checkpoint` |
|---|---|---|---|
| Learn from scratch | `random` | `true` | `null` |
| Use pretrained GNN, freeze its weights | `pretrained` | `false` | `null` with study handoff |
| Use pretrained GNN, update it during training | `pretrained` | `true` | `null` with study handoff |

```yaml
model:
  road_encoder:
    mode: joint            # Only supported mode
    initialization: pretrained  # random | pretrained
    trainable: false
    checkpoint: null       # Filled by the HPC study launcher
```

In a study, `pretrained_from` chooses the GNN pretraining experiment. Submitting a pretrained joint experiment with `--stage all` runs **pretraining → checkpoint handoff → joint training → generation (if enabled)**. Outside a study, set `checkpoint` to a compatible pretraining `encoder.pt`. Do not use pretraining `best.pt` here. Random initialization must be trainable.

```yaml
study:
  name: example_comparison
  experiments:
    pretrain_bpr: configs/experiments/hpc_smoke_test/nyc_link_pretrain_bpr.yaml
    joint_frozen:
      config: configs/experiments/hpc_smoke_test/joint_frozen.yaml
      pretrained_from: pretrain_bpr
```

**Compatibility:** Pretraining and joint training must use matching GNN settings and ordered node/edge feature definitions. Different numbers of roads are fine; different feature meanings or input widths are not.

## 3. Paths and model settings

| Setting | Values / what it does |
|---|---|
| `experiment.name` | Required, descriptive run name; creates the output folder |
| `experiment.seed` | Nonnegative integer; random seed |
| `paths.graph_config` | **Required:** path to the graph YAML |
| `paths.train_sequences` | **Required:** prepared training road-sequence Parquet (not raw GPS) |
| `paths.test_sequences` | Optional held-out road sequences |
| `paths.trip_id_map` | Original trip mapping, used for grouped validation |
| `paths.road_network` | Road GeoPackage; needed to reconstruct GPS coordinates at generation |
| `paths.length_distribution` | Training trajectory lengths (`.npy`); needed for generation |
| `paths.output_root` | Run folders and default MLflow data; normally `outputs/experiments` |
| `model.diffusion.hidden_dim` | Even integer ≥ 4; denoiser width |
| `model.diffusion.residual_layers` | Positive integer; denoising blocks |
| `model.diffusion.diffusion_steps` | Positive integer; noise schedule length (also affects DDPM generation time) |
| `model.diffusion.beta_start`, `beta_end` | Noise schedule values between 0 and 1; end must be ≥ start |
| `model.diffusion.similarity_chunk_size` | Positive integer; lower it if road scoring runs out of GPU memory |
| `model.conditioning.enabled` | `false` only in the current unconditioned trainer |

The path `paths.test_reference_gps` is optional and relates to a handoff to the **existing shared evaluation pipeline**; this experiment guide does not configure or run quality metrics.

## 4. Training, losses, and checkpoints

| Setting | Supported values / purpose |
|---|---|
| `training.epochs` | Positive integer; complete passes through training data |
| `training.batch_size` | Positive integer; trajectories per batch |
| `training.max_batches` | Positive integer to limit batches **per epoch**, or `null` for all batches |
| `training.grad_clip_norm` | Positive float; gradient clipping |
| `training.optimizer.name` | **`adam` / `adamw`**; use `adamw` for decoupled weight decay |
| `training.optimizer.learning_rate` | Positive float, commonly `0.001` |
| `training.optimizer.weight_decay` | Nonnegative float |
| `training.scheduler.name` | **`none` / `cosine` / `reduce_on_plateau`** |
| `training.scheduler.min_lr` | Minimum LR; `cosine` requires a scalar below starting LR |
| `training.scheduler.mode` | **`min` / `max`** for plateau; use `min` for losses |
| `training.scheduler.factor`, `patience` | Plateau LR multiplier (0–1) and non-improving epoch count |
| `training.scheduler.threshold_mode` | **`rel` / `abs`**; plateau improvement threshold interpretation |
| `training.validation.enabled`, `fraction` | Enable an internal validation split; fraction strictly between 0 and 1 |
| `training.validation.split_group_columns` | **`[uid, tid]`**; keep pieces of an original trip together |
| `training.checkpoint.monitor` | **`auto` / `train_loss` / `val_loss`**; `auto` prefers validation when enabled |
| `training.checkpoint.save_best`, `save_last` | Booleans; save best weights and/or latest resumable state |
| `training.checkpoint.every_epochs` | Positive integer; numbered checkpoint interval |
| `training.checkpoint.resume_from` | Compatible checkpoint path or `null` to start fresh |

`reduce_on_plateau` also supports `threshold`, `cooldown`, `min_lr`, and `eps`. `none` keeps a constant learning rate. `cosine` varies it across the configured epochs. `val_loss` monitoring requires validation to be enabled.

**Loss choices** under `training.losses`:

| Loss | Why use it |
|---|---|
| `noise_mse` | Teach the diffusion denoiser to predict added noise |
| `reconstruction_mse` | Encourage recovery of clean road representations |
| `hard_connectivity` | Report invalid predicted road transitions; **not differentiable** |

Each loss accepts `enabled: true/false`, `weight: <nonnegative number>`, `include_in_total: true/false`, and `every_steps: <positive integer>`. If a loss contributes to the total, `every_steps` must be `1`. At least one differentiable loss (`noise_mse` or `reconstruction_mse`) must contribute positively. Use `include_in_total: false` for a diagnostic-only hard-connectivity measurement.

**Typical combinations:**

```yaml
# Quick functionality test: bounded work, no internal holdout.
training:
  epochs: 4
  max_batches: 10
  validation: {enabled: false}
  checkpoint: {monitor: auto, save_best: true, save_last: false, every_epochs: 20, resume_from: null}
```

For a longer run, remove `max_batches` (set `null`), enable grouped validation, monitor `auto`, and save `last.pt` for resume. Keep the rest of the original YAML unchanged.

Other common combinations:

| Goal | Settings that work together |
|---|---|
| Choose checkpoints by validation | `validation.enabled: true` + `checkpoint.monitor: auto` (or `val_loss`) |
| Keep constant learning rate | `scheduler.name: none` |
| Lower LR when learning stalls | `scheduler.name: reduce_on_plateau` + `mode: min` + sensible `patience` |
| Record connectivity without using it to select checkpoints | `hard_connectivity.enabled: true` + `include_in_total: false` |
| Resume an interrupted run | `checkpoint.save_last: true`, then `resume_from: path/to/last.pt` |

## 5. Generation, runtime, and MLflow

```yaml
generation:
  enabled: true
  checkpoint: null          # Use selected joint checkpoint from current 'all' run
  checkpoint_choice: best   # best | last
  road_layer: roads
  output_name: nyc_generated_gps.parquet
  num_trajectories: 8      # Positive integer, or null for length-distribution count
  batch_size: 4
  sampler: {name: ddpm}     # Only supported sampler
```

To run `generate` without training, set `generation.checkpoint` to an **existing joint** `best.pt` or `last.pt`; do not point to `encoder.pt`.

| Setting | Supported values / purpose |
|---|---|
| `runtime.device` | **`auto` / `cpu` / `cuda`** |
| `runtime.precision` | **`fp32`** only |
| `runtime.num_workers` | Nonnegative integer; `0` disables worker subprocesses |
| `tracking.enabled` | Boolean; local output files are saved even if false |
| `tracking.provider` | **`mlflow`** only |
| `tracking.storage_scope` | **`shared` / `job`**; shared DB versus separate job DBs |
| `tracking.tracking_uri` | Explicit MLflow backend URI or `null` for automatic storage |
| `tracking.experiment_name`, `run_name`, `tags` | UI grouping, optional display name, and searchable labels |
| `tracking.logging.metrics_every_steps` | Step metric frequency |
| `tracking.logging.gnn_diagnostics_every_steps` | Gradient diagnostic cadence; `0` disables those diagnostics |
| `tracking.logging.gnn_embedding_enabled`, `gnn_embedding_every_epochs`, `gnn_embedding_sample_size` | Whether, how often, and how many fixed roads to inspect for embedding diagnostics |
| `tracking.logging.log_system_metrics` | Enable MLflow's system metrics |
| `tracking.artifacts.*` | Booleans controlling saved checkpoint, generated-file, and log copies in MLflow |

See [MLflow](../../hpc/urban_syn/README_MLFLOW.md) for viewing runs locally. Generated GPS Parquet is stored under `outputs/experiments/<experiment.name>/<run_id>/generated/`. Generation creates trajectories; quality checks remain with the shared evaluation pipeline.

## 6. Separate GNN pretraining configuration

Pretraining uses `python -m urban_syn.link_pretrain` and its **own experiment YAML**, not `model.diffusion`:

| Setting | Choices / why |
|---|---|
| `pretraining.objective` | **`bpr` / `bce`**; rank true links above negatives or classify true/false links |
| `pretraining.negatives.strategy` | **`khop` / `mixed`**; nearby non-links, or nearby + uniform non-links |
| `pretraining.negatives.k_hops` | Positive integer; directed nearby candidate search radius |
| `pretraining.negatives.amount` | Number of negatives per positive link |
| `pretraining.negatives.hard_fraction` | Fraction of nearby negatives for `mixed` |
| `pretraining.graph_split` | Train/validation/test split plus disjoint train supervision |
| `pretraining.decoder` | Directional link scorer's hidden size, layers, activation, dropout, normalization |
| `training.early_stopping` | Whether to stop after validation stalls and how long to wait |

Use the [pretraining guide](../../src/urban_syn/link_pretrain/README.md) for its full workflow. All three joint strategies use the same shared graph definition; **do not silently swap graph features between pretraining and joint runs**.
