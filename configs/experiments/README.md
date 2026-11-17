# Experiment YAML configuration guide

This guide describes the settings supported by the **URBAN-SYN joint GNN + Diff-RNTraj experiment runner**. It is intended for teammates configuring training, generation, checkpointing, and experiment tracking without needing to read the Python implementation.

Experiment files live in `configs/experiments/`, for example `nyc_joint.yaml`. They reference a separate graph configuration such as `configs/graph/nyc_graph.yaml`. The graph configuration owns the road features and GNN architecture; this file owns the diffusion model and the experiment itself. See [the graph configuration guide](../graph/README.md).

## 1. Using an experiment YAML

From the repository root, with the project environment available:

```bash
# Preview the planned stages.
uv run python -m urban_syn.experiment all \
  --config configs/experiments/nyc_joint.yaml --dry-run

# Check source files without constructing the graph or training.
uv run python -m urban_syn.experiment all \
  --config configs/experiments/nyc_joint.yaml --validate-only

# Prepare the road graph if needed, train, then generate if enabled.
uv run python -m urban_syn.experiment all \
  --config configs/experiments/nyc_joint.yaml

# Generate using a previously saved checkpoint, without retraining.
uv run python -m urban_syn.experiment generate \
  --config configs/experiments/nyc_joint.yaml
```

The available targets are `validate`, `prepare-graph`, `train`, `generate`, `export-eval`, and `all`. The `all` target includes generation when `generation.enabled: true` and the evaluator handoff when a test reference is configured. Standalone `generate` requires `generation.checkpoint` to point to an existing compatible checkpoint.

On HPC, use `bash hpc/urban_syn/submit.sh all configs/experiments/nyc_joint.yaml` instead of running the Python command on a login node. Slurm CPU, GPU, memory, and time requests belong to the submission script or `SBATCH_*` overrides, not to this YAML.

**Path convention:** Relative paths inside the experiment YAML are resolved from the **repository root**, not the YAML file's directory. The HPC container binds the repository at `/workspace`. Run from the project root or explicitly set `--project-root` when supported by your command.

## 2. Complete example

The example below illustrates every current experiment-level attribute. Values are examples, not recommended hyperparameters for every research run.

```yaml
experiment:
  name: nyc_joint_example            # Label used for the output folder
  seed: 101                          # Reproducibility seed

paths:
  graph_config: configs/graph/nyc_graph.yaml
  train_sequences: fmm_scripts/outputs/urban_syn_ready/nyc_15s_enriched_moves_train_prepared_road_sequences.parquet
  test_sequences: fmm_scripts/outputs/urban_syn_ready/nyc_15s_enriched_moves_test_prepared_road_sequences.parquet
  trip_id_map: fmm_scripts/data/nyc_15s_fmm_trip_id_map.csv
  road_network: fmm_scripts/data/road_network/nyc.gpkg
  length_distribution: fmm_scripts/outputs/urban_syn_ready/nyc_15s_enriched_moves_train_length_distribution.npy
  test_reference_gps: null          # Add a reference GPS path for evaluator handoff
  output_root: outputs/experiments

model:
  diffusion:
    hidden_dim: 128                 # Even integer, at least 4
    residual_layers: 10
    diffusion_steps: 500
    beta_start: 0.0001
    beta_end: 0.02
    similarity_chunk_size: 128
  road_encoder:
    mode: joint                     # Only implemented mode
  conditioning:
    enabled: false                  # true is not implemented yet

training:
  epochs: 30
  batch_size: 32
  max_batches: null                # null means no per-epoch batch limit
  grad_clip_norm: 1.0
  optimizer:
    name: adamw                     # adam | adamw
    learning_rate: 0.001
    weight_decay: 0.01
  scheduler:
    name: reduce_on_plateau        # none | cosine | reduce_on_plateau
    mode: min                      # min | max
    factor: 0.1                    # LR multiplier on plateau
    patience: 5                    # Non-improving epochs tolerated
    threshold: 0.0001             # Minimum meaningful improvement
    threshold_mode: rel           # rel | abs
    cooldown: 0                   # Cooldown epochs after a reduction
    min_lr: 0.0                   # Scalar or per-optimizer-group list (plateau only)
    eps: 1.0e-8                  # Ignore very small LR changes
  losses:
    noise_mse:
      enabled: true
      weight: 1.0
      include_in_total: true
      every_steps: 1
    reconstruction_mse:
      enabled: true
      weight: 1.0
      include_in_total: true
      every_steps: 1
    hard_connectivity:
      enabled: true
      weight: 1.0
      include_in_total: true
      every_steps: 1
  validation:
    enabled: false
    fraction: 0.1
    split_group_columns: [uid, tid]
  checkpoint:
    monitor: auto                   # auto | train_loss | val_loss
    save_last: true
    save_best: true
    every_epochs: 1
    resume_from: null

tracking:
  enabled: true
  provider: mlflow                  # Only implemented tracking provider
  tracking_uri: null               # Automatic local SQLite location
  storage_scope: shared            # shared | job
  experiment_name: gnn_diff_rntraj_nyc
  run_name: null                   # Optional MLflow display name
  tags:                             # User-defined string-to-string labels
    model: gnn_diff_rntraj
    dataset: nyc_15s_enriched_moves
  logging:
    metrics_every_steps: 100
    gnn_diagnostics_every_steps: 100
    gnn_embedding_probe_size: 128
    log_system_metrics: false
    log_resolved_config: true
    log_git_info: true
  artifacts:
    log_best_checkpoint: true
    log_last_checkpoint: false
    log_generated_trajectories: true
    log_training_logs: true

generation:
  enabled: true
  checkpoint: null                 # null when using checkpoint from current 'all' run
  checkpoint_choice: best          # best | last
  road_layer: roads
  output_name: nyc_generated_gps.parquet
  num_trajectories: 100           # null uses available length-distribution counts
  batch_size: 2
  sampler:
    name: ddpm                       # Only implemented sampler

runtime:
  device: auto                      # auto | cpu | cuda
  precision: fp32                   # Only implemented precision
  num_workers: 4
```

## 3. Field reference

Fields with a default can usually be omitted. The example includes them all for discoverability. Numeric constraints are enforced by Pydantic, and unknown experiment keys are rejected.

### Experiment identity and paths

| Attribute | Supported value / default | Meaning |
|---|---|---|
| `experiment.name` | Nonempty letters, digits, `_`, `-`, `.`; must start with a letter or digit; **required** | Folder name and identity for the experiment. |
| `experiment.seed` | Integer `>= 0`; default `101` | Seed for reproducibility. |
| `paths.graph_config` | Path; **required** | YAML defining road graph preprocessing, feature selection, and GNN architecture. |
| `paths.train_sequences` | Path; **required** | Prepared training road-sequence Parquet with edge IDs and position rates. |
| `paths.test_sequences` | Path or `null`; default `null` | Held-out road sequences, kept separate from training. |
| `paths.trip_id_map` | Path or `null`; default `null` | FMM ID to original trip mapping, required for grouped validation. |
| `paths.road_network` | Path or `null`; default `null` | GeoPackage road geometries used for generation. |
| `paths.length_distribution` | Path or `null`; default `null` | **Training** trajectory-length distribution used in generation. |
| `paths.test_reference_gps` | Path or `null`; default `null` | Optional held-out GPS reference supplied to the evaluation handoff. |
| `paths.output_root` | Directory path; default `outputs/experiments` | Where run folders, checkpoints, metrics, and default local MLflow storage live. |

### Diffusion model and conditioning

| Attribute | Supported value / default | Meaning |
|---|---|---|
| `model.diffusion.hidden_dim` | Even integer `>= 4`; default `128` | Denoiser's internal hidden width. |
| `model.diffusion.residual_layers` | Positive integer; default `10` | Number of denoising residual blocks. |
| `model.diffusion.diffusion_steps` | Positive integer; default `500` | Number of forward diffusion steps used during training. |
| `model.diffusion.beta_start` | Float strictly between `0` and `1`; default `0.0001` | Initial noise-schedule beta. |
| `model.diffusion.beta_end` | Float strictly between `0` and `1`; default `0.02` | Final beta; must be `>= beta_start`. |
| `model.diffusion.similarity_chunk_size` | Positive integer; default `128` | Chunk size for road-similarity scoring, affects memory use. |
| `model.road_encoder.mode` | **`joint` only**; default `joint` | Train road GNN together with the diffusion model. Frozen/pretrained modes are not implemented. |
| `model.conditioning.enabled` | **`false` only at present**; default `false` | Reserved for Jessica's metadata conditioning integration. `true` is rejected until implementation exists. |

### Training, optimizer, and learning-rate scheduler

| Attribute | Supported value / default | Meaning |
|---|---|---|
| `training.epochs` | Positive integer; default `30` | Number of training epochs. |
| `training.batch_size` | Positive integer; default `256` | Trajectories per training batch. |
| `training.max_batches` | Positive integer or `null`; default `null` | Cap training batches **per epoch**, useful for bounded runs. |
| `training.grad_clip_norm` | Positive float; default `1.0` | Gradient norm clipping threshold. |
| `training.optimizer.name` | `adam` \| `adamw`; default `adamw` | PyTorch optimizer selection. |
| `training.optimizer.learning_rate` | Positive float; default `0.001` | Initial learning rate for trainable parameters. |
| `training.optimizer.weight_decay` | Float `>= 0`; default `0.01` | Weight decay regularization. |
| `training.scheduler.name` | `none`, `cosine`, `reduce_on_plateau`; default `none` | Select the learning-rate schedule. |
| `training.scheduler.min_lr` | Finite float `>= 0` or nonempty list of such floats; default `0.0` | Lower learning-rate bound. `cosine` requires a scalar below the initial LR; a list for plateau must match the number of optimizer parameter groups. |
| `training.scheduler.mode` | `min`, `max`; default `min` | Plateau improvement direction: lower or higher monitored metric. Our current objectives use `min`. |
| `training.scheduler.factor` | Float `0 < factor < 1`; default `0.1` | Multiply the current LR by this number when a plateau is detected. Use `0.5` to halve it. |
| `training.scheduler.patience` | Integer `>= 0`; default `5` | Number of non-improving epochs tolerated before a reduction. |
| `training.scheduler.threshold` | Finite float `>= 0`; default `0.0001` | Required magnitude of improvement. |
| `training.scheduler.threshold_mode` | `rel`, `abs`; default `rel` | Interpret the threshold relatively or absolutely. |
| `training.scheduler.cooldown` | Integer `>= 0`; default `0` | Number of cooldown epochs after a reduction. |
| `training.scheduler.eps` | Finite float `>= 0`; default `1e-8` | Skip LR updates smaller than this. |

**Scheduler behavior:** `none` keeps the optimizer learning rate constant. `cosine` anneals over `training.epochs`. `reduce_on_plateau` observes the training/validation objective selected by the checkpoint monitor once per epoch. The `mode: max` option is exposed for parity with PyTorch, but our current checkpoint objectives are minimized, so use `min`. The old `patience_epochs` key remains accepted as a backwards-compatible alias for `patience`, but do not specify both.

**Comparing Yun's schedule:** `StepLR(step_size=..., gamma=0.5)` reduces the LR every fixed number of epochs, regardless of training progress. `ReduceLROnPlateau(factor=0.5, ...)` also halves the LR, but only after a plateau. `StepLR` is not currently a supported option in this experiment pipeline.

### Loss functions

The current loss registry has **exactly three** keys under `training.losses`: `noise_mse`, `reconstruction_mse`, and `hard_connectivity`. Each accepts the same four subfields.

| Attribute | Supported value / default | Meaning |
|---|---|---|
| `training.losses.<loss>.enabled` | Boolean; default `true` | Whether to calculate the loss/metric. |
| `training.losses.<loss>.weight` | Float `>= 0`; default `1.0` | Multiplier when included in the objective. |
| `training.losses.<loss>.include_in_total` | Boolean; default `true` | Include the weighted term in the reported total objective; `false` makes it diagnostic-only. |
| `training.losses.<loss>.every_steps` | Integer `>= 1`; default `1` | Measurement cadence. **Must equal `1` if enabled and included in total.** |

The individual terms mean:

| Loss | Meaning | Differentiable? |
|---|---|---|
| `noise_mse` | MSE between true Gaussian noise and predicted noise. | Yes |
| `reconstruction_mse` | MSE between clean and reconstructed road embeddings. | Yes |
| `hard_connectivity` | Count of invalid consecutive predicted road transitions. | **No** |

At least one differentiable loss (`noise_mse` or `reconstruction_mse`) must be enabled, included in total, and positively weighted. Hard connectivity may affect the **reported total objective and checkpoint selection** without affecting the gradients. BPR/BCE graph objectives are not available yet.

### Validation and checkpointing

| Attribute | Supported value / default | Meaning |
|---|---|---|
| `training.validation.enabled` | Boolean; default `false` | Reserve part of the configured training data for validation. |
| `training.validation.fraction` | Float strictly between `0` and `1`; default `0.1` | Fraction of original training trips used for validation. |
| `training.validation.split_group_columns` | **`[uid, tid]` only**; default `[uid, tid]` | Prevent different pieces of one original trip from crossing train/validation boundaries. |
| `training.checkpoint.monitor` | `auto` \| `train_loss` \| `val_loss`; default `auto` | Metric used for choosing `best.pt`. |
| `training.checkpoint.save_last` | Boolean; default `true` | Save latest checkpoint for recovery. |
| `training.checkpoint.save_best` | Boolean; default `true` | Save checkpoint with lowest monitored objective. |
| `training.checkpoint.every_epochs` | Integer `>= 1`; default `1` | Frequency of numbered epoch checkpoints. |
| `training.checkpoint.resume_from` | Checkpoint path or `null`; default `null` | Continue a previous training run; `null` starts a new run. |

`auto` chooses `val/objective` when validation is enabled, otherwise `train/objective`. `train_loss` always monitors training objective. `val_loss` requires `training.validation.enabled: true`. These monitor values refer to the **total objective**, including any enabled, included non-differentiable hard-connectivity term. The metric is **not** `gradient_objective`.

Use the training split for optional internal validation. Keep Yun's held-out test split separate and do not select checkpoints using the final test set.

### MLflow tracking

| Attribute | Supported value / default | Meaning |
|---|---|---|
| `tracking.enabled` | Boolean; default `true` | Enable MLflow logging. Disabling it does **not** disable local metrics/checkpoints. |
| `tracking.provider` | **`mlflow` only**; default `mlflow` | Tracking backend adapter. |
| `tracking.tracking_uri` | URI/string or `null`; default `null` | Explicit MLflow backend if supplied; otherwise use environment/default SQLite layout. |
| `tracking.storage_scope` | `shared` \| `job` | For automatic SQLite locations: `shared` keeps one database, `job` isolates Slurm jobs. Your current local configuration uses `shared` as its default. |
| `tracking.experiment_name` | Nonempty string; default `gnn_diff_rntraj_nyc` | MLflow experiment grouping multiple runs. |
| `tracking.run_name` | String or `null`; default `null` | Optional display label for a run. |
| `tracking.tags` | Mapping of string keys to string values; default `{}` | Searchable MLflow tags, such as model and dataset. |
| `tracking.logging.metrics_every_steps` | Integer `>= 1`; default `100` | Step-level metric logging interval. |
| `tracking.logging.gnn_diagnostics_every_steps` | Integer `>= 1`; default `100` | Frequency of GNN/denoiser gradient diagnostics. |
| `tracking.logging.gnn_embedding_probe_size` | Integer `>= 1`; default `128` | Number of fixed road segments observed for embedding statistics and drift. |
| `tracking.logging.log_system_metrics` | Boolean; default `false` | Toggle MLflow-supported system metrics. Slurm monitoring is separate. |
| `tracking.logging.log_resolved_config` | Boolean; default `true` | Save resolved experiment settings to the tracking run. |
| `tracking.logging.log_git_info` | Boolean; default `true` | Record available Git metadata. |
| `tracking.artifacts.log_best_checkpoint` | Boolean; default `true` | Copy best checkpoint into MLflow artifacts. |
| `tracking.artifacts.log_last_checkpoint` | Boolean; default `false` | Copy last checkpoint into MLflow artifacts. |
| `tracking.artifacts.log_generated_trajectories` | Boolean; default `true` | Log generated GPS Parquet as an artifact. |
| `tracking.artifacts.log_training_logs` | Boolean; default `true` | Log training log files as artifacts. |

**`shared` versus `job`:** `shared` lets you compare sequential runs in one MLflow UI, but concurrent SQLite writes on a network filesystem may lock. `job` gives separate databases and requires consolidation for cross-job comparisons. An explicit `tracking_uri` takes precedence over this automatic layout. MLflow does **not** need to run as a server during training; only the optional UI does.

### Generation and runtime

| Attribute | Supported value / default | Meaning |
|---|---|---|
| `generation.enabled` | Boolean; default `false` | Include generation as part of `all`. |
| `generation.checkpoint` | Checkpoint path or `null`; default `null` | Required for standalone `generate`; leave `null` for checkpoint from the current `all` run. |
| `generation.checkpoint_choice` | `best` \| `last`; default `best` | Which checkpoint from the current training run to use for generation. |
| `generation.road_layer` | Nonempty string; default `roads` | GeoPackage layer containing road geometries. |
| `generation.output_name` | Filename ending `.parquet`, no directories; default `nyc_generated_gps.parquet` | Name of generated GPS output file. |
| `generation.num_trajectories` | Integer `>= 1` or `null`; default `null` | Generated trajectory count; `null` follows the available length-distribution counts. |
| `generation.batch_size` | Integer `>= 1`; default `2` | Number of trajectories sampled in each generation batch. |
| `generation.sampler.name` | **`ddpm` only**; default `ddpm` | Current reverse diffusion sampler. DDIM/DPM-Solver are not registered yet. |
| `runtime.device` | `auto` \| `cpu` \| `cuda`; default `auto` | Device selection; `cuda` fails if no GPU is available. |
| `runtime.precision` | **`fp32` only**; default `fp32` | Numerical precision; AMP is not implemented in this configuration. |
| `runtime.num_workers` | Integer `>= 0`; default `4` | PyTorch DataLoader worker processes. |

## 4. Outputs, repeats, and common pitfalls

Independent training runs with the same YAML get distinct output folders:

```text
outputs/experiments/
├── <experiment.name>/
│   └── <run_id>/
│       ├── resolved_config.json
│       ├── data_split.json
│       ├── checkpoints/{best.pt,last.pt,epoch_*.pt}
│       ├── metrics/history.jsonl
│       ├── logs/train.log
│       └── generated/
└── mlflow/
    ├── tracking.db             # shared layout
    └── artifacts/
```

Even when `tracking.enabled: false`, the run directory, `history.jsonl`, training log, checkpoints, and generated files remain available. See [the MLflow guide](../../hpc/urban_syn/README_MLFLOW.md) for downloading and viewing HPC runs.

Changing the diffusion/GNN architecture can make a checkpoint incompatible with a new configuration. Reuse `best.pt` for generation with **matching** model and graph settings. If you change graph features or preprocessing transforms, rebuild the graph cache rather than silently using the old graph.

For implementation-level restrictions, see `src/urban_syn/experiment/config.py`, `src/urban_syn/diffusion/config.py`, and [the GNN config guide](../graph/README.md). Those Python modules are authoritative if the pipeline changes.
