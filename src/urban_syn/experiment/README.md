# Joint GNN + Diff-RNTraj Pipeline

This is the current training and generation pipeline for **URBAN-SYN**, our road-network-aware synthetic mobility project. It combines a graph neural network (GNN), which learns representations of roads, with Diff-RNTraj, a diffusion model that generates road sequences and positions along those roads.

This guide is for teammates working with the model, datasets, or experiment runs. For containers and HPC usage, see the [container guide](../../../hpc/urban_syn/README.md) and [Slurm guide](../../../hpc/urban_syn/README_SLURM.md). For tracking, results, and local MLflow viewing, see the [MLflow guide](../../../hpc/urban_syn/README_MLFLOW.md).

## 1. How the pieces fit together

```text
Original NYC trajectories                  Shared NYC road network (.gpkg)
          |                                           |
   15-second sampling                                 |
          |                                           |
     FMM map matching                                 |
          |                                           |
 Split by Yun's held-out users                        |
          |                                           |
  Convert matched road sequences                      |
          |                                           |
          |                              Prepare PyG graph if cache missing
          |                              (road features + GNN graph)
          |                                           |
          +------------------------+------------------+
                                   |
                         Joint GNN + diffusion training
                         (GNN and denoiser both update)
                                   |
                       Checkpoints and MLflow metrics
                                   |
                           DDPM trajectory sampling
                                   |
                     Road IDs + position rates to GPS
                                   |
                       Generated trajectory Parquet
                                   |
                   Optional shared-evaluator handoff
```

**Preprocessing and model execution are separate.** The experiment runner does not sample GPS points, run FMM, or choose the train/test split. Those steps produce the model-ready inputs ahead of time. However, the runner **does** prepare the PyG road graph automatically if the graph cache is missing.

The current model trains the GNN and diffusion denoiser together. The road embeddings are not fixed. Generation loads the selected checkpoint, uses DDPM, and reconstructs geographic coordinates from the road network. Mapping points back onto roads does not by itself guarantee that all generated road transitions are connected.

## 2. Prepare the model data

Our NYC workflow uses FMM output from 15-second sampled trajectories and follows **Yun's user-level train/test split**. Her `test_trips.csv` lists held-out `(uid, tid)` pairs. All FMM pieces belonging to those parent trips must stay on the test side, including any different `move_id` or `piece` values.

The FMM splitter uses `(uid, tid)` to determine membership. If a different input variant is prepared later, first confirm that the supplied test list covers every trip of each held-out user in that variant. **Do not create a new random split** when comparing with Yun's results.

The commands below assume sampling, FMM preparation, and FMM matching have already been performed. Run them from `fmm_scripts/`.

### Split the map-matching results

```bash
uv run split_fmm_matches.py \
  --config ./config/nyc.yaml \
  --fmm_matches ./outputs/nyc_15s_fmm_match.csv \
  --trip_id_map ./data/nyc_15s_fmm_trip_id_map.csv \
  --test_trips ./data/test_trips.csv \
  --output_dir ./outputs/nyc_15s_enriched_moves \
  --prefix nyc_15s
```

The splitter writes separate train and test FMM CSVs, a file for unassigned matches, and a summary. Some FMM rows can have `id=0` with no recoverable original trajectory identity. They remain unassigned rather than being forced into either split. This is different from **road `edge_id=0`**, which is a valid edge ID in our road network.

### Convert the splits into model-ready sequences

```bash
uv run python -m urban_syn_extension.convert_fmm_sequences \
  --train_fmm_matches ./outputs/nyc_15s_enriched_moves/nyc_15s_train_fmm_matches.csv \
  --test_fmm_matches ./outputs/nyc_15s_enriched_moves/nyc_15s_test_fmm_matches.csv \
  --output_dir ./outputs/urban_syn_ready \
  --prefix nyc_15s_enriched_moves
```

**Keep the converter's input paths pointed at the same directory the splitter just wrote.** Otherwise, it may silently use older split files that happen to exist elsewhere. The example uses `outputs/nyc_15s_enriched_moves/`, but it is fine to organize those files elsewhere as long as the splitter, converter, and graph YAML all use the actual matching paths.

Main outputs, relative to `fmm_scripts/`:

| Output | Purpose |
|---|---|
| `outputs/urban_syn_ready/*_train_prepared_road_sequences.parquet` | Training road IDs, position rates, and sequence lengths |
| `outputs/urban_syn_ready/*_test_prepared_road_sequences.parquet` | Held-out road sequences, kept separate from training |
| `outputs/urban_syn_ready/*_train_length_distribution.npy` | Training trajectory-length distribution used during generation |
| `outputs/urban_syn_ready/*_test_length_distribution.npy` | Held-out length distribution, for analysis rather than driving generation |
| `outputs/urban_syn_ready/*_sequence_preparation_summary.json` | Which matched rows were usable or skipped |

Each prepared sequence has `fmm_id`, `edge_ids`, `rates`, and `num_points`. The `edge_ids` are **shared road-network identifiers**, not GNN tensor indices. The model handles its own mapping and padding conventions.

### Training-only road-transition features

If the graph YAML enables `graph.transforms.edge_transition_count`, set its `match_csv` to the **training** FMM CSV. Do not use the test CSV or the unsplit FMM output to construct training graph features.

For this split, the source is:

```yaml
# In configs/graph/nyc_graph.yaml, under graph.transforms.edge_transition_count
match_csv: fmm_scripts/outputs/nyc_15s_enriched_moves/nyc_15s_train_fmm_matches.csv
```

This setting is only an example of the path value. Keep the other options in your existing graph YAML unchanged.

## 3. Running the experiment pipeline

From the URBAN-SYN repository root in the local development environment:

```bash
# Preview the stages without checking files.
uv run python -m urban_syn.experiment all \
  --config configs/experiments/nyc_joint.yaml --dry-run

# Check source inputs without training or building the graph.
uv run python -m urban_syn.experiment all \
  --config configs/experiments/nyc_joint.yaml --validate-only

# Run graph preparation, training, and optional generation.
uv run python -m urban_syn.experiment all \
  --config configs/experiments/nyc_joint.yaml
```

| Target | What it does |
|---|---|
| `validate` | Checks training and graph source inputs |
| `prepare-graph` | Reuses the PyG cache or builds it from the road network and graph YAML |
| `train` | Prepares the graph if necessary, then trains the GNN and denoiser |
| `generate` | Prepares the graph if necessary, then loads a specified checkpoint and generates GPS trajectories, without training |
| `export-eval` | Writes a manifest for the team's existing evaluator |
| `all` | Runs `validate`, `prepare-graph`, and `train`, followed by generation when enabled, and evaluation handoff when configured |

### What happens when the graph cache is missing?

The graph builder consumes the existing road GeoPackage and graph configuration. It creates:

```text
outputs/pyg/nyc_gnn/processed/
├── road_graph.pt
└── preprocessor.joblib
```

Both files are **outputs**, not prerequisites for a fresh run. When both are already present, the preparation stage reuses them. If either is missing, it runs the graph builder. Rebuilding can also require the training FMM CSV if the transition-count feature is enabled.

`--validate-only` verifies source files and settings, but never runs the graph builder. It also does not check every row of the datasets. The source road network, prepared sequences, and generation inputs still need to be present where configured.

### Generate without retraining

Point `generation.checkpoint` to a compatible checkpoint in the experiment YAML, then run:

```bash
uv run python -m urban_syn.experiment generate \
  --config configs/experiments/nyc_joint.yaml
```

For the `all` target, leave `generation.checkpoint: null` to use the selected checkpoint from that training run. Set `generation.enabled: true` to include generation in `all`.

## 4. YAML configuration guide

We use two YAML files rather than duplicating graph settings in every experiment:

| File | What belongs there |
|---|---|
| `configs/graph/nyc_graph.yaml` | Graph source/cache paths, road features, graph transforms, and the GNN architecture (`model.gnn`) |
| `configs/experiments/nyc_joint.yaml` | Dataset paths, diffusion architecture, training, losses, checkpointing, tracking, generation, and runtime choices |

The experiment YAML points to the graph YAML using `paths.graph_config`. Paths in the experiment YAML are resolved against the repository root, not against the location of the YAML. Pydantic validates the configuration against supported field types and values.

**To start another experiment:** copy an existing experiment YAML, give it a useful `experiment.name`, update only the settings you want to study, and pass it to `--config`. You can also reuse the same YAML for multiple independent runs. A new training run gets its own output directory and MLflow run, unless you explicitly resume from a checkpoint.

### Graph YAML settings

The graph configuration stays separate because multiple experiments can reuse the same road network, features, and encoder design. Common settings include:

| Graph YAML attribute | Meaning |
|---|---|
| `dataset.network_path` | Source road GeoPackage used by the graph builder |
| `dataset.root` | Cache directory where the builder writes `processed/road_graph.pt` and `preprocessor.joblib` |
| `features` | Which road attributes are encoded, including numeric and categorical inputs |
| `graph.transforms` | Graph feature transforms, including optional training-only transition counts |
| `model.gnn` | GNN type, layer structure, dimensions, and other encoder options |

**Changing graph features or the GNN architecture may require a rebuilt cache and a new training run.** The automatic preparation stage detects missing artifacts, not every change to the YAML. If you intentionally change the graph's feature specification, regenerate the cached artifacts rather than assuming the old cache reflects the new configuration.

### Example configuration

```yaml
experiment:
  name: nyc_joint_example
  seed: 101

paths:
  graph_config: configs/graph/nyc_graph.yaml
  train_sequences: fmm_scripts/outputs/urban_syn_ready/nyc_15s_enriched_moves_train_prepared_road_sequences.parquet
  road_network: fmm_scripts/data/road_network/nyc.gpkg
  length_distribution: fmm_scripts/outputs/urban_syn_ready/nyc_15s_enriched_moves_train_length_distribution.npy
  output_root: outputs/experiments

model:
  diffusion:
    hidden_dim: 128
    residual_layers: 10
    diffusion_steps: 500
    beta_start: 0.0001
    beta_end: 0.02
    similarity_chunk_size: 128
  road_encoder:
    mode: joint
  conditioning:
    enabled: false

training:
  epochs: 30
  batch_size: 32
  optimizer:
    name: adamw
    learning_rate: 0.001
    weight_decay: 0.01
  losses:
    noise_mse: {enabled: true, weight: 1.0, include_in_total: true}
    reconstruction_mse: {enabled: true, weight: 1.0, include_in_total: true}
    hard_connectivity: {enabled: true, weight: 1.0, include_in_total: true}

tracking:
  enabled: true
  provider: mlflow
  experiment_name: gnn_diff_rntraj_nyc
  storage_scope: shared

generation:
  enabled: true
  checkpoint: null
  checkpoint_choice: best
  sampler:
    name: ddpm
```

This is an introduction, not a full configuration. Use an existing experiment YAML for all available fields.

### Experiment identity and paths

| Attribute | Meaning |
|---|---|
| `experiment.name`, `experiment.seed` | Name for output organization, and reproducibility seed |
| `paths.graph_config` | Graph YAML containing the GNN and graph-building configuration |
| `paths.train_sequences`, `paths.test_sequences` | Prepared train and held-out sequence Parquet paths |
| `paths.trip_id_map` | FMM ID to original trip identity map, needed for grouped validation |
| `paths.road_network` | GeoPackage used to reconstruct generated GPS coordinates |
| `paths.length_distribution` | Training sequence-length distribution for generation |
| `paths.test_reference_gps` | Optional original held-out GPS data for the evaluation handoff |
| `paths.output_root` | Parent directory for runs, artifacts, and local MLflow storage |

### Model and training

| Attribute | Meaning |
|---|---|
| `model.diffusion.hidden_dim`, `residual_layers` | Denoiser width and number of residual blocks |
| `model.diffusion.diffusion_steps` | Diffusion training steps and DDPM schedule length |
| `model.diffusion.beta_start`, `beta_end` | Noise-schedule endpoints |
| `model.diffusion.similarity_chunk_size` | Chunk size for road-similarity computations, useful for GPU memory control |
| `model.road_encoder.mode` | Currently `joint`, both GNN and denoiser are trainable |
| `model.conditioning.enabled` | Reserved setting for Jessica's future metadata conditioning integration. Keep `false`; no conditioning provider is connected yet. |
| `training.epochs`, `batch_size` | Training passes and trajectories per batch |
| `training.max_batches` | Optional cap on batches **per epoch**, `null` means no cap |
| `training.grad_clip_norm` | Gradient clipping threshold |
| `training.optimizer.name`, `learning_rate`, `weight_decay` | Adam or AdamW configuration |
| `training.scheduler.name`, `min_lr`, `patience_epochs` | No scheduler, cosine schedule, or plateau-based reduction |

### Losses

Current terms are `noise_mse`, `reconstruction_mse`, and `hard_connectivity`. Each term has these controls:

| Attribute under `training.losses.<loss_name>` | Meaning |
|---|---|
| `enabled` | Compute the term or disable it |
| `weight` | Scale its reported contribution |
| `include_in_total` | Include it in the reported combined objective, or monitor it only |
| `every_steps` | How often it is computed, must be `1` if included in the objective |

`noise_mse` compares predicted and sampled diffusion noise. `reconstruction_mse` compares reconstructed and original road embeddings, rather than the position-rate channel. **Hard connectivity counts invalid predicted road transitions but does not provide gradients**, because it decodes discrete road IDs. The pipeline reports a separate gradient-bearing objective so we can distinguish what is optimized from what is monitored. Future road-link losses and pretraining strategies are not part of the current loss registry.

### Validation and checkpoints

| Attribute | Meaning |
|---|---|
| `training.validation.enabled` | Make a validation split from the supplied training data |
| `training.validation.fraction` | Proportion of training trips assigned to validation |
| `training.validation.split_group_columns` | Keep `[uid, tid]` together so pieces from one parent trip cannot cross the validation boundary |
| `training.checkpoint.monitor` | `auto`, `train_loss`, or `val_loss` |
| `training.checkpoint.save_best`, `save_last` | Keep best and latest training snapshots |
| `training.checkpoint.every_epochs` | Frequency of additional numbered checkpoints |
| `training.checkpoint.resume_from` | Resume a compatible previous run; `null` starts a new run |

**Validation is separate from Yun's held-out test set.** If enabled, the pipeline splits only the prepared training data and uses `paths.trip_id_map` to respect original trip membership. The test data is not used for checkpoint selection.

**How `best.pt` is selected:** `training.checkpoint.monitor: auto` uses the lowest `val/objective` when validation is enabled, otherwise the lowest `train/objective`. `train_loss` always monitors the training objective, and `val_loss` requires validation. These are **total reported objectives**, including any loss term configured with `include_in_total: true`. Hard connectivity can dominate that total but is non-differentiable, so it influences `best.pt` selection even though it does not contribute gradients. The separate `train/gradient_objective` (or validation counterpart) is logged for interpretation, but is not currently a selectable checkpoint monitor.

### MLflow tracking and diagnostics

| Attribute | Meaning |
|---|---|
| `tracking.enabled`, `provider` | Enable tracking; currently supports MLflow |
| `tracking.experiment_name`, `run_name`, `tags` | How runs are named and organized in MLflow |
| `tracking.tracking_uri` | Explicit backend URI, or `null` for the configured local backend |
| `tracking.storage_scope` | `shared` for one SQLite database or `job` for separate databases per Slurm job |
| `tracking.logging.metrics_every_steps` | Step-level metric logging frequency |
| `tracking.logging.gnn_diagnostics_every_steps` | GNN and denoiser gradient diagnostic frequency |
| `tracking.logging.gnn_embedding_probe_size` | Fixed sample size for embedding monitoring |
| `tracking.logging.log_system_metrics`, `log_resolved_config`, `log_git_info` | System metrics, saved configuration, and Git details |
| `tracking.artifacts.log_best_checkpoint`, `log_last_checkpoint` | Checkpoint artifact upload choices |
| `tracking.artifacts.log_generated_trajectories`, `log_training_logs` | Trajectory and log artifact upload choices |

MLflow stores run parameters, losses, gradient norms, and selected artifacts. **An MLflow server is not required during training.** A shared local database allows sequential runs to be compared in one UI; per-job databases isolate concurrent Slurm writers but require separate inspection or later consolidation. With `tracking.enabled: false`, the pipeline still writes its local `history.jsonl`, training logs, checkpoints, and generated files. For snapshot download and Windows/macOS viewing commands, see [README_MLFLOW.md](../../../hpc/urban_syn/README_MLFLOW.md).

The GNN embedding probe uses a **fixed subset of roads in evaluation mode** at checkpoint observations. This measures embedding changes without training-time dropout noise; the GNN still runs normally for optimization.

### Generation and runtime

| Attribute | Meaning |
|---|---|
| `generation.enabled` | Add generation to the `all` workflow |
| `generation.checkpoint` | Existing checkpoint for standalone generation |
| `generation.checkpoint_choice` | Select `best` or `last` after training in `all` |
| `generation.road_layer` | Road-geometry layer in the GeoPackage, typically `roads` |
| `generation.num_trajectories`, `batch_size` | Number of trajectories to sample and batch size |
| `generation.output_name` | Generated GPS Parquet filename |
| `generation.sampler.name` | Currently `ddpm` |
| `runtime.device`, `precision`, `num_workers` | Device choice, FP32, and DataLoader workers |

## 5. Saved outputs and next steps

Each independent training run writes under a unique path, for example:

```text
outputs/experiments/
├── <experiment_name>/
│   └── <run_id>/
│       ├── resolved_config.json
│       ├── data_split.json
│       ├── checkpoints/       # best.pt, last.pt, epoch_*.pt
│       ├── metrics/           # history.jsonl
│       ├── logs/              # train.log
│       └── generated/         # GPS Parquet and generation manifest
└── mlflow/                    # shared or per-job tracking stores
```

When a held-out GPS reference is configured, the `export-eval` stage writes an evaluation handoff manifest. It **does not run** the team's shared quality metrics by itself.

Current functionality includes joint GNN/diffusion training, automatic graph-cache preparation, configurable losses, grouped validation, resume checkpoints, MLflow, embedding diagnostics, and DDPM-based GPS generation. The container and Slurm scripts in `hpc/urban_syn/` run this same pipeline on HPC. Future research work may add alternative sampling methods, differentiable road-link objectives, GNN pretraining, and Jessica's metadata conditioning once its model interface is agreed. The current `conditioning.enabled` entry is configuration-only.
