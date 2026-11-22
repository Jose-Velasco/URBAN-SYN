# URBAN-SYN YAML Configuration Reference

Use this reference when editing a YAML file and you need to know **what a setting does, which values it accepts, and what must be configured alongside it**. The shorter guides explain the usual workflows:

- [Experiment guide](experiments/README.md) — training, generation, and experiment setup.
- [Road graph guide](graph/README.md) — graph features and GNN setup.
- [HPC/Slurm guide](../hpc/urban_syn/README_SLURM.md) — input files and submission commands.
- [GNN pretraining guide](../src/urban_syn/link_pretrain/README.md) — standalone link prediction.

There are **four kinds of YAML**. Use the matching section below; similarly named keys can have different defaults in different files.

| YAML type | Example location | Controls |
|---|---|---|
| **Joint experiment** | `configs/experiments/nyc_gnn_comparison/joint_finetune.yaml` | Diffusion model, training, pretrained encoder, generation, tracking |
| **GNN pretraining** | `configs/experiments/hpc_smoke_test/nyc_link_pretrain_bpr.yaml` | Road-link learning, negative sampling, validation, early stopping |
| **Road graph** | `configs/experiments/hpc_smoke_test/nyc_graph.yaml` or `configs/graph/nyc_graph.yaml` | Road features, graph transforms, GNN architecture |
| **Study** | `configs/experiments/hpc_smoke_test/study.yaml` | Named experiments, pretrained dependencies, Slurm defaults |

**Paths:** Specify repository-relative paths unless an absolute path is required. On HPC, the repository is mounted inside the container at `/workspace`. YAML examples below show only the relevant settings, not complete configurations. Copy an existing experiment YAML as your starting point.

---

## 1. Joint diffusion experiment

A joint experiment loads the road graph, builds a road GNN and diffusion model, trains them, and optionally generates synthetic GPS trajectories. `model.road_encoder` controls whether the GNN starts from random weights or a pretrained checkpoint.

### Identity and input files

| Setting | Allowed values / default | Purpose |
|---|---|---|
| `experiment.name` | **Required**; letters, digits, `_`, `-`, `.`; starts with a letter/digit | Names the experiment's output directory. |
| `experiment.seed` | Nonnegative integer; `101` | Controls reproducible initialization and sampling. |
| `paths.graph_config` | **Required** path to graph YAML | Selects road features, graph cache, and GNN architecture. |
| `paths.train_sequences` | **Required** Parquet path | Prepared map-matched training trajectories. |
| `paths.test_sequences` | Path or `null`; `null` | Optional held-out road sequences; not used for choosing the training checkpoint. |
| `paths.trip_id_map` | CSV path or `null`; `null` | Maps FMM pieces back to original trips; **required for grouped validation**. |
| `paths.road_network` | GeoPackage path or `null`; `null` | Road geometry needed to convert sampled roads into GPS coordinates. |
| `paths.length_distribution` | `.npy` path or `null`; `null` | Training trajectory-length distribution used when generating. |
| `paths.test_reference_gps` | Path or `null`; `null` | Optional interface to an evaluation handoff; **not required for training or generation**. |
| `paths.output_root` | Directory; `outputs/experiments` | Where per-run checkpoints, logs, and generated files are stored. |

### Diffusion model and GNN initialization

| Setting | Allowed values / default | Purpose |
|---|---|---|
| `model.diffusion.hidden_dim` | Even integer ≥ 4; `128` | Width of the diffusion denoiser. |
| `model.diffusion.residual_layers` | Positive integer; `10` | Number of denoising residual blocks. |
| `model.diffusion.diffusion_steps` | Positive integer; `500` | Number of steps in the diffusion noise schedule; also used by DDPM generation. |
| `model.diffusion.beta_start` | Between 0 and 1; `0.0001` | Noise variance at start of schedule. |
| `model.diffusion.beta_end` | Between 0 and 1, ≥ `beta_start`; `0.02` | Noise variance at end of schedule. |
| `model.diffusion.similarity_chunk_size` | Positive integer; `128` | Roads scored per similarity chunk; **lower to reduce GPU memory**. |
| `model.road_encoder.mode` | `joint` only; `joint` | Compatibility setting; does **not** decide freezing or initialization. |
| `model.road_encoder.initialization` | **`random` / `pretrained`**; `random` | Start the GNN from random weights or a pretrained `encoder.pt`. |
| `model.road_encoder.trainable` | `true` / `false`; `true` | Whether diffusion training updates GNN weights. |
| `model.road_encoder.checkpoint` | Path or `null`; `null` | Encoder weights for `pretrained`; leave `null` when a **study** injects `pretrained_from`. |
| `model.conditioning.enabled` | **`false` only**; `false` | Reserved for metadata conditioning; enabling it is not implemented. |

**Supported encoder combinations:**

| Experiment | `initialization` | `trainable` | `checkpoint` |
|---|---|---|---|
| Train GNN and diffusion from scratch | `random` | `true` | `null` |
| Pretrained GNN, frozen during diffusion training | `pretrained` | `false` | `encoder.pt` path, or `null` with study `pretrained_from` |
| Pretrained GNN, fine-tuned with diffusion | `pretrained` | `true` | `encoder.pt` path, or `null` with study `pretrained_from` |

A **random, frozen** GNN is rejected. For pretrained experiments, the GNN architecture and **ordered input feature groups** must match the encoder checkpoint. The number of roads itself does not need to match when the feature schema is compatible.

### Training and optimizer

| Setting | Allowed values / default | Purpose |
|---|---|---|
| `training.epochs` | Positive integer; `30` | Maximum full passes over the training dataset. |
| `training.batch_size` | Positive integer; `256` | Number of trajectories per training batch. Reduce if GPU memory is insufficient. |
| `training.max_batches` | Positive integer or `null`; `null` | Limit batches **per epoch**; `null` uses all batches. Useful for quick runs. |
| `training.grad_clip_norm` | Positive number; `1.0` | Caps gradient magnitude for training stability. |
| `training.optimizer.name` | **`adam` / `adamw`**; `adamw` | Optimizer for trainable model parameters. |
| `training.optimizer.learning_rate` | Positive number; `0.001` | Initial optimizer step size. |
| `training.optimizer.weight_decay` | Nonnegative number; `0.01` | Weight regularization. |

### Learning-rate scheduler

| Setting | Allowed values / default | Purpose |
|---|---|---|
| `training.scheduler.name` | **`none` / `cosine` / `reduce_on_plateau`**; `none` | Constant rate, smooth decay, or reduce when progress stalls. |
| `training.scheduler.min_lr` | Nonnegative number, or nonempty list for plateau; `0.0` | Minimum rate. **Cosine requires one number below the initial learning rate**. |
| `training.scheduler.mode` | **`min` / `max`**; `min` | Whether smaller or larger monitored values are better; use `min` for current losses. |
| `training.scheduler.factor` | Between 0 and 1; `0.1` | Multiplier applied when plateau scheduler reduces LR. |
| `training.scheduler.patience` | Nonnegative integer; `5` | Non-improving epochs before reducing LR. |
| `training.scheduler.threshold` | Nonnegative number; `0.0001` | Smallest meaningful change. |
| `training.scheduler.threshold_mode` | **`rel` / `abs`**; `rel` | Relative or absolute threshold. |
| `training.scheduler.cooldown` | Nonnegative integer; `0` | Epochs to wait after LR reduction. |
| `training.scheduler.eps` | Nonnegative number; `1e-8` | Ignore very small LR updates. |

`reduce_on_plateau` monitors the same training/validation objective used for checkpoint selection. Plateau-specific fields are irrelevant with `none` or `cosine`. Older YAMLs may use `patience_epochs` instead of `patience`; **do not specify both**.

### Diffusion loss terms

These are the **only** current keys under `training.losses`. Each accepts the same four settings.

| Loss key | What it measures | Affects gradients? |
|---|---|---|
| `noise_mse` | Error in predicting added diffusion noise | **Yes** |
| `reconstruction_mse` | Error in reconstructing clean road embeddings | **Yes** |
| `hard_connectivity` | Count of invalid consecutive road transitions | **No**; monitoring/checkpoint selection only if included |

| Setting (replace `<loss>` with the key above) | Allowed values / default | Purpose |
|---|---|---|
| `training.losses.<loss>.enabled` | `true` / `false`; `true` | Calculate this term. |
| `training.losses.<loss>.weight` | Nonnegative number; `1.0` | Multiplier applied if included in the total. |
| `training.losses.<loss>.include_in_total` | `true` / `false`; `true` | Include in reported objective and checkpoint ranking. |
| `training.losses.<loss>.every_steps` | Integer ≥ 1; `1` | Frequency of computation. **Must be 1 if included in total**. |

Keep at least one **differentiable** loss enabled, positively weighted, and included in total. Setting `hard_connectivity.include_in_total: true` changes the reported objective/checkpoint ranking, but it **cannot train connectivity directly** because that term has no gradient. To log it without influencing best-checkpoint selection, set `include_in_total: false`.

### Validation and checkpointing

| Setting | Allowed values / default | Purpose |
|---|---|---|
| `training.validation.enabled` | `true` / `false`; `false` | Set aside part of the training data for validation. |
| `training.validation.fraction` | Between 0 and 1; `0.1` | Portion reserved from **training**, not from the held-out test set. |
| `training.validation.split_group_columns` | **`[uid, tid]` only** | Keep pieces of an original trip together across the split. Requires `paths.trip_id_map`. |
| `training.checkpoint.monitor` | **`auto` / `train_loss` / `val_loss`**; `auto` | Select metric for `best.pt`. `auto` prefers validation when enabled. |
| `training.checkpoint.save_best` | `true` / `false`; `true` | Save checkpoint with best monitored objective. |
| `training.checkpoint.save_last` | `true` / `false`; `true` | Save latest checkpoint for recovery/resuming. |
| `training.checkpoint.every_epochs` | Integer ≥ 1; `1` | Frequency of additional numbered checkpoints. |
| `training.checkpoint.resume_from` | Existing checkpoint path or `null`; `null` | Resume training, rather than starting a new run. |

`monitor: val_loss` requires `validation.enabled: true`. Checkpoint scores use the **total configured objective**, including any non-differentiable term included in total. Avoid using the held-out test set to choose checkpoints.

### MLflow and local logs

| Setting | Allowed values / default | Purpose |
|---|---|---|
| `tracking.enabled` | `true` / `false`; `true` | Log to MLflow. Run folders and checkpoints still save when false. |
| `tracking.provider` | **`mlflow` only** | Tracking backend. |
| `tracking.tracking_uri` | URI/string or `null`; `null` | Explicit MLflow backend; `null` uses environment/default storage. |
| `tracking.storage_scope` | **`shared` / `job`**; `shared` | Shared SQLite database or separate storage per Slurm job. |
| `tracking.experiment_name` | Nonempty string; `gnn_diff_rntraj_nyc` | MLflow group for comparing related runs. |
| `tracking.run_name` | String or `null`; `null` | Optional readable label. |
| `tracking.tags` | String-to-string mapping; `{}` | Searchable descriptors such as strategy and dataset. |
| `tracking.logging.metrics_every_steps` | Integer ≥ 1; `100` | Step-level metric logging interval. |
| `tracking.logging.gnn_diagnostics_every_steps` | Integer ≥ 0; `100` | Gradient diagnostics interval; **0 disables**. |
| `tracking.logging.gnn_embedding_enabled` | `true` / `false`; `true` | Record road embedding diagnostic statistics. |
| `tracking.logging.gnn_embedding_every_epochs` | Integer ≥ 1; `1` | How often to measure embeddings. |
| `tracking.logging.gnn_embedding_sample_size` | Integer ≥ 1; `128` | Number of fixed roads sampled for embedding diagnostics. |
| `tracking.logging.log_system_metrics` | `true` / `false`; `false` | Track CPU/memory/GPU metrics supported by MLflow. |
| `tracking.logging.log_resolved_config` | `true` / `false`; `true` | Save resolved settings with run. |
| `tracking.logging.log_git_info` | `true` / `false`; `true` | Save Git revision metadata. |
| `tracking.artifacts.log_best_checkpoint` | `true` / `false`; `true` | Copy `best.pt` to MLflow artifacts. |
| `tracking.artifacts.log_last_checkpoint` | `true` / `false`; `false` | Copy `last.pt` to MLflow artifacts. |
| `tracking.artifacts.log_generated_trajectories` | `true` / `false`; `true` | Copy generated Parquet to MLflow artifacts. |
| `tracking.artifacts.log_training_logs` | `true` / `false`; `true` | Copy training logs to MLflow artifacts. |

Older YAMLs may call the embedding sample field `gnn_embedding_probe_size`. The current name is **`gnn_embedding_sample_size`**; do not specify both. On shared network filesystems, concurrent SQLite writers may contend; use `job` storage when isolation is needed. MLflow's web server is **not required during training**.

### Generation and runtime

| Setting | Allowed values / default | Purpose |
|---|---|---|
| `generation.enabled` | `true` / `false`; `false` | Adds generation to a joint `all` run. |
| `generation.checkpoint` | Path or `null`; `null` | For standalone `generate`, provide an existing joint checkpoint; `null` uses current `all` run. |
| `generation.checkpoint_choice` | **`best` / `last`**; `best` | Which current-run checkpoint to generate from. Save the selected checkpoint during training. |
| `generation.road_layer` | Nonempty GeoPackage layer; `roads` | Source road geometries. |
| `generation.output_name` | Filename ending `.parquet`; `nyc_generated_gps.parquet` | Name under the run's `generated/` directory. |
| `generation.num_trajectories` | Positive integer or `null`; `null` | Number to generate; `null` uses available length-distribution counts. |
| `generation.batch_size` | Positive integer; `2` | Trajectories generated in parallel; lower to reduce memory. |
| `generation.sampler.name` | **`ddpm` only** | Current supported reverse diffusion method. |
| `runtime.device` | **`auto` / `cpu` / `cuda`**; `auto` | Execution device. `cuda` requires an available GPU. |
| `runtime.precision` | **`fp32` only** | Numeric precision; mixed precision is not configured here. |
| `runtime.num_workers` | Nonnegative integer; `4` | Number of DataLoader worker processes; use 0 to disable multiprocessing. |

For generation, configure `paths.road_network`, `paths.length_distribution`, and a compatible joint checkpoint. `--stage all` on HPC (or `all` locally) includes generation **only if enabled**. Generating trajectories is separate from evaluating their quality.

---

## 2. Standalone GNN link-pretraining experiment

Pretraining learns road embeddings by predicting whether two road segments are connected **in the directed road graph**. It trains a GNN and a directional link scorer, then exports `encoder.pt` for joint diffusion training. It does **not** need trajectory Parquet inputs or a diffusion model. It reads the same graph YAML as the joint model.

### Experiment, graph split, and negatives

| Setting | Allowed values / default | Purpose |
|---|---|---|
| `experiment.name` | Required directory-safe name | Pretraining output directory. |
| `experiment.seed` | Nonnegative integer; `101` | Reproducible split, negative sampling, and initialization. |
| `paths.graph_config` | **Required** path | Graph features and GNN used for link pretraining. |
| `paths.output_root` | Directory; `outputs/experiments` | Local outputs and checkpoints. |
| `pretraining.objective` | **`bpr` / `bce`**; `bpr` | BPR ranks true links above negatives; BCE classifies links as real/non-links. |
| `pretraining.graph_split.val_fraction` | Between 0 and 1; `0.10` | Fraction of true directed links for validation. |
| `pretraining.graph_split.test_fraction` | Between 0 and 1; `0.10` | Fraction of true directed links for held-out testing. |
| `pretraining.graph_split.disjoint_train_ratio` | Between 0 and 1; `0.20` | Portion of training positives reserved for supervision instead of message-passing edges. |
| `pretraining.positives_per_seed` | **`1` only** | One positive outgoing link per sampled seed road. |
| `pretraining.negatives.strategy` | **`khop` / `mixed`**; `mixed` | Nearby hard negatives only, or nearby plus uniform non-links. |
| `pretraining.negatives.amount` | Integer ≥ 1; `2` | Negatives sampled per positive. |
| `pretraining.negatives.k_hops` | Integer ≥ 2; `2` | Maximum directed hop distance for hard-negative candidates (distance 2 through K). |
| `pretraining.negatives.hard_fraction` | Between 0 and 1; `0.75` | Target portion of hard negatives for `mixed`; not used to choose an independent fraction for `khop`. |
| `pretraining.negatives.evaluation_chunk_size` | Integer ≥ 1; `128` | How many held-out positive links are sampled/scored per chunk; **does not reduce the test-set size**. |

The split must satisfy `val_fraction + test_fraction < 1`. Training message passing uses the training graph, not links held out for validation/test. True links from the complete known graph are excluded from negative sampling.

### Directional link scorer

| Setting | Allowed values / default | Purpose |
|---|---|---|
| `pretraining.decoder.hidden_dim` | Positive integer; `128` | Hidden width of the link-scoring MLP. |
| `pretraining.decoder.num_layers` | Integer ≥ 2; `2` | Total linear layers, **including** final logit output layer. |
| `pretraining.decoder.activation` | Nonempty PyG-resolved activation; `relu` | Hidden-layer nonlinearity, e.g. `relu`, `gelu`, `elu`. |
| `pretraining.decoder.dropout` | Number from 0 to below 1; `0.0` | Hidden-layer dropout. |
| `pretraining.decoder.normalization` | **`layer_norm` / `batch_norm` / `null`**; `null` | Hidden-layer normalization. |

The decoder is trained for link prediction but **not transferred** to diffusion. `encoder.pt` contains the reusable road GNN; `best.pt` is the pretraining checkpoint used for testing or resuming.

### Pretraining optimizer, scheduler, stopping, and checkpoints

| Setting | Allowed values / default | Purpose |
|---|---|---|
| `training.epochs` | Integer ≥ 1; `100` | Maximum pretraining epochs. |
| `training.batch_size` | Integer ≥ 1; `128` | **Seed roads** per batch, not trajectories. |
| `training.max_batches` | Integer ≥ 1 or `null`; `null` | Batch cap per epoch; `null` processes all available batches. |
| `training.grad_clip_norm` | Positive number; `1.0` | Limit gradient norm. |
| `training.optimizer.name` | **`adam` / `adamw`**; `adamw` | Optimizer for GNN and link scorer. |
| `training.optimizer.learning_rate` | Positive number; `0.001` | Initial learning rate. |
| `training.optimizer.weight_decay` | Nonnegative number; `0.01` | Regularization. |
| `training.scheduler.name` | **`none` / `cosine` / `reduce_on_plateau`**; `reduce_on_plateau` | Learning-rate schedule. |
| `training.scheduler.min_lr` | Nonnegative scalar; `0.0` | Learning-rate floor; cosine requires it below initial LR. |
| `training.scheduler.mode` | **`min` only** | Pretraining monitors lower validation loss. |
| `training.scheduler.factor` | Between 0 and 1; `0.5` | Rate multiplier after plateau. |
| `training.scheduler.patience` | Integer ≥ 0; `10` | Non-improving validation epochs before LR reduction. |
| `training.scheduler.threshold` | Nonnegative number; `0.0001` | Smallest meaningful improvement. |
| `training.scheduler.threshold_mode` | **`rel` / `abs`**; `rel` | Relative or absolute threshold. |
| `training.scheduler.cooldown` | Integer ≥ 0; `0` | Pause plateau detection after reduction. |
| `training.scheduler.eps` | Nonnegative number; `1e-8` | Ignore tiny LR updates. |
| `training.early_stopping.enabled` | `true` / `false`; `true` | Stop when validation loss stops improving. |
| `training.early_stopping.patience` | Integer ≥ 1; `15` | Number of non-improving validation epochs allowed. |
| `training.early_stopping.min_delta` | Nonnegative number; `0.0` | Minimum improvement to reset patience. |
| `training.checkpoint.save_best` | `true` / `false`; `true` | Save best validation checkpoint. |
| `training.checkpoint.save_last` | `true` / `false`; `true` | Save latest for recovery. |
| `training.checkpoint.every_epochs` | Integer ≥ 1; `10` | Frequency of numbered checkpoints. |
| `training.checkpoint.resume_from` | Existing checkpoint path or `null`; `null` | Resume pretraining. |

**Pretraining always uses validation loss for checkpoint selection.** Scheduler and early stopping both use this validation signal, but they do different things: one changes learning rate; the other ends training.

### Pretraining tracking and data loading

| Setting | Allowed values / default | Purpose |
|---|---|---|
| `tracking.enabled` | `true` / `false`; `true` | Enable MLflow; local files still save when false. |
| `tracking.provider` | **`mlflow` only** | Tracking provider. |
| `tracking.experiment_name` | Nonempty string; `road_link_pretrain_nyc` | MLflow experiment grouping. |
| `tracking.tracking_uri` | URI or `null`; `null` | Explicit or automatic tracking backend. |
| `tracking.storage_scope` | **`shared` / `job`**; `shared` | Shared tracking database or per-job storage. |
| `tracking.run_name` | String or `null`; `null` | Optional readable run label. |
| `tracking.tags` | String-to-string mapping; `{}` | Searchable MLflow metadata. |
| `tracking.metrics_every_steps` | Integer ≥ 1; `100` | Training metric logging cadence. **Here it is directly under `tracking`**, not `tracking.logging`. |
| `tracking.logging.gnn_diagnostics_every_steps` | Integer ≥ 0; `100` | Encoder gradient diagnostic cadence; 0 disables. |
| `tracking.logging.gnn_embedding_enabled` | `true` / `false`; `true` | Record road embedding statistics. |
| `tracking.logging.gnn_embedding_every_epochs` | Integer ≥ 1; `1` | Epoch interval for embedding diagnostics. |
| `tracking.logging.gnn_embedding_sample_size` | Integer ≥ 1; `128` | Fixed number of roads used for comparison. |
| `runtime.device` | **`auto` / `cpu` / `cuda`**; `auto` | Execution device. |
| `runtime.precision` | **`fp32` only** | Numeric precision. |
| `runtime.num_workers` | Integer ≥ 0; `4` | Sampling/DataLoader workers. |
| `runtime.prefetch_factor` | Integer ≥ 1; `2` | Batches each worker prepares ahead; meaningful when workers > 0. |
| `runtime.persistent_workers` | `true` / `false`; `true` | Keep workers between epochs (requires workers > 0 to have an effect). |
| `runtime.pin_memory` | `true` / `false`; `true` | Allocate pinned CPU memory for faster transfer to GPU. |

Pretraining and joint training have **different YAML schemas**. For example, pretraining uses `tracking.metrics_every_steps`, while joint training uses `tracking.logging.metrics_every_steps`. Do not copy the whole `training` or `tracking` section from one type into the other.

---

## 3. Road graph and GNN configuration

This YAML is shared by pretraining and joint diffusion. It prepares a directed **road-as-node** graph: each road segment is a graph node, and directed edges represent physically connected road transitions. The original `edge_id` may be zero; diffusion uses `edge_id + 1` internally so zero can represent padding.

### Road inputs and raw features

| Setting | Allowed values / default | Purpose |
|---|---|---|
| `dataset.network_path` | **Required** GeoPackage path | Source road segments, geometries, and OSM attributes. |
| `dataset.layer` | GeoPackage layer name; `roads` | Layer to read. |
| `dataset.root` | **Required** directory | Where `processed/road_graph.pt` and `processed/preprocessor.joblib` are cached. |
| `features.numeric` | List drawn from **`log_length`, `speed_kph`, `lanes`, `width`** | Road length, speed, lane count, width. |
| `features.categorical` | List drawn from **`highway`, `surface`, `access`, `junction`** | OSM road type, surface, access restriction, junction kind. |
| `features.boolean` | List drawn from **`oneway`, `bridge`, `tunnel`** | Binary road attributes. |
| `features.geometry` | List drawn from **`start_lon`, `start_lat`, `mid_lon`, `mid_lat`, `end_lon`, `end_lat`, `relative_x_km`, `relative_y_km`, `bearing_sin`, `bearing_cos`, `straightness`** | Road location, direction, and shape. |
| `graph.allow_self_loops` | `true` / `false`; `false` | Whether the built graph permits road-to-itself transitions. |

Each feature group accepts a list of **column names**, not their observed values. For example, `highway` can represent OSM categories such as `residential`, `primary`, or `footway`; those observed categories are **data-dependent**, not a fixed enum in the YAML. Categorical values are one-hot encoded. Individual groups may be `[]`, but at least one raw feature must remain.

### Optional graph transforms

Add a transform under `graph.transforms` and set `enabled: true` to compute it. Then select its output in `model.node_features` or `model.edge_features`. Every transform has `enabled: true/false` (omitted/false means disabled).

| Transform / settings | Accepted values / typical default | Creates / why use it |
|---|---|---|
| `degree_features.log_scale` | Boolean; `true` | `degree_features`: local incoming/outgoing road connections. |
| `local_degree_profile.log_scale` | Boolean; `true` | `local_degree_profile`: degree patterns around a road. |
| `random_walk_pe.walk_length` | Positive integer; `3` | `random_walk_pe`: random-walk positional information. |
| `random_walk_pe.drop_first` | Boolean; `true` | Drop the first return-probability column; with this enabled, walk length must be ≥ 2. |
| `laplacian_pe.k` | Positive integer; `16` | `laplacian_pe`: spectral positional encoding (can be expensive). |
| `betweenness.k` | Positive integer; `100` | `betweenness`: sample size for approximate network centrality. |
| `betweenness.seed` | Integer; `42` | Reproducible sampling. |
| `betweenness.normalized` | Boolean; `true` | Normalized centrality. |
| `betweenness.parallel` | Boolean; `false` | Multicore approximate centrality; can use extra memory. |
| `pagerank_features.alpha` | Number strictly between 0 and 1; `0.85` | `pagerank_features`: directed road importance. |
| `pagerank_features.tolerance` | Positive number; `1e-8` | Iterative convergence tolerance. |
| `clustering_features.turbo` | Boolean; `false` | `clustering_features`: local clustering, faster mode at extra memory cost. |
| `constraint_features` | Only `enabled` | `constraint_features`: structural-hole constraint. |
| `edge_transition_count.match_csv` | **Required** training FMM CSV with `cpath` when enabled | Observed directed road-to-road transitions. |
| `edge_transition_count.smoothing.method` | **`additive` only**; `additive` | Smooth outgoing transition probabilities. |
| `edge_transition_count.smoothing.alpha` | Positive number; `1.0` | Smoothing pseudocount. |
| `adamic_adar_edge_features.batch_size` | Positive integer; `25000` | `adamic_adar_edge_features`: shared-neighbor edge similarity. |
| `random_walk_edge_features` | Only `enabled` | `random_walk_edge_features`: relative random-walk features along links. |
| `laplacian_edge_features` | Only `enabled` | `laplacian_edge_features`: relative spectral features along links. |

`edge_transition_count` produces **two selectable outputs**: `edge_transition_count` (raw counts) and `edge_transition_prob` (smoothed probabilities). Use a **training-only** FMM match CSV, never held-out test matches. `random_walk_edge_features` needs `random_walk_pe`; `laplacian_edge_features` needs `laplacian_pe`. Put the corresponding node transform first. Expensive graph transforms run when building the graph cache, **not once per epoch**.

### Select GNN input groups

| Setting | Accepted values / default | Purpose |
|---|---|---|
| `model.node_features` | Nonempty unique list; `[x]` | Graph node attributes concatenated as GNN input. `x` contains selected raw feature groups. |
| `model.edge_features` | Unique list, can be `[]`; `[]` | Graph edge attributes concatenated as edge input. GINE requires a nonempty usable edge input. |

Common node groups: `x`, `degree_features`, `local_degree_profile`, `random_walk_pe`, `laplacian_pe`, `betweenness`, `pagerank_features`, `clustering_features`, `constraint_features`. Common edge groups: `edge_transition_count`, `edge_transition_prob`, `adamic_adar_edge_features`, `random_walk_edge_features`, `laplacian_edge_features`.

**Selecting a group does not enable its transform.** It must already exist in the graph cache. Changing graph feature definitions or transforms requires rebuilding the cache (or using a different `dataset.root`); the loader otherwise reuses existing processed files.

### GNN architecture

| Setting | Accepted values / default | Purpose |
|---|---|---|
| `model.gnn.architecture` | **`gatv2` / `gine`**; `gatv2` | Attention-based message passing or edge-aware GINE. |
| `model.gnn.hidden_dim` | Positive integer; `128` | Hidden GNN width. |
| `model.gnn.output_dim` | Positive integer; `64` | Width of the learned road embedding. |
| `model.gnn.num_layers` | Positive integer; `2` | Number of GNN convolution layers. |
| `model.gnn.dropout` | 0 to below 1; `0.1` | Dropout between layers. |
| `model.gnn.activation` | PyG activation name; `relu` | Nonlinearity, e.g. `relu`, `leaky_relu`, `elu`, `gelu`, `silu`, `tanh`. |
| `model.gnn.normalization` | **`layer`, `layer_norm`, `batch`, `batch_norm`, `none`, `null`**; `layer` | Normalization between GNN layers. |
| `model.gnn.residual` | `true` / `false`; `true` | Add residual/skip connections. |
| `model.gnn.gatv2.heads` | Positive integer; `4` | Number of attention heads. |
| `model.gnn.gatv2.concat` | Boolean; `true` | Concatenate intermediate head outputs rather than average. |
| `model.gnn.gatv2.final_concat` | Boolean; `false` | Concatenate final attention heads. |
| `model.gnn.gatv2.attention_dropout` | 0 to below 1; `0.1` | Dropout of attention coefficients. |
| `model.gnn.gatv2.add_self_loops` | Boolean; `false` | Add GATv2 attention self-loops (separate from graph topology option). |
| `model.gnn.gine.train_eps` | Boolean; `true` | Learn GINE's skip/aggregation weighting. |
| `model.gnn.gine.eps` | Finite number; `0.0` | Initial GINE epsilon weight. |
| `model.gnn.gine.mlp.hidden_dim` | Positive integer; `128` | Hidden width in each GINE layer's MLP. |
| `model.gnn.gine.mlp.num_layers` | Positive integer; `2` | MLP depth. |
| `model.gnn.gine.mlp.activation` | PyG activation name; `relu` | MLP hidden activation. |
| `model.gnn.gine.mlp.dropout` | 0 to below 1; `0.1` | MLP dropout. |
| `model.gnn.gine.mlp.normalization` | PyG normalization name or `null`; `layer_norm` | MLP internal normalization, e.g. `layer_norm`/`batch_norm`. |
| `model.gnn.gine.mlp.plain_last` | Boolean; `true` | Leave last MLP layer without activation, normalization, and dropout. |

`gatv2` requires `hidden_dim % heads == 0` when concatenating intermediate heads across multiple layers. If `final_concat: true`, `output_dim % heads == 0` must also hold. `gine` needs an edge feature group. Activation names and GINE-MLP normalization strings are resolved by the installed PyG version, so they are **not fixed enum lists**; unsupported names fail at model construction.

### Useful graph combinations

| Goal | Good starting combination |
|---|---|
| Simple graph without FMM transition data | `gatv2`, `node_features: [x, degree_features]`, `edge_features: []`; enable `degree_features`. |
| Include observed road usage | Enable `edge_transition_count` with **training-only** matches; select `edge_transition_prob`. |
| Add positional context | Enable `random_walk_pe` or `laplacian_pe` and select its node feature; optional matching edge transform. |
| Use GINE | Set `architecture: gine` and provide `edge_features`, e.g. `[edge_transition_prob]`. |
| Compare pretrained vs. scratch fairly | Keep same graph YAML, ordered feature lists, GNN architecture, and diffusion settings across runs. |

---

## 4. Study YAML and HPC submission settings

A study YAML is a **launcher index**, not a replacement for experiment YAMLs. It assigns short names to experiments, identifies which joint experiments depend on pretraining, and supplies default Slurm resources. Selecting `--experiment joint_frozen --stage all` runs **that experiment plus its pretraining dependency**, not every experiment in the study.

```yaml
study:
  name: nyc_comparison
  graph_config: configs/experiments/hpc_smoke_test/nyc_graph.yaml
  experiments:
    pretrain_bpr: configs/experiments/hpc_smoke_test/nyc_link_pretrain_bpr.yaml
    joint_scratch: configs/experiments/nyc_gnn_comparison/joint_scratch.yaml
    joint_frozen:
      config: configs/experiments/nyc_gnn_comparison/joint_frozen.yaml
      pretrained_from: pretrain_bpr
    joint_finetune:
      config: configs/experiments/nyc_gnn_comparison/joint_finetune.yaml
      pretrained_from: pretrain_bpr
hpc:
  partition: gpuqs
  gres: gpu:a100:1
  cpus_per_task: 8
  memory: 64G
  time: "01:00:00"
  exclude_nodes: []
  mail_type: NONE
```

| Setting | Accepted values | Purpose |
|---|---|---|
| `study.name` | **Required** directory-safe string | Study identity and location of registered encoder references. |
| `study.graph_config` | Path; recommended for all studies, required when using `pretrained_from` | Confirms dependent pretraining uses the same graph YAML as joint training. |
| `study.experiments.<key>` | Path string **or** object containing `config` | Short name used by `--experiment`. Each key must be directory-safe. |
| `study.experiments.<key>.config` | Experiment YAML path | Path when an experiment entry is an object. |
| `study.experiments.<key>.pretrained_from` | Key of a **pretraining** experiment | Run/link that source GNN before a joint frozen/fine-tuned experiment. |
| `hpc.partition` | Slurm partition name | Queue to submit to. |
| `hpc.gres` | Slurm resource string, e.g. `gpu:a100:1` | GPU request. |
| `hpc.cpus_per_task` | CPU count | Requested CPU cores. |
| `hpc.memory` | Slurm memory string, e.g. `64G` | RAM requested. |
| `hpc.time` | Slurm time string, e.g. `01:00:00` | Job time limit. |
| `hpc.exclude_nodes` | List of hostnames, e.g. `[cs002]` | Skip known-unavailable nodes when needed. |
| `hpc.mail_user` | Email string (optional) | Destination for Slurm notifications. |
| `hpc.mail_type` | Slurm mail events, e.g. `NONE` or `BEGIN,END,FAIL` | When Slurm sends emails. |

The submission command can override study resource defaults with `SBATCH_PARTITION`, `SBATCH_GRES`, `SBATCH_CPUS_PER_TASK`, `SBATCH_MEM_PER_NODE`, `SBATCH_TIMELIMIT`, `SBATCH_EXCLUDE` and email options `URBAN_SYN_MAIL_USER`, `URBAN_SYN_MAIL_TYPE`. Keep personal addresses outside committed examples.

**Automatic encoder handoff:** For a joint entry with `pretrained_from`, set `model.road_encoder.initialization: pretrained`, use `trainable: false` or `true`, and set `checkpoint: null` in the joint YAML. The launcher submits pretraining first, waits for success (`afterok`), and passes that **specific pretraining job's** selected `encoder.pt` to joint training. You do not need to paste checkpoint paths or submit the second job manually.

**Reuse instead of retraining:** A previously selected encoder can also be used for a direct joint training run with the appropriate existing checkpoint selection. For commands and exact stage behavior, use the [Slurm guide](../hpc/urban_syn/README_SLURM.md).

---

## 5. Common configuration mistakes

| Symptom | Check |
|---|---|
| A selected graph feature is missing | Enable the producing transform **and rebuild** the graph cache. |
| Pretrained encoder refuses to load | Match GNN architecture, ordered node/edge feature groups, and feature widths with the checkpoint. |
| `pretrained_from` conflicts with checkpoint | Set the joint YAML's `model.road_encoder.checkpoint: null`; let the launcher inject it. |
| `--stage all` trains but generates nothing | Set `generation.enabled: true` and supply road geometry/length distribution. |
| Generation takes unexpectedly long | Set a small positive `generation.num_trajectories` during short tests; DDPM uses the configured diffusion steps. |
| Training is unexpectedly short | Remove `training.max_batches` cap by setting it to `null`. |
| Validation won't enable | Add `paths.trip_id_map` and keep `split_group_columns: [uid, tid]`. |
| GINE fails without edge attributes | Enable/select a computed edge feature group. |
| Too many GPU memory errors | Lower `training.batch_size`, `generation.batch_size`, or `model.diffusion.similarity_chunk_size`. |
| Results are logged to different MLflow groups | Use the same `tracking.experiment_name` and compatible storage backend for the runs being compared. |

For exact validations, the code is authoritative: `src/urban_syn/experiment/config.py`, `src/urban_syn/link_pretrain/config.py`, `src/urban_syn/gnn/config.py`, `src/urban_syn/graph/config.py`, and `hpc/urban_syn/resolve_submission.py`.
