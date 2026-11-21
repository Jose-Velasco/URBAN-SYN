# Joint GNN + Diff-RNTraj Pipeline

This pipeline learns to generate road-based trajectories using a **road GNN** (representations of connected roads) and **Diff-RNTraj** (a diffusion model that generates road sequences and positions). It outputs a GPS Parquet file.

```text
Prepared map-matched road sequences   Road network + graph YAML
                 \                        /
                  \                  PyG graph
                   \                    /
                    Joint model training
                           |
                    Joint checkpoint
                           |
                    DDPM generation
                           |
                    GPS Parquet file
```

You can start the GNN randomly, freeze a pretrained GNN, or fine-tune a pretrained GNN. Pretraining uses the separate [road-link pipeline](../link_pretrain/README.md).

## 1. Required inputs

The pipeline **does not** download GPS data, run FMM map matching, or create a train/test split. Prepare these before training:

| Input | Used for |
|---|---|
| Prepared training road sequences (`.parquet`) | Train the joint model |
| Road network (`.gpkg`) | Build the PyG graph and reconstruct GPS positions |
| Graph YAML | Road features, transforms, and GNN architecture |
| Training trajectory-length distribution (`.npy`) | Choose generated trajectory lengths |
| Prepared test sequences / trip ID map | Optional validation or other downstream work |

Relative paths in YAML files start at the **repository root**, not the YAML's folder. The graph cache (`road_graph.pt`, `preprocessor.joblib`) is created if missing. **After changing road features or GNN settings, rebuild the cache** rather than reusing stale graph data.

### Preparing NYC road sequences

If the map matching is already done but the training/test road-sequence Parquets are missing, run the existing split and conversion tools from `fmm_scripts/`:

```bash
uv run split_fmm_matches.py \
  --config ./config/nyc.yaml \
  --fmm_matches ./outputs/nyc_15s_fmm_match.csv \
  --trip_id_map ./data/nyc_15s_fmm_trip_id_map.csv \
  --test_trips ./data/test_trips.csv \
  --output_dir ./outputs/nyc_15s_enriched_moves \
  --prefix nyc_15s

uv run python -m urban_syn_extension.convert_fmm_sequences \
  --train_fmm_matches ./outputs/nyc_15s_enriched_moves/nyc_15s_train_fmm_matches.csv \
  --test_fmm_matches ./outputs/nyc_15s_enriched_moves/nyc_15s_test_fmm_matches.csv \
  --output_dir ./outputs/urban_syn_ready \
  --prefix nyc_15s_enriched_moves
```

Keep the same held-out user/trip split when comparing models. Use the **training** FMM matches for graph transition-count features, not test data. FMM trip `id=0` can indicate a row with unknown trip identity; road `edge_id=0` is a valid road ID.

The converter writes `*_train_prepared_road_sequences.parquet`, `*_test_prepared_road_sequences.parquet`, and `*_train_length_distribution.npy`. Set their actual paths in your experiment YAML.

## 2. Choose a training mode

| Mode | GNN starts with | GNN updated while training diffusion? |
|---|---|---|
| Scratch | Random weights | Yes |
| Frozen | Pretrained `encoder.pt` | No |
| Fine-tuned | Pretrained `encoder.pt` | Yes |

In the joint experiment YAML:

```yaml
model:
  road_encoder:
    initialization: pretrained    # random for scratch
    trainable: false             # true for fine-tuned or scratch
    checkpoint: null             # Filled by study on HPC
```

For a pretrained experiment, the HPC study links the joint YAML to a pretraining YAML using `pretrained_from`. A single `--stage all` command runs pretraining and then joint diffusion automatically. It **does not** run all study experiments.

```bash
bash hpc/urban_syn/submit.sh \
  --study configs/experiments/hpc_smoke_test/study.yaml \
  --experiment joint_frozen --stage all
```

Change the experiment key to `joint_scratch` or `joint_finetune` for the other strategies. See [Slurm guide](../../../hpc/urban_syn/README_SLURM.md) for resource overrides and checkpoint reuse.

## 3. Run locally

From the repository root:

```bash
# Check required files and settings.
uv run python -m urban_syn.experiment all \
  --config configs/experiments/nyc_joint.yaml --validate-only

# Train and optionally generate.
uv run python -m urban_syn.experiment all \
  --config configs/experiments/nyc_joint.yaml
```

| Stage | Action |
|---|---|
| `validate` | Check configuration and required source files |
| `prepare-graph` | Reuse or build PyG graph cache |
| `train` | Train joint GNN/diffusion model |
| `generate` | Load a joint checkpoint and generate GPS trajectories |
| `export-eval` | Prepare a manifest for the existing shared evaluator; does not calculate quality metrics |
| `all` | Run preparation and training; add enabled generation and optional evaluator handoff |

A `generate`-only run **does not retrain**. Set `generation.checkpoint` to an existing joint `best.pt` in the YAML first. For `all`, use `generation.checkpoint: null` so generation uses the selected checkpoint from that run.

## 4. Generated files

To include generation after training:

```yaml
generation:
  enabled: true
  checkpoint: null
  checkpoint_choice: best
  output_name: nyc_generated_gps.parquet
  num_trajectories: 8
  sampler:
    name: ddpm
```

Generation needs a compatible joint checkpoint, the road GeoPackage, and the training length distribution. Keep the full generation settings (such as `batch_size` and `road_layer`) in your actual YAML.

```text
outputs/experiments/<experiment_name>/<run_id>/
  checkpoints/best.pt
  logs/
  metrics/
  generated/
    nyc_generated_gps.parquet
    generation_manifest.json
```

Generated GPS rows include `trajectory_id`, `step`, road identifiers, position rates, longitude, and latitude. Internal `model_road_id` values reserve zero for padding; real zero-based road IDs are represented as `edge_id + 1` in the model.

**Generating a file does not establish that roads connect or trips look realistic.** The existing shared evaluation pipeline handles trajectory quality separately; this trainer does not replace it.

For a complete list of training/MLflow settings, see [Experiment YAML guide](../../../configs/experiments/README.md). For pretraining checkpoints, see [Road GNN pretraining](../link_pretrain/README.md).
