# NYC full-training YAMLs

These are **explicit, editable starting configurations** for full URBAN-SYN experiments, not tuned or guaranteed-to-finish training budgets. They reuse the uploaded NYC dataset paths, road features, model sizes, optimizer, scheduler, and loss choices. Adjust epochs, batch size, wall time, and generation count after profiling the actual data and GPU.

| File | Purpose |
|---|---|
| `nyc_graph.yaml` | Shared road graph, training-only FMM transition features, and GATv2 encoder; built once/cached. |
| `link_pretrain_bpr.yaml` / `link_pretrain_bce.yaml` | Directed link-prediction GNN pretraining; only the objective and experiment labels differ. |
| `joint_scratch.yaml` | Diffusion and GNN trained end to end from random initialization. |
| `joint_frozen.yaml` | Pretrained GNN weights remain frozen while diffusion trains. |
| `joint_finetune.yaml` | Pretrained GNN and diffusion weights train together. |
| `study.yaml` | BPR-encoder comparison: both pretrain options, scratch, BPR-frozen, BPR-finetune. |
| `study_bce.yaml` | Same comparison, but frozen/fine-tune use the BCE-pretrained encoder. |

## Copy and submit from the repository root on HPC

```bash
# Inspect the actual inputs/resources without running training.
bash hpc/urban_syn/submit.sh --study configs/experiments/nyc_full_training/study.yaml --experiment pretrain_bpr --stage all --validate-only

# Compare one condition per submission. For pretrained joint experiments,
# --stage all launches pretraining then a dependent joint job automatically.
bash hpc/urban_syn/submit.sh --study configs/experiments/nyc_full_training/study.yaml --experiment pretrain_bpr --stage all
bash hpc/urban_syn/submit.sh --study configs/experiments/nyc_full_training/study.yaml --experiment pretrain_bce --stage all
bash hpc/urban_syn/submit.sh --study configs/experiments/nyc_full_training/study.yaml --experiment joint_scratch --stage all
bash hpc/urban_syn/submit.sh --study configs/experiments/nyc_full_training/study.yaml --experiment joint_frozen --stage all
bash hpc/urban_syn/submit.sh --study configs/experiments/nyc_full_training/study.yaml --experiment joint_finetune --stage all

# For BCE-based pretrained joint comparisons, use the other study:
bash hpc/urban_syn/submit.sh --study configs/experiments/nyc_full_training/study_bce.yaml --experiment joint_frozen --stage all
```

**Important:** Each command above submits the named experiment, **not** every experiment in the study. In particular, submitting `pretrain_bpr` manually and then `joint_frozen --stage all` can train BPR twice. To reuse an already-published encoder for a study, choose `--stage train` rather than `--stage all` (but `train` does not include generation). Alternatively use `--encoder-checkpoint` for a specific existing encoder. See `hpc/urban_syn/README_SLURM.md` for all submission options.

## Expected input file locations (relative to URBAN-SYN repo root)

```text
fmm_scripts/
├── data/
│   ├── nyc_15s_fmm_trip_id_map.csv
│   └── road_network/nyc.gpkg
└── outputs/
    ├── nyc_15s_enriched_moves/nyc_15s_train_fmm_matches.csv
    └── urban_syn_ready/
        ├── nyc_15s_enriched_moves_train_prepared_road_sequences.parquet
        ├── nyc_15s_enriched_moves_test_prepared_road_sequences.parquet
        └── nyc_15s_enriched_moves_train_length_distribution.npy
```

The training-only FMM CSV is needed by `graph.transforms.edge_transition_count`. Test road sequences are not used for creating that graph feature. The PyG cache at `outputs/pyg/nyc_full_training/processed/` and experiment outputs at `outputs/experiments/` are *created by the pipeline* and need not be uploaded.

## Differences from the older YAML examples

- Removed outdated `model.diffusion` and `training` blocks from the **graph** YAML. The graph file now contains only graph/encoder settings.
- Updated `road_encoder.mode` + explicit `initialization`, `trainable`, `checkpoint` to support scratch/frozen/finetune. The older joint-only mode is retained for compatibility.
- Replaced obsolete `gnn_embedding_probe_size` with `gnn_embedding_sample_size` and includes the current epoch-level embedding diagnostic options.
- Removed smoke caps (`max_batches: 10`, `epochs: 4`) and retained your proposed **1000 joint epochs**, **100 pretraining epochs**, and 256 training batch size as *initial budgets*, not validated optimal values.
- Enabled **grouped training-only validation** for selecting the joint best checkpoint, and `save_last` for recovery. The supplied no-spatial-loss objective is preserved: `hard_connectivity` is diagnostic-only (`include_in_total: false`).
- Enabled full generation (`num_trajectories: null`, training-distribution count) after joint training; this may substantially extend runtime. Set `generation.enabled: false` if generating separately, or choose a positive count to limit it.
- Uses one graph YAML for all conditions and a separate cache root to avoid silently reusing graph/preprocessor artifacts from smoke runs.

## Compatibility and resource warnings

- These files use the **current refactored code**, especially study checkpoint injection (`pretrained_from`), the pretraining YAML schema, and current experiment tracking names. Do not run them on an older branch without updating the code.
- The pretrained encoder must match the GNN architecture, input feature selection, and **feature order** in this graph YAML. Checkpoint source is pinned by the study launcher, not picked from the newest arbitrary run.
- Graph betweenness, Laplacian features, and structural-hole constraint are retained from your example, but may be expensive to build. Review graph build and Slurm logs before long submissions.
- The HPC A100 queue, GPU memory availability, and `3-00:00:00` walltime are assumptions to confirm against the live cluster. A 1000-epoch job is not guaranteed to finish within the allocation.
- `tracking.storage_scope: shared` makes MLflow runs easy to compare, but concurrent writers on network-mounted SQLite can lock; use `job` storage or a managed tracking server if needed.
- No actual training or HPC submissions are performed by these YAML templates.

Full per-attribute descriptions are in `configs/README.md`. These files intentionally include supported options explicitly rather than relying on Python defaults.
