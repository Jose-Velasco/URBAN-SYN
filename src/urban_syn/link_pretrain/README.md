# Road GNN Pretraining

Before joint diffusion training, the road GNN can learn **which directed roads connect to which**. This gives the joint model pretrained road embeddings instead of starting from random weights.

Pretraining works on the existing **PyG road graph**, not raw GPS trajectories. It is separate from the diffusion trainer, and it does not run FMM map matching.

## 1. How it works

1. Divide directed road links into train, validation, and test sets.
2. Train a GNN and link scorer to distinguish real directed connections from sampled non-connections.
3. Select the checkpoint with the lowest validation loss, then test it.
4. Export **`encoder.pt`** for use in joint diffusion training.

The pretraining YAML selects either `bce` (binary link classification) or `bpr` (ranking real links above negative examples). Negative examples can come from nearby graph hops or a mixture of nearby and uniform candidates. Existing roads are excluded from negatives.

Validation/test links are **not** used for GNN message passing. The same validation comparisons are reused during a run. A final test evaluates the selected encoder without influencing checkpoint selection.

## 2. Run pretraining

Locally, from the repository root:

```bash
uv run python -m urban_syn.link_pretrain all \
  --config configs/experiments/hpc_smoke_test/nyc_link_pretrain_bpr.yaml
```

On HPC, using a study:

```bash
bash hpc/urban_syn/submit.sh \
  --study configs/experiments/hpc_smoke_test/study.yaml \
  --experiment pretrain_bpr --stage all
```

Change the study experiment to `pretrain_bce` to run the BCE configuration. For a quick check, add `--dry-run` or `--validate-only`. To test an existing checkpoint locally without retraining:

```bash
uv run python -m urban_syn.link_pretrain test \
  --config configs/experiments/hpc_smoke_test/nyc_link_pretrain_bpr.yaml \
  --run-dir outputs/experiments/EXPERIMENT_NAME/RUN_ID
```

Replace `EXPERIMENT_NAME/RUN_ID` with the actual saved run directory.

## 3. Important settings

| YAML setting | Controls |
|---|---|
| `paths.graph_config` | Road graph and GNN architecture shared with the joint model |
| `pretraining.objective` | `bpr` or `bce` |
| `pretraining.graph_split` | Train/validation/test proportions and disjoint training supervision |
| `pretraining.negatives` | Local/mixed negative samples, number per positive, sampling distance |
| `pretraining.decoder` | Link scorer size and regularization |
| `training.epochs`, `training.max_batches` | Training duration; `max_batches: null` runs all batches per epoch |
| `training.early_stopping`, `training.checkpoint` | Stop rules and checkpoint saving |
| `runtime.num_workers` | CPU workers that prepare training samples |

`pretraining.negatives.evaluation_chunk_size` controls how many held-out **positive links** are processed at once when preparing evaluation samples, not the training batch size.

## 4. Checkpoints and reuse

```text
outputs/experiments/<pretraining_experiment>/<run_id>/
  checkpoints/
    best.pt        # Full GNN + link scorer training checkpoint
    encoder.pt     # GNN only, for joint diffusion
    last.pt        # Optional, for resuming
  metrics.jsonl
  metrics.parquet
  test_metrics.json
```

**Use `encoder.pt` for joint training.** `best.pt` includes the pretraining link scorer and optimizer; it is not the joint model checkpoint.

A study connects pretraining and joint training like this:

```yaml
study:
  experiments:
    pretrain_bpr: configs/experiments/hpc_smoke_test/nyc_link_pretrain_bpr.yaml
    joint_frozen:
      config: configs/experiments/nyc_gnn_comparison/joint_frozen.yaml
      pretrained_from: pretrain_bpr
```

Submitting `joint_frozen --stage all` on HPC automatically runs pretraining first and passes that job's selected encoder to joint training. `joint_finetune` works the same way but allows the GNN to update during joint training. You do **not** have to enter the encoder path manually for this workflow.

The joint model checks that the GNN architecture and ordered input features match the exported encoder. A different road **count** is okay, but incompatible feature definitions are not.

See [Slurm](../../../hpc/urban_syn/README_SLURM.md) for one-command experiments, job failures, and checkpoint reuse. See [Joint pipeline](../experiment/README.md) for what happens after pretraining.
