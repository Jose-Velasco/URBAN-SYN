# Run URBAN-SYN on HPC (Slurm + Apptainer)

**Slurm** allocates CPU/GPU resources. **Apptainer** runs the container's Python environment. Submit from the **repository root on an HPC login node**; the training itself runs on a compute node.

See [Container setup](README.md) if `~/containers/urban-syn.sif` has not been built yet. See [Experiment YAMLs](../../configs/experiments/README.md) to change training settings and [Graph YAMLs](../../configs/graph/README.md) to change road features.

## 1. Files needed on HPC

Keep the same relative paths used by the YAMLs. A typical NYC layout looks like this:

```text
~/.../URBAN-SYN/                         # Repository root (location is up to you)
├── configs/
│   └── experiments/
│       └── hpc_smoke_test/
│           ├── study.yaml              # Which experiment to run; HPC defaults
│           ├── nyc_graph.yaml          # Road features, transforms, GNN settings
│           ├── nyc_link_pretrain.yaml  # BCE pretraining configuration
│           ├── nyc_link_pretrain_bpr.yaml
│           ├── joint_scratch.yaml
│           ├── joint_frozen.yaml
│           └── joint_finetune.yaml
├── fmm_scripts/
│   ├── data/
│   │   ├── nyc_15s_fmm_trip_id_map.csv
│   │   └── road_network/
│   │       └── nyc.gpkg
│   └── outputs/
│       ├── urban_syn_ready/
│       │   ├── nyc_15s_enriched_moves_train_prepared_road_sequences.parquet
│       │   ├── nyc_15s_enriched_moves_test_prepared_road_sequences.parquet
│       │   └── nyc_15s_enriched_moves_train_length_distribution.npy
│       └── nyc_15s_enriched_moves/
│           └── nyc_15s_train_fmm_matches.csv
├── hpc/urban_syn/
│   ├── submit.sh
│   ├── run_urban_syn.sbatch
│   └── ...
├── src/urban_syn/                      # Model and graph code
└── outputs/                            # Created by runs
    ├── pyg/nyc_gnn/processed/          # Graph cache (created if absent)
    └── experiments/                    # Checkpoints, generated GPS, MLflow

~/containers/urban-syn.sif             # Apptainer image; stored outside repo
```

The names and locations in `configs/experiments/hpc_smoke_test/` above are one **example**. If your `study.yaml` points to experiment YAMLs in another directory (such as `nyc_gnn_comparison/`), keep those files **at the referenced paths**, rather than moving them to match this picture.

| Input | Used for |
|---|---|
| Prepared **train** road-sequence Parquet | Joint training; required |
| Road GeoPackage (`nyc.gpkg`) | Building graph, reconstructing GPS during generation; required |
| Training length-distribution `.npy` | Sampling generated trajectory lengths; needed for generation |
| Training FMM matches CSV (`cpath` column) | Only when the graph uses `edge_transition_count` |
| Trip-ID mapping CSV | Grouped validation; may be required by the experiment YAML |
| Prepared **test** sequence Parquet | Held-out data when configured; not a substitute for training data |
| Graph cache (`road_graph.pt`, `preprocessor.joblib`) | Output; generated automatically if missing |

These are *prepared* inputs from the existing FMM/data workflow. This pipeline does not download road networks, map-match raw GPS, or prepare the train/test split.

To upload a missing file, copy it from your workstation to the **same path relative to the HPC repository root**. For example, after replacing the remote login and repository location:

```bash
scp fmm_scripts/data/road_network/nyc.gpkg \
  USER@coe-hpc1.sjsu.edu:~/PATH/TO/URBAN-SYN/fmm_scripts/data/road_network/
```

Use the same pattern for the train/test Parquet files, trip map, length distribution, and optional training FMM CSV. Create destination folders first if necessary. Before submitting, compare `paths.*` and the graph `dataset.network_path` / `graph.transforms.edge_transition_count.match_csv` against what you uploaded.

## 2. What each experiment runs

A study YAML lists experiments. `--experiment` selects **one**; `--stage all` runs that experiment's stages, **not the entire study**.

| Study key | Workflow |
|---|---|
| `joint_scratch` | Initialize GNN randomly → joint diffusion training → generation if enabled |
| `joint_frozen` | Pretrain GNN → select encoder → freeze GNN → diffusion training → generation |
| `joint_finetune` | Pretrain GNN → select encoder → train GNN and diffusion together → generation |
| `pretrain_bpr` / `pretrain_bce` | Train/test the directed road-link GNN; save `encoder.pt` |

For pretrained joint runs, the launcher submits **two jobs**. The second waits for the first to succeed (`afterok`) and loads the encoder from that **specific** pretraining job; you don't need to edit checkpoint paths or manually wait. For generation, the joint YAML must have:

```yaml
generation:
  enabled: true
  checkpoint: null       # Use best checkpoint from this joint training run
  checkpoint_choice: best
  num_trajectories: 8   # Small amount for a quick run; choose a larger count later
```

## 3. Copy-paste HPC examples

These commands use an A100 allocation and exclude `cs002`, matching the tested HPC setup. They do **not** require exported shell variables or editing paths if the study YAML and inputs use the layout above. Run them **one at a time** to keep the outputs easy to follow.

### From scratch (train + generate)

```bash
SBATCH_EXCLUDE=cs002 \
SBATCH_PARTITION=gpuqs \
SBATCH_GRES=gpu:a100:1 \
SBATCH_TIMELIMIT=01:00:00 \
SBATCH_MEM_PER_NODE=64G \
SBATCH_CPUS_PER_TASK=8 \
bash hpc/urban_syn/submit.sh \
  --study configs/experiments/hpc_smoke_test/study.yaml \
  --experiment joint_scratch \
  --stage all
```

### Pretrained, frozen GNN (pretrain + joint + generate)

```bash
SBATCH_EXCLUDE=cs002 \
SBATCH_PARTITION=gpuqs \
SBATCH_GRES=gpu:a100:1 \
SBATCH_TIMELIMIT=01:00:00 \
SBATCH_MEM_PER_NODE=64G \
SBATCH_CPUS_PER_TASK=8 \
bash hpc/urban_syn/submit.sh \
  --study configs/experiments/hpc_smoke_test/study.yaml \
  --experiment joint_frozen \
  --stage all
```

### Pretrained, fine-tuned GNN (pretrain + joint + generate)

```bash
SBATCH_EXCLUDE=cs002 \
SBATCH_PARTITION=gpuqs \
SBATCH_GRES=gpu:a100:1 \
SBATCH_TIMELIMIT=01:00:00 \
SBATCH_MEM_PER_NODE=64G \
SBATCH_CPUS_PER_TASK=8 \
bash hpc/urban_syn/submit.sh \
  --study configs/experiments/hpc_smoke_test/study.yaml \
  --experiment joint_finetune \
  --stage all
```

**Email notifications (optional):** Add `URBAN_SYN_MAIL_USER=your.email@example.edu` and `URBAN_SYN_MAIL_TYPE=BEGIN,END,FAIL` before `bash` in any command. Replace the example address. The `SBATCH_*` settings override the study YAML `hpc:` defaults. Increase `SBATCH_TIMELIMIT` for long runs; GPU time limits are not controlled by `training.epochs`.

To run a pretraining-only experiment, change `--experiment` to `pretrain_bpr` or `pretrain_bce`. The standalone syntax is still supported: `bash hpc/urban_syn/submit.sh all configs/experiments/nyc_joint.yaml`.

## 4. Reuse a checkpoint or generate without retraining

```bash
# Reuse a previously registered successful encoder; no pretraining job.
bash hpc/urban_syn/submit.sh \
  --study configs/experiments/hpc_smoke_test/study.yaml \
  --experiment joint_frozen --stage train
```

For an **exact** encoder, add `--encoder-checkpoint path/to/encoder.pt` to a joint submission; this skips automatic pretraining. When using `study.pretrained_from`, keep `model.road_encoder.checkpoint: null` in the joint YAML to avoid competing checkpoint selections.

To generate only, set `generation.checkpoint` in the joint YAML to a compatible **joint** `best.pt`, then use `--stage generate`. Pretraining `encoder.pt` cannot generate trajectories on its own.

## 5. Watch jobs and inspect outputs

```bash
squeue -u "$USER"                               # Pending/running jobs
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed  # Including finished jobs
```

```text
hpc/urban_syn/logs/
  urban-syn-JOB_ID.out/.err          # Standalone job
  urban-syn-pretrain-JOB_ID.out/.err # Automatic pretraining
  urban-syn-joint-JOB_ID.out/.err    # Dependent joint training
  job-JOB_ID.txt                    # Summary and exit code
  gpu-JOB_ID.csv                    # GPU resource monitoring
  resources-JOB_ID.log              # CPU/memory monitoring
```

```bash
tail -n 40 hpc/urban_syn/logs/urban-syn-joint-JOB_ID.out
tail -n 40 hpc/urban_syn/logs/urban-syn-joint-JOB_ID.err
find outputs/experiments -path '*/generated/*.parquet' -type f
```

Each run saves files under `outputs/experiments/<experiment.name>/<run_id>/`, including checkpoints, metrics, and generated GPS if enabled. MLflow is stored under `outputs/experiments/mlflow/`; see [MLflow README](README_MLFLOW.md) to view it locally.

## 6. If something fails

| Symptom | Check |
|---|---|
| Joint job is pending with `(Dependency)` | Normal until pretraining succeeds. Check pretraining job status. |
| Joint job never starts | `afterok` requires pretraining to exit successfully; inspect pretraining `.err` and `job-JOB_ID.txt`. |
| Missing file / invalid YAML path | Check paths against the tree above, including the training-only FMM CSV if enabled. |
| GNN pretrained checkpoint doesn't load | Graph feature order and GNN architecture must match the pretrained encoder. |
| CUDA error | Check assigned GPU and node in `job-JOB_ID.txt`; request a working node. |
| Run completed training but Slurm reports `FAILED` | Inspect `.err` and the last `exit_code`: post-training checkpoint bookkeeping can also fail. |
| Need to resume | Set `training.checkpoint.resume_from` to a compatible `last.pt` that was saved by an earlier run. |

Use `--validate-only` for file/config checks and `--dry-run` to preview planned stages before allocating a GPU. These checks don't establish that the model's generated trips are realistic; that is handled separately by the shared evaluation workflow.
