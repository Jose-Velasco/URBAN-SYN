# URBAN-SYN Dev Setup

This project uses:
- **VS Code Dev Containers** for environment setup (**requires** docker installed)
- **uv** for Python dependency management

---

## 🚀 Quick Start (VS Code Dev Containers)

1. Open repo in VS Code  
2. Press `Ctrl+Shift+P` → **Reopen in Container**  
3. Wait for setup to finish  

That’s it ✅

---

## What happens automatically

- Dev container builds environment
- `uv sync --frozen` runs
- `.venv/` is created with all dependencies

---

## Run Python scripts

```bash
uv run python your_script.py
```

## Notes
- .venv/ is auto-created (do not commit it)
- Always use uv run instead of activating manually
- Keep dependencies reproducible via uv

## URBAN-SYN Model: Joint GNN + Diff-RNTraj

Our current trajectory-generation model combines a **road-network GNN** with **Diff-RNTraj**. The GNN learns road-segment embeddings, while the diffusion model learns to generate sequences of roads and positions along them.

The pipeline handles graph preparation, joint training, checkpointing, and DDPM trajectory generation.

**Before running:** Prepare the map-matched training and test sequences using our FMM workflow. The pipeline can automatically build the PyG road graph if its cache is missing, but it does not perform FMM map matching or create the train/test split.

### Quick start

From the repository root, using the project's Python environment:

```bash
# Check the required inputs.
uv run python -m urban_syn.experiment all \
  --config configs/experiments/nyc_joint.yaml \
  --validate-only

# Run training and generation (when enabled).
uv run python -m urban_syn.experiment all \
  --config configs/experiments/nyc_joint.yaml
```

For generation without retraining, use the `generate` stage with an existing checkpoint specified in the experiment YAML.

On SJSU HPC, submit the pipeline through Slurm:

```bash
bash hpc/urban_syn/submit.sh all \
  configs/experiments/nyc_joint.yaml
```

### Configuration and documentation

- [Experiment YAML guide](configs/experiments/README.md): diffusion, training, losses, validation, checkpoints, generation, and MLflow settings.
- [Graph/GNN YAML guide](configs/graph/README.md): road features, graph transforms, GNN architectures, and supported values.
- [Pipeline README](src/urban_syn/experiment/README.md): preprocessing, pipeline stages, model behavior, and outputs.
- [HPC container guide](hpc/urban_syn/README.md) and [Slurm guide](hpc/urban_syn/README_SLURM.md): Docker, Apptainer, and HPC execution.
- [MLflow guide](hpc/urban_syn/README_MLFLOW.md): experiment tracking, metrics, and viewing HPC runs locally.

Each training run creates its own output directory under `outputs/experiments/`. Metrics, checkpoints, and generated trajectories are saved even when MLflow tracking is disabled.

## Baselines

This project includes baseline implementations adapted from existing research repositories.

### LSTM-TrajGAN
- Source: https://github.com/GeoDS/LSTM-TrajGAN
- Paper: [LSTM-TrajGAN: A Deep Learning Approach to Trajectory Privacy Protection](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.GIScience.2021.I.12)

The code for this baseline is included under:
`baselines/gan/lstm_trajgan/`

See `baselines/gan/lstm_trajgan/NOTES.md` for details.

### TS-TrajGen
- Source: https://github.com/WenMellors/TS-TrajGen
- Paper: [Continuous Trajectory Generation Based on Two-Stage GAN](https://arxiv.org/abs/2301.07103)

The code for this baseline is included under:
`baselines/gan/ts_trajgen/`

See `baselines/gan/ts_trajgen/README.md` for details.

### Diff-RNTraj
- Source: https://github.com/wtl52656/Diff-RNTraj
- Paper: [Diff-RNTraj: A Structure-aware Diffusion Model for Road Network-constrained Trajectory Generation](https://arxiv.org/abs/2402.07369)

The code for this baseline is included under:
`baselines/diffusion/diff_rntraj/`

See `baselines/diffusion/diff_rntraj/README.md` for details.

### DiffTraj
- Source: https://github.com/Yasoz/DiffTraj
- Paper: [DiffTraj: Generating GPS Trajectory with Diffusion Probabilistic Model](https://arxiv.org/abs/2304.11582)

The code for this baseline is included under:
`baselines/diffusion/difftraj/`

See `baselines/diffusion/difftraj/README.md` for details.

### MobilityGPT
- Source: https://github.com/ammarhydr/MobilityGPT
- Paper: [MobilityGPT: Enhanced Human Mobility Modeling with a GPT model](https://arxiv.org/abs/2402.03264)

The code for this baseline is included under:
`baselines/transformer/MobilityGPT/`

See `baselines/transformer/MobilityGPT/README.md` for details.
