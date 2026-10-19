# Baseline: DiffTraj

This directory contains the adapted DiffTraj baseline used for the NYC trajectory-generation experiments.

The implementation is based on the original DiffTraj repository and keeps the model (the conditional UNet and
the diffusion process) as close to the authors' code as practical. NYC data preparation and a few training settings are
added. Data preparation and generation each come as a notebook and as a script, so the two can be compared.

## Source

- Original repository: https://github.com/Yasoz/DiffTraj
- Paper: [DiffTraj: Generating GPS Trajectory with Diffusion Probabilistic Model](https://arxiv.org/abs/2304.11582) (NeurIPS 2023)
- Local baseline path: `baselines/diffusion/difftraj/repo/` (the authors' `readme.md` and `LICENSE` are kept there)

## Files added for this project

Everything else in `repo/` is the authors' code and notebooks (changed files are listed under NYC adaptation
notes).

- `Dockerfile`, `docker-compose.yml`, `.dockerignore`: the container.
- `hpc/train_nyc.sbatch`: HPC job that trains, then generates.
- `README.md`: this file.
- `repo/prepare_data.py`: builds the training and test files from the NYC moves.
- `repo/DiffTraj_DataFormat.ipynb`: the notebook version of `prepare_data.py`.
- `repo/generate.py`: the script version of `process/Traj_Generation.ipynb`.

## NYC adaptation notes

Changes in the authors' code are marked `# Change for DiffTraj (NYC):` with the original lines commented out
above them, so `grep -rn "Change for DiffTraj" repo/` lists them all.

Training (`main.py`, `utils/config.py`):

- Reads the NYC training files made by `prepare_data.py` (`traj.npy`, `head.npy`). They are already in the
  model's (N, 2, 256) layout, so the original slicing and axis swap are commented out.
- 256 points per trajectory instead of 200, batch size 256 instead of 1024, and 2,000 epochs instead of 200:
  the settings of the best NYC run.
- A checkpoint is saved every 50 epochs instead of every 10.
- GPU 0 by default unless `CUDA_VISIBLE_DEVICES` is already set (e.g. by Slurm).
- The run name is `NYC_moves_train`, so results go to `repo/DiffTraj/NYC_moves_train_steps=500_len=256_0.05_bs=256/`.

New scripts:

- `prepare_data.py` (script version of `DiffTraj_DataFormat.ipynb`) builds the training and test files from the
  MAT-Dataset NYC moves: one trajectory per move
  (all travel modes by default), resampled to 256 points, with DiffTraj's 8 conditions per trajectory (departure
  time slot, distance, duration, number of points, average step length, average speed, start and end cell on a
  16 x 16 grid). Normalisation statistics are computed on the training trips only.
- `generate.py` is a script version of the generation notebook, so generation can run in a batch job. It generates
  one trajectory per test trip, in the same order as the test conditions, so generated trajectory *i* can be
  compared with real test trajectory *i*.

Notebooks:

- `DiffTraj_DataFormat.ipynb` (in `repo/`, next to `prepare_data.py`): the notebook version of `prepare_data.py` (Colab or local), with extra plots of
  the NYC trajectories and the grid, and optional saves (all trajectories, augmented training trips).

`process/Traj_Generation.ipynb` (the authors' generation notebook, in the authors' `process/` folder, adapted for
NYC and Colab):

- Generates the same way as `generate.py`: the trained NYC model (in evaluation mode), one trajectory per test
  condition (`head_test.npy`) in batches and in order, rescaled with the training-only statistics
  (`traj_stats.npy`), with the same sampling settings and seed. The original generated only one batch from the
  authors' example conditions with Chengdu's values.
- Saves the generated trajectories like `generate.py`, in a `notebook/` subfolder of the run's `Files/<timestamp>/`
  folder, and compares them with `generate.py`'s output for the same run if it exists.
- Adds plots of the real training trajectories, the generated trajectories (with and without the 16 x 16 grid), a
  check for points outside NYC, and the training loss.

The authors' other notebook (`process/Traj2parquet.ipynb`, Chengdu data conversion) is kept unchanged.

## Data-split policy

- **Train:** all trips except the test trips. Used for training and for the normalisation statistics.
- **Test:** the same test trips as Diff-RNTraj and TS-TrajGen, read from the shared list `fmm_scripts/nyc_test.csv`
  (`uid`, `tid`; not copied into this folder). Every move of a test trip goes to the test set. `generate.py` generates one trajectory per test trip for evaluation.
- **No validation set:** like the original code, training has no validation split; checkpoints are simply saved
  every 50 epochs.

## Data and outputs

All data and outputs are gitignored; only the code is committed.

- **Raw input:** the MAT-Dataset NYC moves (`data/nyc_output_tabular/output/enriched_moves.parquet` in the
  URBAN-SYN root) and the test list (`fmm_scripts/nyc_test.csv`, gitignored).
- **Made by `prepare_data.py` (in `repo/`):** the training trajectories and conditions (`traj.npy`, `head.npy`),
  the test conditions (`head_test.npy`), the training statistics (`traj_stats.npy`), the real test trajectories
  in the same order (`real_trajectories_test.npy`), and a summary (`prepare_data_info.json`).
- **Training output (in `repo/DiffTraj/<run>/`):** checkpoints in `models/<timestamp>/unet_<epoch>.pt`, losses in
  `results/`, logs in `logs/`, and a copy of the config and model code in `Files/<timestamp>/`.
- **Generated trajectories:** `Files/<timestamp>/synthetic_trajectories.npy` (latitude/longitude per point), with
  `synthetic_index.npy` (which condition each one came from) and `generation_info.json`.

## Container

One image (`difftraj:latest`): Python 3.11 with torch 2.7.1 (CUDA 11.8, the same build as the Diff-RNTraj
image), numpy, pandas, pyarrow, matplotlib, tqdm and colored 1.x (for the authors' logger).
`docker-compose.yml` mounts the whole URBAN-SYN folder and starts in `repo/`, so code changes don't need a
rebuild. GPU-enabled through `gpus: all`. On the HPC the same image runs as an Apptainer `.sif`.

## How to run

1. **Data preparation: run it locally.**
2. **Training and generation: on the HPC** with `hpc/train_nyc.sbatch`, or locally if you prefer.

### 1. Data preparation (local, Docker Desktop running)

From this folder:
```bash
docker compose build                                                  # once
docker compose run --rm difftraj python prepare_data.py               # all travel modes
docker compose run --rm difftraj python prepare_data.py --modes 2 3 6 # only bus, car, taxi
```
Other options: `--min-points N` (drop moves with fewer GPS points, default 2) and
`--parquet .../traj_cleaned.parquet` (one trajectory per user track instead of per move).

The same steps are in `repo/DiffTraj_DataFormat.ipynb` (notebook version). It runs in Colab (it mounts Google
Drive and reads the data and test files from there) or locally from `repo/` (it reads the parquet from
`URBAN-SYN/data/`). Its settings are at the top: `DATA_SOURCE`, `MODES` and `MIN_POINTS`. For the test split
it reads the same `fmm_scripts/nyc_test.csv` (in Colab, a copy in the Drive folder). It also prints the training statistics that `Traj_Generation.ipynb` uses.

### 2. Train and generate

**On the HPC:** submit `hpc/train_nyc.sbatch` from this folder, with the image as `difftraj.sif` next to it and
the files from step 1 in `repo/`. The job trains, then runs `generate.py` on the test conditions;
`GENERATE_ONLY=1` skips training and `MODEL=...` picks a checkpoint. Usage is in the comments at the top of the
script.

**Locally** (from this folder):
```bash
docker compose run --rm difftraj python main.py                       # train (settings in repo/utils/config.py)
docker compose run --rm difftraj python generate.py                   # generate from the newest checkpoint
docker compose run --rm difftraj python generate.py --model "DiffTraj/<run>/models/<timestamp>/unet_2000.pt"
docker compose run --rm difftraj python generate.py --limit 50        # quick test
```

### Two ways to generate

- **`generate.py` (script, used by the HPC job):** loads a checkpoint (the newest one unless `--model` is given),
  generates one trajectory per row of `head_test.npy` (`--head` to use other conditions) and rescales them with
  `traj_stats.npy`. Output goes to the run's `Files/<timestamp>/` folder.
- **`process/Traj_Generation.ipynb` (Colab notebook):** open it in Colab or Jupyter from the `process/` folder
  (it adds the parent folder, `repo/`, to the import path and reads the data files from there). Set `path`, `file`
  and `checkpoint` in the model cell to your run. It does the same as `generate.py` (test conditions,
  `traj_stats.npy`, same settings and seed), saves to the run's `Files/<timestamp>/notebook/` folder, compares the
  result with `generate.py`'s output, and plots it.
