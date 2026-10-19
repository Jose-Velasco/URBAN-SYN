# Baseline: Diff-RNTraj

This directory contains the adapted Diff-RNTraj baseline used for the NYC trajectory-generation experiments.

The implementation is based on the original Diff-RNTraj repository and keeps the model and training code as close
to the authors' version as practical. NYC is added as a new dataset next to Porto and Chengdu, and a few fixes make
the code work at NYC's scale and with current PyTorch.

## Source

- Original repository: https://github.com/wtl52656/Diff-RNTraj
- Paper: [Diff-RNTraj: A Structure-aware Diffusion Model for Road Network-constrained Trajectory Generation](https://arxiv.org/abs/2402.07369) (IEEE TKDE)
- Local baseline path: `baselines/diffusion/diff_rntraj/repo/`

## Files added for this project

Everything else in `repo/` is the authors' code (changed files are listed under NYC adaptation notes).

- `Dockerfile`, `docker-compose.yml`, `.dockerignore`: the container with FMM and both Python environments.
- `hpc/train_nyc.sbatch`: HPC job that trains, then generates.
- `README.md`: this file.
- `repo/build_graph_nyc.py`: builds the NYC road graph from the training trips (edge list instead of `graph_A.csv`).
- `repo/train_road_embed.py`: trains the road embedding (`road_embed.txt`).
- `fmm_scripts/diffrntraj_version/` (in the URBAN-SYN root): the FMM pipeline that makes `repo/data/NYC/`, with
  its own README.
- `fmm_scripts/nyc_test.csv` (gitignored): the test trips held out from training.

## NYC adaptation notes

Every change in the original code is marked `# Change for Diff-RNTraj (NYC):` with the original lines commented
out above it, so `grep -rn "Change for Diff-RNTraj" repo/` lists them all. Porto and Chengdu work as before.

Changes to the training and generation scripts:

- `--dataset NYC` added with its own map bounds, data folder (`data/NYC/`) and road count; the road count
  (`id_size`) is read from the converted data, so it stays right when the data is rebuilt. The map bounds are the
  bounding box of the NYC road network and need updating if the road network changes.
- Training uses only the training split (`gen_train/`); TS-TrajGen's test trips are held out.
- The learning rate is halved every `epochs / 10` epochs (at least 3) instead of every 3, so long runs keep
  learning. At the default 30 epochs this is the same as before.
- GPU 0 is used by default unless `CUDA_VISIBLE_DEVICES` is already set (e.g. by Slurm); the original hard-coded
  the authors' GPU numbers.
- Unused imports that aren't needed for NYC were commented out (`nni`, and `chinese_calendar`, whose holiday
  feature is set to 0 because Chinese holidays don't apply to NYC).

Generation fixes:

- Generates as many trajectories per length as the training set has, using its length distribution (20-100 points),
  and rounds the per-length counts instead of truncating them.
- Removed a leftover debug `exit()` that stopped generation after length 39.
- Generation clears the old NYC output first: the original appends to the per-length files, so running it twice
  mixed two runs in one file.
- Road similarity is computed in chunks of 2,048 points, because the full matrix ran out of GPU memory.
- Generated trajectories use the dataset's own road count instead of Porto's hard-coded 13,695, which dropped
  NYC roads with higher IDs.

Sparse road graph:

- The original reads a dense N x N road adjacency matrix (`graph_A.csv`), which is far too large for NYC's ~41,500
  roads. The same graph is now stored as an edge list (`graph_edge.edgelist`), loaded as sorted int64 keys, and the
  constraint loss looks up edges with `torch.isin`. The constraint mask and loss are the same as with the dense
  matrix (checked on a test graph).

Other:

- In-place tensor update in the diffusion residual block replaced by an out-of-place one (same math); newer
  PyTorch rejects the in-place version during backward.
- Two new scripts: `build_graph_nyc.py` builds the NYC road graph from the training trips only (so test trips
  don't leak into the graph), and `train_road_embed.py` trains the road embedding (node2vec with p = q = 1) in
  NumPy + PyTorch, in the same format as the original.

## Data-split policy

- **Train (`gen_train/`):** all trips except TS-TrajGen's test trips. Used for training, the road graph and the
  number of trajectories generated per length.
- **Test (`gen_test/`):** the same test trips as TS-TrajGen (listed in `fmm_scripts/nyc_test.csv`), held out
  so both baselines are evaluated on the same trips.
- **All (`gen_all/`):** train + test, kept for reference.
- **No validation set:** like the original code, training has no validation split. Despite its name,
  `val-best-model.pt` is saved whenever the training loss improves, so it is the epoch with the lowest training
  loss; generation uses it. `train-mid-model.pt` is the last epoch.

## Data and outputs

All data and outputs are gitignored; only the code is committed.

- **Raw input:** the MAT-Dataset NYC moves (`data/nyc_output_tabular/output/enriched_moves.parquet`) and the test
  split (`fmm_scripts/nyc_test.csv`).
- **Model input (`repo/data/NYC/`):** made by the FMM pipeline in `fmm_scripts/diffrntraj_version/`: road data and
  road ID maps, the train/test/all trajectory files, the road network, the road graph and the road embedding.
- **Outputs:** the trained model, losses and logs in `repo/results/NYC/`, and the generated trajectories in
  `repo/generate_data/NYC/` (one file per length, 20-100).

## Container

One image (`diffrntraj:latest`) has everything from the raw NYC data to generated trajectories:

- FMM for map matching, built the same way as the TS-TrajGen `fmm_scripts/dockerfile` (same Ubuntu 22.04 packages and FMM
  commit), so the TS-TrajGen FMM container isn't needed here and is not changed.
- A Python environment for the model (Python 3.11, gdal, rtree, networkx < 3, pandas, scikit-learn,
  torch 2.7.1 with CUDA 11.8), used by `python`.
- A separate Python environment for the data pipeline (osmnx 2.0.7, geopandas, pyogrio, pyarrow, pyyaml,
  shapely, networkx), used by `prep-python`. It's separate because the model needs networkx < 3.

`docker-compose.yml` mounts the whole URBAN-SYN folder, so code changes don't need a rebuild. The build ends with an
import check of every package. GPU-enabled through `gpus: all`. On the HPC the same image runs as an Apptainer `.sif`.

## How to run

1. **FMM pipeline: run it locally** (it downloads OpenStreetMap data, so it needs internet). It turns the NYC data
   into `repo/data/NYC/`. Only rerun it when the data or FMM settings change.
2. **Training and generation: on the HPC** with `hpc/train_nyc.sbatch`, or locally if you prefer.

### 1. FMM pipeline (local, Docker Desktop running)

From the URBAN-SYN folder:
```bash
cd fmm_scripts/diffrntraj_version
python run_diffrntraj_pipeline.py --to embed      # all steps
python run_diffrntraj_pipeline.py --from convert --to embed   # resume from a step
python run_diffrntraj_pipeline.py --dry-run       # print the commands only
```
Every step runs in the image; the first run builds it (it compiles FMM, so it takes a while). Details are in
`fmm_scripts/diffrntraj_version/README.md`.

### 2. Train and generate

**On the HPC:** submit `hpc/train_nyc.sbatch` from this folder, with the image as `diffrntraj.sif` next to it.
The job trains, then generates with the trained model; `GENERATE_ONLY=1` skips training. Usage and options are in
the comments at the top of the script.

**Locally** (from this folder):
```bash
docker compose build                                                           # once
docker compose run --rm diff-rntraj python multi_main.py --dataset NYC         # train (30 epochs by default)
docker compose run --rm diff-rntraj python generate_data.py --dataset NYC      # generate
```
Generation uses `repo/results/NYC/val-best-model.pt` (the epoch with the lowest training loss; see the data-split
policy) and replaces any earlier output in `repo/generate_data/NYC/`. Pass the same options to both scripts
(e.g. `--epochs`, `--diff_T`) if you change them.
