# URBAN-SYN Evaluation

One evaluation for every trajectory generation model in URBAN-SYN. Each baseline writes its output in a different format, so this folder first converts every model's output, and the matching real test and training sets, into one common format. 

## Models

| `--model` | Baseline | Native output |
|---|---|---|
| `difftraj` | DiffTraj | GPS points |
| `diffrntraj` | Diff-RNTraj | road ID + position on the road |
| `tstrajgen` | TS-TrajGen | road IDs |
| `mobilitygpt` | MobilityGPT | road IDs |
| `lstm_trajgan` | LSTM-TrajGAN | GPS points |

All five use the same split by user: `data/split_by_user/` (made by `data_expansion/split_enriched_moves_by_user.ipynb` from `enriched_moves.parquet`). The test users are listed in `data/split_by_user/test_trips.csv` (`uid, tid`).

## Quick start

Run from the URBAN-SYN folder. In the dev container the packages are in `.venv` (made by `uv sync` when the container is created), so use `uv run` as below, or run `source .venv/bin/activate` once and then plain `python`. The container's system `python` has no numpy. Outside the container, any Python 3.10+ with numpy, pandas and pyyaml works.

```bash
# check that every input file is in place
uv run python -m evaluation.convert --model all --check

# score one model with all 12 metrics
uv run python -m evaluation.evaluate --model diffrntraj

# score several models, or all of them, in one run
uv run python -m evaluation.evaluate --model tstrajgen mobilitygpt
uv run python -m evaluation.evaluate --model all

# only convert, and save the converted tables as CSV files
uv run python -m evaluation.convert --model difftraj --save evaluation/results/converted
```

**DiffTraj and LSTM-TrajGAN** write GPS points, so the evaluation map-matches them with FMM for road connectivity. It does that by itself, but FMM isn't installed in the dev container, so when their matches are missing or out of date, run the evaluation in the `fmm_scripts` image, which has FMM and the evaluation's Python packages (Docker Desktop running; PowerShell or any shell, from the URBAN-SYN folder):

```bash
docker compose -f fmm_scripts/docker-compose.yml build      # once (FMM is compiled the first time)
docker compose -f fmm_scripts/docker-compose.yml run --rm -T -w /urban-syn fmm python -m evaluation.evaluate --model all
```

The image mounts the whole URBAN-SYN folder at `/urban-syn`.

Once their matches are up to date, the dev container commands above work for every model again. `--no-fmm` scores the GPS models without road connectivity, without FMM.

## The metrics

| # | Metric | Better | Compares |
|---|---|---|---|
| 1 | Global geographic JSD | lower | where the GPS points fall on the shared 0.005° grid |
| 2 | OD flow JSD | lower | joint start/end cell of each trajectory |
| 2a | Start point JSD | lower | the start cell of each trajectory (extra: one half of the OD flow) |
| 2b | End point JSD | lower | the end cell of each trajectory (extra: the other half) |
| 3 | Trip distance JSD | lower | path length histogram (1000 bins to 40.71 km, plus overflow) |
| 4 | Radius JSD | lower | mean distance to the trajectory's center (100 bins to 10.6 km, plus overflow) |
| 5 | Hausdorff (km) | lower | paired real and generated trajectories |
| 6 | Normalized DTW (km) | lower | paired trajectories, divided by the warping-path length |
| 7 | EDR | lower | paired trajectories, ε = 100 m |
| 8 | Grid-cell Jaccard | higher | unique cells of paired trajectories |
| 9 | MIA ROC-AUC | 0.5 | can the generated set tell training from test trajectories? |
| 10 | Uniqueness | higher | share of generated trajectories > 50 m from every training trajectory |
| 11 | Off-network rate | lower | share of native GPS points > 30 m from the canonical road network |
| 12 | Road connectivity (RSC) | higher | share of consecutive roads where the next road directly follows |

Every setting (grid, bins, thresholds, sample sizes, seed) is in the `metrics:` section of `eval_config.yaml`, so all models are scored the same way. The JSDs use log base 2 (0 to 1) and compare sets of equal size. Trajectories are paired by ID when the model was conditioned on a test trajectory, and otherwise with one shared rule: a real trajectory that starts and ends in the same grid cells (the share that finds one is reported with the scores). The privacy metrics use every trajectory. Each model also gets a **real train** row (a sample of real training data scored in place of the generated set) as the floor, and a **real test** row for the off-network rate and RSC of the real data.

**Evaluation details**

| | Setting |
|---|---|
| Distribution metrics | Jensen-Shannon divergence, log base 2 (0 = same, 1 = no overlap), against the real test set at the model's unit; when the sets differ in size, the larger is randomly sampled down to the smaller size 10 times (seed 42) and the mean of the 10 random samples is reported |
| Grid | 0.005°, 87 × 90 cells, lng −74.2963164 to −73.868091, lat 40.4983251 to 40.9410935; points outside counted in one extra cell (none occurred) |
| Trip distance, radius | Haversine (R = 6371 km); 1000 bins to 40.71 km and 100 bins to 10.6 km, each plus an overflow bin; radius = mean distance to the arithmetic center |
| Pairing | by ID for the conditioned models; same start and end cell for Diff-RNTraj and the real-data rows; at most 5000 pairs (seed 42) |
| Trajectory metrics | Hausdorff (symmetric, km); DTW with Haversine cost divided by the warping-path length (exact, no FastDTW); EDR with ε = 100 m, match when d < ε, divided by the longer length; Jaccard of unique grid cells |
| Privacy | every trajectory (no sampling); each resampled to 32 evenly spaced points; distance = mean Haversine between corresponding points; MIA: training (members) vs test (non-members), score = minus the distance to the closest generated trajectory, ROC-AUC; each test trajectory's own generated counterpart is left out; uniqueness: more than 50 m from every training trajectory |
| Off-network | native GPS points more than 30 m from TS-TrajGen's road network (`nyc.geo`), the same network for every model |
| Snapped connectivity | directed, next road must start where the previous one ends, consecutive repeats collapsed, transition-weighted; TS-TrajGen's `nyc.rel` (Diff-RNTraj: `u`/`v` of `nyc_drive.gpkg`) |
| Real-data rows | real training data at the model's unit in place of the generated set (sections 1-2); the real test set for off-network and connectivity |

The full results of the run, including every setting of the threshold-dependent metrics, are in `evaluation/results/<run>/table.md` and `scores.csv`.

## Inputs

Each converter reads the baseline's files where the baseline wrote them, and `eval_config.yaml` lists every one of those paths. If a file is missing, the run stops before converting anything and lists every missing file with the step that produces it.

| Model | Files (`eval_config.yaml`) | Made by |
|---|---|---|
| `difftraj` | `Files/<timestamp>/synthetic_trajectories.npy`, `synthetic_index.npy` | `generate.py` (one folder per run) |
| | `real_trajectories_test.npy`, `traj_ids_test.npy`, `real_trajectories_train.npy`, `traj_ids_train.npy` | `prepare_data.py --extras` |
| | FMM matches, in `fmm_dir` | made by the evaluation (see "FMM matches for the GPS models" below) |
| `diffrntraj` | `generate_data/<folder>/` (one `<length>.txt` per length) | the generation run |
| | `data/NYC/extra_file/*.json`, `data/NYC/gen_test/*.bin`, `data/NYC/gen_train/*.bin` | `fmm_scripts/diffrntraj_version/` pipeline |
| `tstrajgen` | `datasets/nyc/TS_TrajGen_GAN_generated_output.csv`, `nyc_mm_test.csv`, `nyc_mm_train.csv`, `nyc.geo`, `nyc.rel` | the TS-TrajGen pipeline (`run_pipeline.py`) |
| `mobilitygpt` | `nyc_workflow/output/nycall_final/...` | the MobilityGPT `nyc_workflow`, run on TS-TrajGen's `nyc_mm_train.csv` / `nyc_mm_test.csv` |
| `lstm_trajgan` | `outputs/lstm_trajgan_nyc_full/results/nyc_syn_traj_test.csv`, `data/nyc_full/nyc_test_latlon.csv`, `nyc_train_latlon.csv` | the LSTM-TrajGAN pipeline |
| | FMM matches, in `fmm_dir` | made by the evaluation |

TS-TrajGen's `nyc.geo` and `nyc.rel` are also the road network of DiffTraj, LSTM-TrajGAN and MobilityGPT (`network: tstrajgen` in the converters), so all four must come from the same road network.

### After retraining a baseline

- **DiffTraj:** every run writes a new `Files/<timestamp>/` folder, so point `generated` and `index` at it. If the split changed, rerun `prepare_data.py --extras` for the real files. The evaluation remakes its FMM matches.
- **Diff-RNTraj:** point `generated_dir` at the generation that matches the test lengths (`repo/models/multi_train.py` reads `gen_test/length_distri.npy`). Its `data/NYC/` files must come from the same pipeline run the model was trained on.
- **LSTM-TrajGAN:** the paths stay the same. The evaluation remakes its FMM matches.
- **TS-TrajGen:** the paths stay the same. If its road network changed (`fmm_scripts/prepare_fmm_inputs.py` was rerun), rerun MobilityGPT; the evaluation remakes the GPS models' FMM matches on the new network.
- **MobilityGPT:** rerun it whenever TS-TrajGen's `nyc_mm_train.csv` / `nyc_mm_test.csv` change.

Then run `convert --check`, and `convert` without `--check` to see each model's counts and notes before scoring.

## Output

Each `evaluate` run writes a new folder `evaluation/results/<date>_<time>_<models>/`:

| File | Content |
|---|---|
| `table.md` | the results by metric family, one column per model, with the real-data value in brackets next to each score |
| `scores.csv` | every value: model, group, metric, setting, value, std, sample size and a note |
| `run.json` | the settings, the input files (size and date), the git commit and diagnostics per model |

The table is also printed at the end of the run. The `results/` folder is not tracked by git.

## Folder layout

```text
evaluation/
├── README.md            
├── eval_config.yaml     input paths per model, road networks and metric settings
├── convert.py           command line: --model, --check, --save, --no-fmm
├── evaluate.py          command line: --model, --no-fmm; scores every model and writes results/
├── inputs.py            reads the config, checks the input files
├── tables.py            the common format and its checks
├── geo.py               distance, projection and point-on-road helpers
├── fmm_matches.py       reads FMM match files into road sequences
├── fmm_prepare.py       map-matches the GPS models with FMM when their matches are missing or out of date
├── converters/          one converter per baseline
└── metrics/             the 12 metrics, the shared grid, JSD and pairing
```

