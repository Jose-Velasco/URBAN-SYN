# URBAN-SYN Evaluation

One evaluation for every trajectory generation model in URBAN-SYN. Each model writes its output in a different format, so this folder first converts every model's output, and the matching real test and training sets, into one common format.

**Everything is set in [`eval_config.yaml`](eval_config.yaml).** It lists the path of every input file: each model's generated and real files (`models:`), the road networks (`networks:`) and the FMM files used to map-match the GPS models (`fmm:`), plus every metric setting (`metrics:`) and the plot settings (`plots:`). To point the evaluation at new files, edit only that file. `convert --check` lists any missing file with the step that makes it (see [Inputs](#inputs)).

## Models

| `--model` | Baseline | Native output |
|---|---|---|
| `difftraj` | DiffTraj | GPS points |
| `diffrntraj` | Diff-RNTraj | road ID + position on the road |
| `tstrajgen` | TS-TrajGen | road IDs |
| `mobilitygpt` | MobilityGPT | road IDs |
| `lstm_trajgan` | LSTM-TrajGAN | GPS points |
| `urban_syn` | URBAN-SYN, our model (GNN road encoder + Diff-RNTraj) | road ID + position on the road, with GPS points |

All six use the same split by user: `data/split_by_user/` (made by `scripts/split_enriched_moves_by_user.ipynb` from `enriched_moves.parquet`). The test users are listed in `data/split_by_user/test_trips.csv` (`uid, tid`).

## Quick start

Run from the URBAN-SYN folder. In the dev container the packages are in `.venv` (made by `uv sync` when the container is created), so use `uv run` as below, or run `source .venv/bin/activate` once and then plain `python`. The container's system `python` has no numpy. Outside the container, any Python 3.10+ with numpy, pandas, pyyaml and pyarrow (for the Parquet files of `urban_syn`) works, plus matplotlib for the plots.

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

# plot real and generated trajectories
uv run python -m evaluation.plot --model all
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

Every setting (grid, bins, thresholds, sample sizes, seed) is in the `metrics:` section of `eval_config.yaml`

**Evaluation details**

| | Setting |
|---|---|
| Distribution metrics | Jensen-Shannon divergence, log base 2 (0 = same, 1 = no overlap), against the real test set at the model's unit; when the sets differ in size, the larger is randomly sampled down to the smaller size 10 times (seed 42) and the mean of the 10 random samples is reported |
| Grid | 0.005°, 87 × 90 cells, lng −74.2963164 to −73.868091, lat 40.4983251 to 40.9410935; points outside counted in one extra cell (none occurred) |
| Trip distance, radius | Haversine (R = 6371 km); 1000 bins to 40.71 km and 100 bins to 10.6 km, each plus an overflow bin; radius = mean distance to the arithmetic center |
| Pairing | by ID for the conditioned models; same start and end cell for Diff-RNTraj and the real-data rows; at most 5000 pairs (seed 42) |
| Trajectory metrics | Hausdorff (symmetric, km); DTW with Haversine cost divided by the warping-path length (exact, no FastDTW); EDR with ε = 100 m, match when d < ε, divided by the longer length; Jaccard of unique grid cells |
| Privacy | every trajectory (no sampling); each resampled to 32 evenly spaced points; distance = mean Haversine between corresponding points; MIA: training (members) vs test (non-members), score = minus the distance to the closest generated trajectory, ROC-AUC; each test trajectory's own generated counterpart is left out; uniqueness: more than 50 m from every training trajectory |
| Off-network | native GPS points more than 30 m from the shared road network (`fmm_scripts/data/road_network/nyc.gpkg`), the same network for every model |
| Snapped connectivity | directed, next road must start where the previous one ends, consecutive repeats collapsed, transition-weighted; `u`/`v` of the shared `nyc.gpkg` for DiffTraj, LSTM-TrajGAN and URBAN-SYN, TS-TrajGen's `nyc.rel` (written from the same `nyc.gpkg` with the same rule) for TS-TrajGen and MobilityGPT, `u`/`v` of `nyc_drive.gpkg` for Diff-RNTraj |
| Real-data rows | real training data at the model's unit in place of the generated set (sections 1-2); the real test set for off-network and connectivity |

The full results of the run, including every setting of the threshold-dependent metrics, are in `evaluation/results/<run>/table.md` and `scores.csv`.

## Inputs

Each converter reads a model's files where the model wrote them, and `eval_config.yaml` lists every one of those paths, along with the road networks and FMM files below. If a file is missing, the run stops before converting anything and lists every missing file with the step that produces it. All paths are relative to the URBAN-SYN folder.

### Model files (`models:`)

| Model | Files | Made by |
|---|---|---|
| `difftraj` | `Files/<timestamp>/synthetic_trajectories.npy`, `synthetic_index.npy` | `generate.py` (one folder per run) |
| | `real_trajectories_test.npy`, `traj_ids_test.npy`, `real_trajectories_train.npy`, `traj_ids_train.npy` | `prepare_data.py --extras` |
| | FMM matches, in `fmm_dir` | made by the evaluation (see [FMM files](#fmm-files-for-the-gps-models-fmm)) |
| `diffrntraj` | `generate_data/<folder>/` (one `<length>.txt` per length) | the generation run |
| | `data/NYC/extra_file/*.json`, `data/NYC/gen_test/*.bin`, `data/NYC/gen_train/*.bin` | `fmm_scripts/diffrntraj_version/` pipeline |
| `tstrajgen` | `datasets/nyc/TS_TrajGen_GAN_generated_output.csv`, `nyc_mm_test.csv`, `nyc_mm_train.csv`, `nyc.geo` | the TS-TrajGen pipeline (`run_pipeline.py`), from the shared network's FMM matches: `prepare_fmm_gps.py` → `fmm ./config/fmm_config_csv_point_ts_trajgen.xml` → `split_fmm_matches.py` in `fmm_scripts/` (see `baselines/gan/ts_trajgen/PIPELINE.md`) |
| `mobilitygpt` | `nyc_workflow/output/nycall_final/...` | the MobilityGPT `nyc_workflow`, run on TS-TrajGen's `nyc_mm_train.csv` / `nyc_mm_test.csv` |
| `lstm_trajgan` | `outputs/lstm_trajgan_nyc_full/results/nyc_syn_traj_test.csv`, `data/nyc_full/nyc_test_latlon.csv`, `nyc_train_latlon.csv` | the LSTM-TrajGAN pipeline |
| | FMM matches, in `fmm_dir` | made by the evaluation (see [FMM files](#fmm-files-for-the-gps-models-fmm)) |
| `urban_syn` | `outputs/experiments/<experiment>/<run_id>/generated/*_generated_gps.parquet`, one `generated_<name>` per run | the experiment's `generate` stage (`src/urban_syn`) |
| | `fmm_scripts/outputs/urban_syn_ready/*_{test,train}_prepared_road_sequences.parquet` | `fmm_scripts/urban_syn_extension/` (see its README) |

**URBAN-SYN's files.** A generated file has one row per point: `trajectory_id`, `step`, `model_road_id`, `edge_id`, `rate`, `position_rate`, `lng`, `lat`. The evaluation scores the generator's own `lat`/`lng` and uses `edge_id` (the shared network's road ID) for the road metrics; `model_road_id` and `rate` are not needed. Each `generated_<name>` file under `files:` becomes one row group named `<name>`, e.g. `finetune`, `scratch` and `frozen` for the three `nyc_full_training` runs. The real test and training pieces are the prepared road sequences the model was tested and trained on, with each point placed on its road's geometry in the shared `nyc.gpkg` (`networks: shared:` in `eval_config.yaml`) the same way the generator places its points. The converter checks this: if a generated point is more than 1 m from where its `edge_id` and `position_rate` put it on that `nyc.gpkg`, the model used a different road network, and the run stops. URBAN-SYN is unconditional, so like Diff-RNTraj it is paired by the shared fallback rule.

### Road networks (`networks:`)

Road connectivity and the off-network rate are measured on a road network. Each network's files are listed once, under `networks:`, and both the converters and the metrics read them from there.

| Network | Files | Used by | Made by |
|---|---|---|---|
| `shared` | `fmm_scripts/data/road_network/nyc.gpkg` | DiffTraj, LSTM-TrajGAN and URBAN-SYN; the off-network rate of every model | the shared road network (`fmm_scripts/README.md`, `build_road_network.py`) |
| `tstrajgen` | `baselines/gan/ts_trajgen/datasets/nyc/nyc.geo`, `nyc.rel` | TS-TrajGen and MobilityGPT | TS-TrajGen's build-inputs stage, from the shared `nyc.gpkg` |
| `diffrntraj_drive` | `fmm_scripts/diffrntraj_version/data/road_network/nyc_drive.gpkg` | Diff-RNTraj | the `fmm_scripts/diffrntraj_version/` pipeline (its own download) |

**Shared network or TS-TrajGen's files.** DiffTraj, LSTM-TrajGAN and URBAN-SYN are scored on the shared road network itself (`NETWORK = "shared"` in their converters), which the evaluation reads directly with Python's built-in `sqlite3` (no geopandas): DiffTraj and LSTM-TrajGAN through FMM's `fmm_nyc.shp`, built with it, and URBAN-SYN through its `edge_id`s. So they need no TS-TrajGen files. TS-TrajGen and MobilityGPT write road IDs of TS-TrajGen's `nyc.geo` (`NETWORK = "tstrajgen"`), which TS-TrajGen builds from the same `nyc.gpkg`, so its `geo_id`s are the same `edge_id`s and both networks give the same scores. Keep FMM's files and TS-TrajGen's build from the same build of `nyc.gpkg`: the evaluation stops if FMM's network and `nyc.gpkg` have a different number of roads.

### FMM files for the GPS models (`fmm:`)

DiffTraj and LSTM-TrajGAN write GPS points, so the evaluation map-matches them onto the shared network with FMM, using the shared map-matching setup in `fmm_scripts/`:

| File | Path | Made by |
|---|---|---|
| FMM config template | `fmm_scripts/config/fmm_config_csv_point.xml` | in git; the evaluation copies it and replaces only the GPS input file, the GPS id column, the output file and the output fields, so `k`, `r`, the GPS error and the network stay the shared settings |
| FMM road network | `fmm_scripts/data/fmm/fmm_nyc.shp` (with `.dbf`, `.shx`) | `build_road_network.py`, together with `nyc.gpkg` |
| Shortest-path table (UBODT) | `fmm_scripts/data/nyc_ubodt.bin` | `ubodt_gen ./config/ubodt_config.xml`, in the FMM image, from `fmm_scripts/` |

The evaluation keeps each match in the model's `fmm_dir` (`fmm_scripts/outputs/difftraj`, `fmm_scripts/outputs/lstm_trajgan`): `<model>_<set>_fmm_points.csv` (FMM's input), `_fmm_trip_id_map.csv`, `_fmm_config.xml`, `_fmm_match.csv` (FMM's output) and `_fmm_source.json`, a fingerprint of the inputs. A match is reused until the trajectories, the FMM road network, the UBODT or the FMM settings change, and then made again.

### After retraining a baseline

- **DiffTraj:** every run writes a new `Files/<timestamp>/` folder, so point `generated` and `index` at it. If the split changed, rerun `prepare_data.py --extras` for the real files. The evaluation remakes its FMM matches.
- **Diff-RNTraj:** point `generated_dir` at the generation that matches the test lengths (`repo/models/multi_train.py` reads `gen_test/length_distri.npy`). Its `data/NYC/` files must come from the same pipeline run the model was trained on.
- **LSTM-TrajGAN:** the paths stay the same. The evaluation remakes its FMM matches.
- **TS-TrajGen:** the paths stay the same. If the shared network in `fmm_scripts/` was rebuilt, rebuild TS-TrajGen on it and rerun MobilityGPT (URBAN-SYN needs retraining on it too); the evaluation remakes the GPS models' FMM matches on the new network.
- **MobilityGPT:** rerun it whenever TS-TrajGen's `nyc_mm_train.csv` / `nyc_mm_test.csv` change.
- **URBAN-SYN:** point each `generated_<name>` at its run's `generated/` file (replace `RUN_ID` with the run folder). Its `urban_syn_ready/` files must come from the by-user split (`split_fmm_matches.py --test_trips ../data/split_by_user/test_trips.csv`) and be the ones the run was trained on.

Then run `convert --check`, and `convert` without `--check` to see each model's counts and notes before scoring.

## Output

Each `evaluate` run writes a new folder `evaluation/results/<date>_<time>_<models>/`:

| File | Content |
|---|---|
| `table.md` | the results by metric family, one column per model, with the real-data value in brackets next to each score |
| `scores.csv` | every value: model, group, metric, setting, value, std, sample size and a note |
| `run.json` | the settings, the input files (size and date), the git commit and diagnostics per model |

The table is also printed at the end of the run. The `results/` folder is not tracked by git.

### Plots

Each `plot` run writes a new folder `evaluation/results/<date>_<time>_plots_<models>/`:

| File | Content |
|---|---|
| `<model>_trajectories.png` | every trajectory, or a random sample when `plots: sample` in `eval_config.yaml` is a number (drawn with the metrics' seed): the real test set in grey, then each generated group in blue |
| `<model>_density.png` | the share of points in each 0.005° grid cell, the cells the global geographic JSD compares, on one log color scale for all panels |
| `overview_trajectories.png`, `overview_density.png` | every model side by side, next to one real test set, named in its panel title: TS-TrajGen's (`plots: overview_real`), or the first model listed when TS-TrajGen isn't plotted; only when more than one model is plotted |

Each panel is named after its model; only a model with several generated groups, such as URBAN-SYN's three runs, gets one panel per group, with the group in brackets. The plots draw the converters' GPS tables, the points the metrics score, so models that write road IDs are drawn the way the metrics see them (MobilityGPT and TS-TrajGen through the midpoints of their roads). Each model's real test set is at its own unit (moves, pieces or sections), so the per-model plots compare like with like. The plots need no FMM matches, so run them in the dev container; the `fmm_scripts` image has no matplotlib.

## Folder layout

```text
evaluation/
├── README.md            
├── eval_config.yaml     input paths per model, road networks and metric settings
├── convert.py           command line: --model, --check, --save, --no-fmm
├── evaluate.py          command line: --model, --no-fmm; scores every model and writes results/
├── plot.py              command line: --model; trajectory and density plots into results/
├── inputs.py            reads the config, checks the input files
├── tables.py            the common format and its checks
├── geo.py               distance, projection and point-on-road helpers
├── fmm_matches.py       reads FMM match files into road sequences
├── fmm_prepare.py       map-matches the GPS models with FMM when their matches are missing or out of date
├── converters/          one converter per model
└── metrics/             the 12 metrics, the shared grid, JSD and pairing
```

