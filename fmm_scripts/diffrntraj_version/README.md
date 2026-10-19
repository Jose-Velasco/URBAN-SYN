# Diff-RNTraj version of fmm_scripts

This folder turns the NYC trajectories into the input files of the Diff-RNTraj baseline
(`baselines/diffusion/diff_rntraj/`). It is a Diff-RNTraj version of the TS-TrajGen `fmm_scripts/`: the files that needed
changes were copied here with a `_diffrntraj` suffix, and none of the TS-TrajGen files are changed. From that folder it
uses `utils.py` and the road network settings in `config/nyc.yaml`.

## Files added for this project

Created only for the Diff-RNTraj version:
- `run_diffrntraj_pipeline.py`: runs every step with one command, in the Diff-RNTraj container.
- `convert_fmm_to_diffrntraj.py`: turns the FMM matches into Diff-RNTraj's data files in `repo/data/NYC/`.
- `README.md`: this file.
- `fmm_scripts/nyc_test_trips.csv` (one folder up, gitignored): TS-TrajGen's test trips, held out from training.

Copies of TS-TrajGen files, changed for Diff-RNTraj (changes listed under NYC adaptation notes):
- `prepare_fmm_inputs_diffrntraj.py`: builds the road network and the GPS points.
- `config/ubodt_config_diffrntraj.xml`: shortest-path table settings.
- `config/fmm_config_csv_point_diffrntraj.xml`: map-matching settings.

Made when the pipeline runs (gitignored): `data/`, `logs/` and `cache/`.

## NYC adaptation notes

Compared with the TS-TrajGen scripts:

- **Shared road network settings:** the road network uses the TS-TrajGen `config/nyc.yaml` as it is: drive-only roads
  (one-way streets stay one-way), downloaded for the bounding box of all GPS points plus 2 km, so trips that leave the
  city, e.g. into New Jersey, can still be matched. The script stops if that config no longer asks for drive-only
  roads.
- **New GPS points:** moves of every travel mode, one point every 15 s, cut into pieces of 20-100 points, which
  is Diff-RNTraj's trajectory format. Trajectory IDs start at 1, because FMM writes ID 0 for some failed
  trajectories.
- **Extra columns for the converter:** the road network keeps a 1-based road ID and a numeric road level, and the
  trip map keeps each piece's user, trip, move, mode, start/end time and number of points.
- **Position on the road:** FMM also writes each point's offset along its road, which the converter turns into
  Diff-RNTraj's position on the road.
- **Separate outputs:** everything is written to this folder's `data/` and `logs/` with its own file names, so it
  never overwrites the TS-TrajGen outputs.

Every changed line is marked `CHANGED` or `NEW`.

## Test split

`fmm_scripts/nyc_test_trips.csv` lists TS-TrajGen's test trips. The converter puts these trips in `gen_test/` and
everything else in `gen_train/`, so Diff-RNTraj trains without them. If the file is missing there is no split.

## How to run

Run this pipeline locally (Docker Desktop running; the first step downloads OpenStreetMap data, so it needs
internet). From the URBAN-SYN folder:
```bash
cd fmm_scripts/diffrntraj_version
python run_diffrntraj_pipeline.py --to embed
```
Steps, in order: road network and GPS points, shortest-path table, map matching, conversion to `repo/data/NYC/`,
road graph, and road embedding (only with `--to embed`). Each runs in the Diff-RNTraj image, which the first run
builds. Other options: `--from <step>`, `--only <step>`, `--dry-run`. After replacing the test split, rerun from the
conversion: `--from convert --to embed`.

It needs `data/nyc_output_tabular/output/enriched_moves.parquet` (in the URBAN-SYN root) and
`fmm_scripts/nyc_test_trips.csv`. Outputs (gitignored) go to `data/` and `logs/` in this folder, and the model
input to `baselines/diffusion/diff_rntraj/repo/data/NYC/`.
