# Baseline: MobilityGPT

This directory contains the adapted MobilityGPT baseline used for the NYC trajectory-generation experiments.

The implementation is based on the original MobilityGPT repository. The model, trainers and fine-tuners are the
authors' code with a few changes (the ones that affect results are listed below), and `repo/nyc_workflow/` adds
the NYC data conversion, training setup, generation and evaluation.

## Source

- Original repository: https://github.com/ammarhydr/MobilityGPT (compared against commit `068aa0e`, Jan 2025)
- Paper: [MobilityGPT: Enhanced Human Mobility Modeling with a GPT model](https://arxiv.org/abs/2402.03264)
- Local baseline path: `baselines/transformer/mobilitygpt/repo/`

## Main files added for this project

- `README.md`: this file.
- `repo/nyc_workflow/preprocessing.py`: converts the NYC map-matched trips into the authors' input format.
- `repo/nyc_workflow/data.py`: trajectory-indexed dataset for gravity sampling, and the train/validation split by
  trip.
- `repo/nyc_workflow/pretraining.py`, `repo/nyc_workflow/dpo.py`: the NYC pretraining and DPO settings, using the
  authors' model and trainers.
- `repo/nyc_workflow/paired.py`, `repo/nyc_workflow/evaluation.py`: generation from test-trip prompts and the
  paired length/distance/revisit metrics.
- `repo/nyc_workflow/metrics.py`: the 12 metrics from the team's evaluation notebook
  (`trajectory_evaluation_ts_lstm_all_metrics_parallel.ipynb`), reimplemented for MobilityGPT's output.
- `repo/nyc_workflow/pipeline.py`: runs all steps with one command.

## NYC adaptation notes

- **Fixes in the authors' code:** adjacency masking now blocks unconnected roads (the original only zeroed their
  logits), and gravity sampling weights whole trajectories instead of token windows (`mobilitygpt/model.py`,
  `mobilitygpt/trainer_utils.py`).
- **Data:** TS-TrajGen's map-matched files (all transport modes). Repeated roads are collapsed and trips with
  unconnected gaps are split into sections. The vocabulary is the 43,890 roads used in training.
- **Training:** pretraining (3,000 updates, batch 64, learning rate 1e-5), then DPO from the pretrained model
  (250 updates; update 100 is reported). Supervised fine-tuning and PPO are skipped because they didn't work on
  NYC.
- **Generation:** each test trip's first 1 or 3 roads are given as a prompt (the authors start from a random
  training origin). Only 2,903 of the 3,584 test trips are evaluated; the rest use roads outside the vocabulary.
- **Metrics:** the team's 12 notebook metrics (`metrics.py`). Off-network rate needs `--canonical-network`.

## Data-split policy

- **Train:** TS-TrajGen's `nyc_mm_train.csv` (14,336 trips).
- **Validation:** 20% of the training trips, split by original trip (seed 42), so sections of one trip stay
  together: 11,390 training and 2,847 validation trips. Gravity statistics use the training part only. The DPO
  preference pairs are drawn from all training trips, validation included.
- **Test:** TS-TrajGen's `nyc_mm_test.csv` (3,584 trips, the same trips as `fmm_scripts/nyc_test.csv`), so it
  matches TS-TrajGen and Diff-RNTraj. Only the 3,123 sections inside the training vocabulary are evaluated.

## Data and outputs

All data and outputs are gitignored; only the code is committed.

- **Raw input:** `repo/NYC/` with the four TS-TrajGen files.
- **Road network (optional, for off-network rate):** the FMM pipeline's `fmm_scripts/data/road_network/nyc.gpkg`
  (layer `roads`), made by `fmm_scripts/prepare_fmm_inputs.py`.
- **Outputs:** one folder per run in `repo/nyc_workflow/output/<run-id>/`:
  - `pretrain/model_best.pt`: the pretrained model (also the DPO reference).
  - `training/model_step_000100.pt`: DPO update 100 (linked as `model_selected.pt`).
  - `evaluation/generated_trajectories.jsonl`: the generated trajectories, in the renumbered road IDs (map back
    with `prepared/NYCAll-Taxi/id_mapping.csv`).
  - `canonical_metrics/`: the notebook metrics.

## Container

There is no container or HPC job script. The recorded run used a local virtual environment: Python 3.13.5,
PyTorch 2.11.0 with CUDA 12.8, on an RTX 3080, with the versions pinned in `repo/nyc_workflow/requirements.txt`.
CUDA is required, and the pipeline runs on Linux (or WSL), not on native Windows.

## How to run

The code in the repository can't run yet: see Known gaps first.

From `repo/`:
```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r nyc_workflow/requirements.txt
.venv/bin/python -m pip install --no-deps -e .

.venv/bin/python -m nyc_workflow run --input-dir NYC \
    --canonical-network /absolute/path/to/nyc.gpkg --network-layer roads
```

Leave out `--canonical-network` to skip off-network rate. Each run creates a new output folder.

## Known gaps

- **`repo/datasets/` is missing.** The root `.gitignore` rule `baselines/**/datasets/*` excluded it, but
  `main.py`, `mobilitygpt/trainer_utils.py` and `nyc_workflow/` import it. The version used for NYC has changes
  (boolean adjacency matrix), so the authors' original `datasets/` is not a drop-in replacement. It needs a
  `.gitignore` exception and a commit.
- **`repo/nyc_workflow/notebook_provenance.json` is missing** (excluded by the root `*.json` rule); `metrics.py`
  reads it.
- **`repo/mobilitygpt/__init__.py` (empty in the original) is missing**, so `pip install -e .` doesn't install
  the `mobilitygpt` package; commands only work when run from `repo/`.
- The trained models and generated trajectories from the recorded run are not in the repository.
