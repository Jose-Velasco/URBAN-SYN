# MobilityGPT (NYC)

This baseline packages the original [MobilityGPT](https://github.com/ahmaddanesh/MobilityGPT) model and training code with the NYC workflow used in URBAN-SYN. Application code lives in [`repo/`](repo/); the shared model, supervised fine-tuning, DPO and PPO CLIs remain available. The license is in [repo/LICENSE](repo/LICENSE).

## NYC workflow

For SJSU Apptainer/Slurm instructions, see the [HPC guide](hpc/README.md).

The pipeline runs: preflight → conservative preprocessing → training-only gravity fit → safety check → gravity-weighted pretraining → preference generation → DPO → checkpoint selection → paired evaluation → notebook-compatible metrics → Markdown/JSON/HTML reports.

Prerequisites are Docker Engine/Compose with the NVIDIA Container Toolkit, a CUDA-capable GPU, and the four raw NYC files. The image uses Python 3.13 and the pinned CUDA 12.8 environment in `repo/nyc_workflow/requirements.txt`.

```bash
docker compose build
docker compose run --rm mobilitygpt python -m nyc_workflow run --input-dir NYC
```

To inspect command options:

```bash
docker compose run --rm mobilitygpt python -m nyc_workflow run --help
docker compose run --rm mobilitygpt python -m nyc_workflow metrics --help
docker compose run --rm mobilitygpt python -m nyc_workflow summarize --help
```

The outer baseline is mounted at `/workspace/mobilitygpt`; the container works in `/workspace/mobilitygpt/repo`, and `repo/NYC/` is mounted read-only. Generated runs stay under `repo/nyc_workflow/output/` on the host. The existing inputs are accepted without renaming: `roadmap.geo` aliases `nyc.geo`, `roadmap.rel` aliases `nyc.rel`, and `NYC_Taxi_trajectory_train.csv` / `NYC_Taxi_trajectory_test.csv` alias `nyc_mm_train.csv` / `nyc_mm_test.csv`. Canonical names are preferred for new input drops.

| Location | Contents |
| --- | --- |
| `repo/NYC/nyc.geo` | Required road geography (`geo_id`, coordinates); legacy alias: `roadmap.geo` |
| `repo/NYC/nyc.rel` | Required directed road relations; legacy alias: `roadmap.rel` |
| `repo/NYC/nyc_mm_train.csv` | Required training trajectories; legacy alias: `NYC_Taxi_trajectory_train.csv` |
| `repo/NYC/nyc_mm_test.csv` | Required test trajectories; legacy alias: `NYC_Taxi_trajectory_test.csv` |
| `repo/nyc_workflow/output/<run-id>/` | Prepared data, checkpoints, generated routes, evaluation files, reports and provenance |

The standard command above automatically calculates `off_network_rate` from the original raw `NYC/nyc.geo` road polylines (or the accepted `roadmap.geo` alias). No extra file, conversion, or command-line flag is needed. This is the fraction of generated road representative points strictly more than 30 metres from the nearest reference road; lower is better. Points use the average of each road geometry's endpoints, so they can fall away from curved roads even for valid road IDs.

An optional canonical road network takes precedence when supplied with `--canonical-network NYC/roads.gpkg --network-layer roads` (and, when needed, `--network-mode-column drive`). Place it under the mounted input directory. An explicitly supplied invalid network still fails validation rather than silently switching references.

JSON results record `network.reference` as `raw_geo_derived` or `supplied_canonical`; HTML and Markdown reports also identify the reference. The raw-derived fallback measures source-map consistency. All 12 metrics are calculated, while canonical `status` remains `incomplete` and `complete` remains `false` without an independently supplied network. This status does not mean training failed or the off-network value is missing. Individual metrics may still be unavailable for other reasons, such as empty eligible samples.

For HPC, upload the updated `repo/` source before the next run; see the [HPC guide](hpc/README.md). This evaluation change also updates preflight checks, pipeline validation/configuration (schema version 4), report labels, and regression tests. Model code, training settings, other metric formulas, and raw data are unchanged. Existing saved results are preserved; changed source/configuration cannot be used to strictly resume an older run.

Raw inputs, environments, caches, generated output and Git metadata are excluded from the image build context and remain outside commits. Historical results stay in the external `classes/MobilityGPT-results` bundle and are not a runtime dependency or relocation target.

[License](repo/LICENSE)
