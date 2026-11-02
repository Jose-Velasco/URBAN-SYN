# Trajectory diffusion package

The current workflow is the sparse metadata Diff-RNTraj pipeline. The weather-only
pipeline is its predecessor and remains available for compatibility.

FMM-converted trajectories flow through exact conversion replay, verified metadata
sidecars, conditional denoising, DDPM reverse sampling, and road-ID/rate output.
Replay reproduces every stored road ID, normalized rate, split, length bucket, and
bucket index before enrichment. The sidecar then joins departure-date metadata
without rewriting the trajectory dataset. Sampling produces road-embedding
vectors and rates; road IDs are decoded by cosine similarity against the road
embedding table, and rates are postprocessed as (clamp(rate, -1, 1) + 1) / 2.

The pilots demonstrate conditioning diagnostics and checkpoint reconstruction, not
trajectory quality. Documentation verification needs no full training,
preprocessing rerun, sidecar rebuild, or FMM rerun.

## Layout

~~~text
src/urban_syn/diffusion/
├── README.md
├── models/
│   ├── README.md
│   ├── __init__.py
│   ├── model.py
│   ├── diff_module.py
│   └── diff_util.py
└── conditional/
    ├── README.md
    ├── __init__.py
    ├── conditioning.py
    ├── conversion.py
    ├── metadata.py
    ├── sparse_metadata.py
    ├── pilot.py
    ├── sparse_pilot.py
    ├── audit_weather.py
    └── tests/
        ├── __init__.py
        ├── test_pilot.py
        └── test_sparse.py
~~~

[models/](models/README.md) documents the architecture and equations.
[conditional/](conditional/README.md) documents data contracts, sidecars, and
diagnostics. conditional/tests/ contains the 13 regression tests.

## Inputs and supported conditions

The NYC run requires trajectory binaries (gen_all, gen_train, gen_test
eid_seqs.bin and rate_seqs.bin), extra_file/raw2new_rid.json,
extra_file/raw_rn_dict.json, graph/graph_edge.edgelist,
graph/road_embed.txt, the trajectory-ID map, FMM matches, the held-out trip
list, and data/output/weather_enrichment.parquet.

Current v2 fields are categorical weather_category, categorical
transportation_mode, and continuous temperature in degrees Celsius. These are
departure-date enrichment values, not pointwise GPS measurements.

## Docker quickstart

Run from the repository root. Every command supplies the Compose path, container
working directory, and PYTHONPATH. Use a fresh output directory per run.

~~~bash
COMPOSE=fmm_scripts/diffrntraj_version/docker-compose.yml
RUN=outputs/diff_rntraj_sparse_$(date -u +%Y%m%dT%H%M%SZ)

docker compose -f "$COMPOSE" run --rm -T --user "$(id -u):$(id -g)" \
  -e PYTHONPATH=/workspace/URBAN-SYN/src -w /workspace/URBAN-SYN \
  diffrntraj-fmm python -m urban_syn.diffusion.conditional.sparse_metadata \
  --dataset data/NYC_rerun/20261102T040340Z \
  --trip-map fmm_scripts/diffrntraj_version/data/nyc_gps_points_fmm_trip_id_map_drive.csv \
  --fmm-matches fmm_scripts/diffrntraj_version/data/nyc_drive_fmm_match.csv \
  --test-trips fmm_scripts/nyc_test.csv \
  --weather data/output/weather_enrichment.parquet \
  --output "$RUN/metadata"

docker compose -f "$COMPOSE" run --rm -T --user "$(id -u):$(id -g)" \
  -e PYTHONPATH=/workspace/URBAN-SYN/src -w /workspace/URBAN-SYN \
  diffrntraj-fmm python -m urban_syn.diffusion.conditional.sparse_pilot \
  --dataset data/NYC_rerun/20261102T040340Z \
  --metadata-dir "$RUN/metadata" --output "$RUN/smoke"

docker compose -f "$COMPOSE" run --rm -T --user "$(id -u):$(id -g)" \
  -e PYTHONPATH=/workspace/URBAN-SYN/src -w /workspace/URBAN-SYN \
  diffrntraj-fmm python -m unittest \
  urban_syn.diffusion.conditional.tests.test_pilot \
  urban_syn.diffusion.conditional.tests.test_sparse -v
~~~

For the predecessor weather-only workflow, use another fresh directory:

~~~bash
RUN=outputs/diff_rntraj_weather_$(date -u +%Y%m%dT%H%M%SZ)

docker compose -f "$COMPOSE" run --rm -T --user "$(id -u):$(id -g)" \
  -e PYTHONPATH=/workspace/URBAN-SYN/src -w /workspace/URBAN-SYN \
  diffrntraj-fmm python -m urban_syn.diffusion.conditional.metadata \
  --dataset data/NYC_rerun/20261102T040340Z \
  --trip-map fmm_scripts/diffrntraj_version/data/nyc_gps_points_fmm_trip_id_map_drive.csv \
  --fmm-matches fmm_scripts/diffrntraj_version/data/nyc_drive_fmm_match.csv \
  --test-trips fmm_scripts/nyc_test.csv \
  --weather data/output/weather_enrichment.parquet \
  --output "$RUN/metadata"

docker compose -f "$COMPOSE" run --rm -T --user "$(id -u):$(id -g)" \
  -e PYTHONPATH=/workspace/URBAN-SYN/src -w /workspace/URBAN-SYN \
  diffrntraj-fmm python -m urban_syn.diffusion.conditional.pilot \
  --dataset data/NYC_rerun/20261102T040340Z \
  --metadata-dir "$RUN/metadata" --output "$RUN/smoke"
~~~

Historical artifacts are in outputs/diff_rntraj_sparse_pilot/ and
outputs/diff_rntraj_weather_pilot/, each with metadata/, smoke/, and pilot.pt
where applicable. Checkpoints store configuration and state dictionaries plus
schema and input hashes. They reconstruct with make_model(config) and need
matching road mappings and embeddings; moving Python modules requires no
migration. v1, v2, and unconditional architectures are distinct.

See [models and equations](models/README.md) and
[conditional contracts and diagnostics](conditional/README.md).

