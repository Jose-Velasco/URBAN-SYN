# Trajectory diffusion package

This package is a local conditional extension of Diff-RNTraj. The current sparse
pipeline supports three default fields plus one opt-in field:

| Field | Kind | Default? | Encoder input |
| --- | --- | --- | --- |
| weather_category | categorical | yes | training-only code, 0 = Unknown |
| transportation_mode | categorical | yes | training-only code, 0 = Unknown |
| temperature | continuous | yes | training-normalized Celsius value plus mask |
| previous_stop_poi_category | categorical | opt-in | training-only code, 0 = Unknown |

The current entrypoint is the sparse diagnostic pilot. Regular conditional
training and a dedicated production generation CLI remain future work.

## Data flow and sources

FMM-converted trajectories first pass exact replay: stored road IDs and rates,
split membership, length buckets, and bucket indices must match the complete
conversion inputs. Verified sidecars then attach metadata without rewriting the
trajectory binaries. The denoiser receives road embeddings, normalized rates,
and the encoded condition. DDPM reverse sampling produces road-embedding vectors
and rates; cosine similarity decodes road IDs, and rates are postprocessed as
(clamp(rate, -1, 1) + 1) / 2.

| Model field | Source | Association |
| --- | --- | --- |
| transportation_mode | trajectory map and data/output/enriched_moves.parquet | move/trajectory label |
| weather_category | data/output/weather_enrichment.parquet | (uid, tid, departure date) |
| temperature | data/output/weather_enrichment.parquet | same three-key date join |
| previous_stop_poi_category | data/output/stops.parquet plus data/output/enriched_occasional.parquet | latest same-parent stop before move departure, then POI candidate policy |

Other columns in data/output are not automatically model inputs. A column becomes
an input only when a sidecar builder maps it to a declared field descriptor and
the checkpoint records that descriptor.

## Layout

~~~text
src/urban_syn/diffusion/
├── README.md
├── models/
│   ├── README.md
│   ├── __init__.py
│   ├── factory.py
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
    ├── road_inputs.py
    ├── sparse_pilot.py
    ├── audit_weather.py
    └── tests/
        ├── __init__.py
        ├── test_pilot.py
        └── test_sparse.py
~~~

[models/](models/README.md) documents the shared denoiser, losses, sampler,
decoding, and checkpoint interfaces. [conditional/](conditional/README.md)
documents alignment, weather and POI contracts, sidecars, and diagnostics.
The test package contains 16 regression and POI-policy tests.

## Required inputs

The current NYC run requires trajectory binaries
(gen_all/gen_train/gen_test eid_seqs.bin and rate_seqs.bin), road mappings
(extra_file/raw2new_rid.json, extra_file/new2raw_rid.json, and
raw_rn_dict.json), graph/graph_edge.edgelist,
graph/road_embed.txt, the trajectory-ID map, FMM matches, the held-out trip
list, and data/output/weather_enrichment.parquet. POI-enabled runs additionally
use data/output/stops.parquet and data/output/enriched_occasional.parquet.
The road mappings, graph, and embeddings are reused unchanged by conditioning.

## Three-field quickstart

Run from the repository root with a fresh output directory. Every command
specifies the Compose file, container working directory, and PYTHONPATH.

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

The weather-only runner has been retired. The historical weather-only metadata,
checkpoint, and pilot results remain under
outputs/diff_rntraj_weather_pilot/metadata/ and smoke/ for compatibility
reference; current diagnostics use only conditional.sparse_pilot.

## POI-enabled quickstart

The default three fields remain active; append the opt-in field and both POI
sources. Use a fresh output directory for the sidecar and pilot checkpoint.

~~~bash
COMPOSE=fmm_scripts/diffrntraj_version/docker-compose.yml
RUN=outputs/diff_rntraj_sparse_poi_$(date -u +%Y%m%dT%H%M%SZ)

docker compose -f "$COMPOSE" run --rm -T --user "$(id -u):$(id -g)" \
  -e PYTHONPATH=/workspace/URBAN-SYN/src -w /workspace/URBAN-SYN \
  diffrntraj-fmm python -m urban_syn.diffusion.conditional.sparse_metadata \
  --dataset data/NYC_rerun/20261102T040340Z \
  --trip-map fmm_scripts/diffrntraj_version/data/nyc_gps_points_fmm_trip_id_map_drive.csv \
  --fmm-matches fmm_scripts/diffrntraj_version/data/nyc_drive_fmm_match.csv \
  --test-trips fmm_scripts/nyc_test.csv \
  --weather data/output/weather_enrichment.parquet \
  --poi data/output/enriched_occasional.parquet \
  --stops data/output/stops.parquet \
  --fields weather_category transportation_mode temperature previous_stop_poi_category \
  --output "$RUN/metadata"

docker compose -f "$COMPOSE" run --rm -T --user "$(id -u):$(id -g)" \
  -e PYTHONPATH=/workspace/URBAN-SYN/src -w /workspace/URBAN-SYN \
  diffrntraj-fmm python -m urban_syn.diffusion.conditional.sparse_pilot \
  --dataset data/NYC_rerun/20261102T040340Z \
  --metadata-dir "$RUN/metadata" --output "$RUN/smoke"
~~~

The fresh POI artifact location used by the current diagnostic is
outputs/diff_rntraj_sparse_poi_pilot/metadata/ and smoke/. It records 170
combined known moves and 432 combined known pieces: 157 training moves / 406
training pieces and 13 test moves / 26 test pieces. These are actual move- and
trajectory-level counts, not the unrelated stop-level missingness percentage.

Checkpoints store configuration, state dictionaries, field descriptors, and
input hashes. Reconstruct them with
from urban_syn.diffusion.models.factory import make_model
and load matching road inputs with
from urban_syn.diffusion.conditional.road_inputs import load_road_inputs
They require matching road mappings and embeddings. Reorganizing Python modules
requires no checkpoint migration, but v1, v2, and POI-enabled configurations remain distinct. No
regular conditional training or dedicated generation command is included yet.

See [model details](models/README.md) and
[conditioning contracts and diagnostics](conditional/README.md).

