# Conditional contracts and diagnostics

This directory owns conversion helpers, metadata sidecars, conditioning,
diagnostics, audit, and tests. The current workflow is schema v2 sparse
conditioning. Schema
v1 weather-only conditioning is its predecessor. See the [package overview](../README.md)
and [model documentation](../models/README.md).

## Alignment before enrichment

metadata.py and sparse_metadata.py first replay the existing conversion. They
read trajectory binaries, the trajectory-ID map, FMM matches, the saved road-ID
mapping, and held-out trip list. Every accepted row must reproduce stored road
IDs and moving rates, with identical split, length bucket, count, and
index_within_length. Duplicate or missing identities, malformed counts, and
drift fail closed before sidecars are written. This establishes source identity;
it is separate from enrichment.

Enrichment joins weather by (uid, tid, departure date), corresponding to the
weather key (uid, tid, DATE). Duplicate full keys are rejected. Date extraction
preserves the upstream naive-as-epoch convention, interpreted as UTC to match
weather DATE; it makes no original-timezone claim. A piece crossing midnight
uses its departure date. Split membership and bucket indices are preserved.
Sidecars and source inputs are bound by SHA-256 hashes.

This weather date join is separate from POI temporal association. POI move
departure is the earliest timestamp for the move in the complete trajectory-ID
map, before FMM filtering. The selector then finds the latest same-(uid, tid)
stop departure at or before that move, applies the 900-second gap policy, and
validates parent identity plus both stop timestamps against the enrichment.

Historical manifests report 19,462 aligned trajectories: 15,794 train and 3,668
test. They record 512 empty_opath FMM exclusions and 58 aligned records with
missing weather source. The weather audit shows the 58 interpolation-created
missing-weather pieces came from 15-second interpolation across a long source
gap; they are separate from the 512 FMM exclusions. All 58 departures have zero
original move points and one crosses midnight.

## Schema

metadata.py remains the weather-only schema-v1 sidecar predecessor. The current
multi-field preparation and diagnostics are sparse_metadata.py and
sparse_pilot.py; the retired weather-only pilot runner is not an entrypoint.

V1 has one training-only categorical field, weather_category. Code 0 is
<unknown>; the historical manifest maps heavy rain=1, light rain=2,
moderate rain=3, sunny=4. Missing source and training-unseen categories remain
aligned with code 0

Current sparse workflow:

| Field | Kind | Source | Contract |
| --- | --- | --- | --- |
| weather_category | categorical | weather DESCRIPTION | training-only vocabulary, code 0 unknown |
| transportation_mode | categorical | trajectory-map label | training-only vocabulary, code 0 unknown |
| temperature | continuous | weather TAVG_C | training-only Celsius mean/scale |
| previous_stop_poi_category | categorical, optional | POI-enriched stop candidates | training-only vocabulary, code 0 unknown |

Sparse means field-wise missingness and partial requests are first-class; it
does not require complete values. Each descriptor records source, vocabulary or
normalization, and exact_zero missing contribution. Rows carry
field_available and field_status. Statuses are known, missing_source,
unseen_training_category, or no_training_support.

Vocabularies and temperature statistics use known gen_train rows only. The
historical v2 manifest reports temperature mean 16.601950940518556, population
scale 8.419925204480267, and 15,736 observed training rows. Zero scale falls
back to 1; no training support leaves a field unavailable.

### Frozen previous-stop POI policy

The POI field is opt-in; the default remains the original three fields. Enable it
with `--poi`, `--stops`, and `--fields ... previous_stop_poi_category`.
Move departure is the earliest `start_timestamp` for that move in the complete,
unfiltered trajectory-ID map, before FMM exclusions. Within the same `(uid, tid)`
the policy selects the latest stop whose departure is no later than that move.
The context gap must satisfy `0 <= move_start - stop_departure <= 900` seconds.
Future stops and gaps above 900 seconds are Unknown. If the latest stop has no
usable POI enrichment, the selector does not fall back to an older stop. Every
accepted FMM piece of a move inherits the move-level assignment, including when
an earlier piece of that move failed FMM.

POI-to-stop association is validated by parent identity and both stop timestamps;
ambiguous associations and conflicting duplicate candidate identities become
Unknown. Usable candidates have a known category, finite nonnegative stored
distance, and distance no greater than 50. Candidates are ranked without
rounding by `(distance, element_type, numeric_osmid, category)`. Conflicting
duplicates are rejected rather than resolved by input order. Unknown reason codes
include `no_preceding_stop`, `excessive_gap`, `ambiguous_association`,
`unavailable_candidate`, and `unseen_training_category`.

The manifest records `distance_crs=EPSG:3857`, `distance_metric=planar`,
`distance_units=projected_metres`, `max_distance=50`, the complete selection
policy, and hashes for the POI and stop inputs. Selected stop/POI identity,
stop/POI coordinates, dates, stored distance, and time gap are provenance
columns only; the encoder receives only the category code and availability mask.
The 50-metre cutoff is a projected
metre cutoff for this existing batch. A future raw-enrichment pipeline must use
a separately versioned 50-ground-metre WGS84 geodesic retrieval policy and
retrieve candidates afresh; the existing projected-radius candidate set is not
complete for that policy. Coordinate or source changes can legitimately change
the nearest candidate, so deterministic tie-breaking cannot prevent that.

There is currently no validation split. Any future validation split must be
grouped before fitting vocabularies, statistics, or other training-derived
artifacts. Before writing sidecars, the existing train/test partition is asserted
disjoint both by parent trip and by user; no user or parent is moved between
splits by POI enrichment.

## Conditioning contract

SparseMetadataEncoder adds known field contributions into [B,C]. Categorical
fields use separate embeddings; temperature uses a linear projection. The
complete contribution, including bias, is masked for unavailable fields, making
all-missing input exactly zero. Known fields remain active when other fields are
missing. Additivity applies to the condition vector, not final predictions.

Training-only dropout drops all fields in a record with probability 0.1 and each
field independently with probability 0.1. Evaluation and generation disable it.
This exposes null and partial paths; it is not classifier-free sampling guidance
(no two-pass score mixing or guidance scale).

Explicit full, partial, and empty requests use encode_requests:

Use a checkpoint whose metadata_fields includes the requested optional POI field;
the three-field checkpoint accepts only weather, mode, and temperature.

~~~python
from urban_syn.diffusion.conditional.conditioning import encode_requests

metadata = encode_requests([
    {"weather_category": "sunny", "transportation_mode": "bus", "temperature": 18.0},
    {"transportation_mode": "walk", "previous_stop_poi_category": "cafe"},
    {},
], checkpoint["metadata_fields"])
road_ids, rates = model.generate_data(graph, embeddings, 3, 20, 64, metadata)
~~~

Omitted or None values are unavailable; unknown categories map to unknown and
mask. Unsupported names raise an error. The model’s legacy argument is still
named weather, but v2 accepts this metadata dictionary or None.

## Diagnostics and historical results
**Context & Scope:** These are historical pilot results sourced from existing reports, not regenerated artifacts. They serve as successful conditioning diagnostics and trainability checks, rather than evaluations of trajectory quality, road realism, or conditional fidelity.

---

## 1. Pilot Results

### Pilot v1: Weather Conditioning ($B = 2, L = 20$)
Executed using one optimizer step across two length-20 examples.

* **Batch Shape:** $[2, 20]$ for both ID and rate outputs
* **Baseline Measurements:**
  * **Noise:** `1.0489805936813354`
  * **Connectivity:** `24.0`
  * **$x_0$:** `0.01071049552410841`
* **Gradients & Sensitivity:**
  * **Weather Gradient Norm:** `0.0008343211375176907`
  * **Condition-Shuffle Prediction Delta:** `0.09675382077693939`
  * **Controlled-Sampling Delta:** `0.018283426761627197`

---

### Pilot v2: Sparse Multi-Field Conditioning ($B = 5, L = 20$)
Executed using one optimizer step across five length-20 examples evaluating variable masking configurations (*fully known*, *mode-only*, and *all-unknown*).

* **Batch Shape:** $[5, 20]$ for both ID and rate outputs
* **Encoder Field Gradient Norms ($\ell_2$):**
  * **Weather:** `0.0010924155358225107`
  * **Mode:** `0.0012009519850835204`
  * **Temperature:** `0.002279658103361726`
* **Field Prediction Deltas:**
  * **Weather:** `0.15676634013652802`
  * **Mode:** `0.08280309289693832`
  * **Temperature:** `0.11593136191368103`
* **Conditioning Sensitivity & Updates:**
  * **Condition-Shuffle Delta:** `0.18998119235038757`
  * **Controlled-Sampling Delta:** `0.03839296102523804`
  * **Missingness Semantics:** All-unknown mask produced exact-zero conditioning as intended.
  * **Parameters:** Successful parameter updates confirmed after the optimizer step.

The reported sparse field gradient values are sums of the individual parameter
gradient norms within each field encoder, as computed by the pilot. They are
not one norm of a concatenated gradient vector.

### Historical POI-enabled v2 diagnostic

The fresh POI-enabled sidecar reports 157 known training moves / 406 training
pieces and 13 known test moves / 26 test pieces: 170 moves / 432 pieces
combined. The POI sparse pilot passed its one-update conditioning diagnostic,
including a POI field gradient norm of `0.0006304843700490892`, a POI prediction
delta of `0.0945969969034195`, condition-shuffle delta `0.30694830417633057`,
and controlled-sampling delta `0.034580230712890625`. These are historical
diagnostic results, not learned POI fidelity or trajectory-quality results.

---

## 2. Metric Definitions & Verification Criteria

| Metric / Check | Definition & Acceptance Criteria |
| :--- | :--- |
| **Gradient Norms** | For each field encoder, the finite non-zero diagnostic is the sum of the individual parameter-gradient Euclidean ($\ell_2$) norms. |
| **Parameter Updates** | Evaluation of encoder weight differences ($\Delta \theta \neq 0$) before and after a single optimizer step. |
| **Prediction Deltas** | Maximum absolute elementwise difference ($\|y_{c_1} - y_{c_2}\|_\infty$) on fixed noisy inputs under altered conditioning. |
| **Controlled-Sampling Differences** | Maximum absolute elementwise difference computed using a fixed random generator. |
| **Random-Draw Replay** | Verification of bitwise identical samples given an identical random seed and conditioning vector. |
| **Shape Conformance** | Strict tensor shape enforcement of $[B, L]$ for both IDs and rates. |
| **Checkpoint Restoration** | Full model reconstruction from configuration files and strict state dictionary reload, producing identical predictions on fixed inputs. |

---

## 3. Scope of Validation

* **Verified:**
  * Numerical stability and computational wiring
  * Missingness and masking semantics (e.g., exact-zero conditioning)
  * End-to-end model trainability
  * Deterministic checkpoint reload and inference
* **Not Verified (Out of Scope):**
  * Conditional fidelity
  * Unbiased marginal behavior under informative missingness
  * Kinematic/road realism
  * Downstream trajectory quality

## Commands, audit, and tests

The [package quickstart](../README.md) contains Docker commands for both
sidecar builders, the sparse pilot, and the combined 16-test suite. The
weather-only runner is retired; its historical results remain documented above.
The read-only
weather audit writes a new report directory:

The POI sidecar is written to a fresh directory and reuses the unchanged FMM
trajectory binaries, road mappings, graph, and embeddings:

~~~bash
docker compose -f fmm_scripts/diffrntraj_version/docker-compose.yml \
  run --rm -T --user "$(id -u):$(id -g)" \
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
  --output outputs/diff_rntraj_sparse_poi_<fresh-run>/metadata
~~~

The current fresh POI artifact reports 157 known training moves, 13 known test
moves, and 170 known moves combined, with 432 known gen_all trajectories (406
train, 26 test). These are move- and trajectory-level coverage figures; they are
not the unrelated 37.5% stop-level missingness statistic. The accompanying POI
sparse pilot passed its conditioning diagnostic
with a nonzero POI gradient norm and a condition-sensitive prediction delta.
That is diagnostic success, not learned POI fidelity.

~~~bash
docker compose -f fmm_scripts/diffrntraj_version/docker-compose.yml \
  run --rm -T --user "$(id -u):$(id -g)" \
  -e PYTHONPATH=/workspace/URBAN-SYN/src -w /workspace/URBAN-SYN \
  diffrntraj-fmm python -m urban_syn.diffusion.conditional.audit_weather \
  --dataset data/NYC_rerun/20261102T040340Z \
  --metadata-dir outputs/diff_rntraj_weather_pilot/metadata \
  --moves data/output/enriched_moves.parquet \
  --output outputs/diff_rntraj_weather_pilot/weather_audit_rerun
~~~

The test modules are
urban_syn.diffusion.conditional.tests.test_pilot and
urban_syn.diffusion.conditional.tests.test_sparse. They cover conversion drift,
duplicate/date joins, split and bucket alignment, training-only statistics,
missing/partial fields, dropout, exact-zero paths, and checkpoint/generation
smoke behavior.

