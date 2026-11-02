# Conditional contracts and diagnostics

This directory owns conversion helpers, metadata sidecars, conditioning, pilots,
audit, and tests. The current workflow is schema v2 sparse conditioning. Schema
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

Historical manifests report 19,462 aligned trajectories: 15,794 train and 3,668
test. They record 512 empty_opath FMM exclusions and 58 aligned records with
missing weather source. The weather audit shows the 58 interpolation-created
missing-weather pieces came from 15-second interpolation across a long source
gap; they are separate from the 512 FMM exclusions. All 58 departures have zero
original move points and one crosses midnight.

## Schema v1 and v2

V1 has one training-only categorical field, weather_category. Code 0 is
<unknown>; the historical manifest maps heavy rain=1, light rain=2,
moderate rain=3, sunny=4. Missing source and training-unseen categories remain
aligned with code 0. V1 has no continuous normalization.

V2 is the current sparse workflow:

| Field | Kind | Source | Contract |
| --- | --- | --- | --- |
| weather_category | categorical | weather DESCRIPTION | training-only vocabulary, code 0 unknown |
| transportation_mode | categorical | trajectory-map label | training-only vocabulary, code 0 unknown |
| temperature | continuous | weather TAVG_C | training-only Celsius mean/scale |

Sparse means field-wise missingness and partial requests are first-class; it
does not require complete values. Each descriptor records source, vocabulary or
normalization, and exact_zero missing contribution. Rows carry
field_available and field_status. Statuses are known, missing_source,
unseen_training_category, or no_training_support.

Vocabularies and temperature statistics use known gen_train rows only. The
historical v2 manifest reports temperature mean 16.601950940518556, population
scale 8.419925204480267, and 15,736 observed training rows. Zero scale falls
back to 1; no training support leaves a field unavailable.

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

~~~python
from urban_syn.diffusion.conditional.conditioning import encode_requests

metadata = encode_requests([
    {"weather_category": "sunny", "transportation_mode": "bus", "temperature": 18.0},
    {"transportation_mode": "walk"},
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

---

## 2. Metric Definitions & Verification Criteria

| Metric / Check | Definition & Acceptance Criteria |
| :--- | :--- |
| **Gradient Norms** | Finite, non-zero Euclidean ($\ell_2$) norm of available gradients for each field encoder. |
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
sidecar builders, both pilots, and the combined 13-test suite. The read-only
weather audit writes a new report directory:

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

