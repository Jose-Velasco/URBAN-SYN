# Optional metadata in the joint GNN pipeline

The canonical model is `urban_syn.diffusion.model.Diff_RNTraj`, constructed by
`urban_syn.experiment`. The `diffusion/models/` files and v1/v2 binary sidecars
remain historical references. Their pilot weights cannot be loaded into the
joint GNN model. The current prepared metadata format is version 3.

Prepare fresh metadata from the exact observations used to produce FMM input:

```bash
PYTHONPATH=src python -m urban_syn.diffusion.conditional.prepared_metadata \
  --train fmm_scripts/outputs/urban_syn_ready/nyc_15s_enriched_moves_train_prepared_road_sequences.parquet \
  --test fmm_scripts/outputs/urban_syn_ready/nyc_15s_enriched_moves_test_prepared_road_sequences.parquet \
  --mapping fmm_scripts/data/nyc_15s_fmm_trip_id_map.csv \
  --observations PATH_TO_CORRESPONDING_OBSERVATIONS.parquet \
  --weather data/output/weather_enrichment.parquet \
  --identity uid tid move_id piece \
  --output outputs/prepared_metadata_v3
```

For move-only inputs, use `--identity uid tid move_id`. Use the sampled
observations for sampled FMM inputs and the unsampled observations otherwise.
`--timestamp` defaults to `datetime`; `--mode` defaults to `label`. Labels and
point counts come from these observations. Conflicting labels, incomplete or
duplicate identities, count mismatches, uncovered sequences, and shared train/
test users fail preparation. Raw road IDs, including zero, remain unchanged;
the dataset maps raw zero to a positive model ID. Upstream partitions remain fixed.

Optional POI uses `--poi data/output/enriched_occasional.parquet --stops
 data/output/stops.parquet`. Select `previous_stop_poi_category` in the enabled
configuration's fields to train with it. Each move inherits one category from the
latest same-parent stop departing at or before its complete-map move departure,
with a gap of at most 900 seconds. Association validates both stop timestamps.
Candidates require finite nonnegative stored distance <=50 projected metres
(EPSG:3857), and rank by distance, element type, numeric OSM ID, and category.
Ambiguous associations become unknown; selection never falls back to older stops.

Weather joins by `(uid, tid, departure date)` and preserves the upstream UTC
naive-as-epoch date convention. Temperature uses departure-date `TAVG_C` in
Celsius. No instantaneous-temperature or original-timezone interpretation is added.

The preparation manifest records alignment, source SHA-256 hashes, sidecar hashes,
and original partition IDs. Loading rejects changed inputs or sidecars. Validation
selection uses the upstream grouped trip split. Vocabulary and temperature
population mean/scale fit only optimizer-training indices; zero scale uses one.
The run's `metadata_manifest.json` and checkpoint add these fitted field
definitions, statistics, architecture, and exact training/validation IDs. Resume
requires an identical schema, preprocessing, and artifact provenance.

Run the paired configurations from the repository root:

```bash
PYTHONPATH=src python -m urban_syn.experiment train --config configs/experiments/metadata_conditioning/disabled.yaml
PYTHONPATH=src python -m urban_syn.experiment train --config configs/experiments/metadata_conditioning/enabled.yaml
```

Both configurations share partitions, seed, model settings, and training budget.
Conditioning defaults to disabled. Enabled defaults are dimension 8, record
dropout 0.1, and independent field dropout 0.1. The encoder sums field contributions,
broadcasts over sequence length, and concatenates before input projection.
Unavailable fields contribute exactly zero, including projection bias. Unknown
categories and omitted request fields are unavailable. Evaluation disables dropout.

For generation set `generation.checkpoint` to a joint checkpoint, supply one
`generation.metadata_requests` dictionary per generated trajectory, then run:

```bash
PYTHONPATH=src python -m urban_syn.experiment generate --config YOUR_GENERATION_CONFIG.yaml
```

Requests are ordered by increasing sequence length, then trajectory order within
that length. For example `[{weather_category: sunny, temperature: 18.0}, {}]`
requires two trajectories. Values use checkpoint vocabularies and normalization.
Unsupported fields fail. Set `generation.mask_metadata: true` to suppress all
fields. Compare known and masked requests using the same checkpoint, request
list, experiment seed, lengths, and batch size, with different output names.
This is ordinary DDPM sampling; there is no classifier-free guidance mixing.
The Python API also accepts `metadata=` from `encode_requests(requests,
model.metadata_schema['fields'])` in `model.generate_data`.

Run the self-contained compatibility smoke in a fresh directory:

```bash
PYTHONPATH=src python -m urban_syn.diffusion.conditional.sparse_pilot --output outputs/metadata_joint_smoke_NEW
PYTHONPATH=src python -m pytest src/urban_syn/diffusion/conditional/tests -q
```

The smoke writes synthetic prepared inputs, metadata, a fitted manifest, separate
enabled/disabled/frozen joint checkpoints, and a report. It checks one finite
update, encoder/GNN gradients, frozen weights, checkpoint reload, positive
sampled model IDs, and same-seed known/masked reverse outputs. These synthetic
results establish compatibility, not conditional quality. Existing output
folders are never overwritten by preparation or this smoke.

## Compatibility with the GNN pipeline and on/off comparisons

The legacy fixed-embedding conditioner expects converted binary length buckets,
a saved legacy road-ID vocabulary, and a separate fixed embedding table. The
joint pipeline instead reads prepared FMM sequence Parquets with raw `edge_ids`
and maps those IDs into the current graph's node ordering through the canonical
`RoadSequenceDataset`. Model IDs are one-based, with zero reserved for padding;
raw road zero is still valid. The joint GNN learns the road embeddings used by
diffusion. Legacy binary indices and embedding tables cannot substitute for
these inputs or establish alignment to the new prepared sequences.

Preparation uses the exact sampled or unsampled observation Parquet supplied to
FMM and its complete identity map. Configure every identity component, including
`piece` for piece-level trajectories. Map and observation identities must cover
each other, and each map identity must identify one FMM ID. If observations also
carry FMM IDs, they must agree with that map. Counts must match prepared sequence
lengths. Missing, blank, duplicate, or ambiguous identities fail explicitly;
metadata is attached by `fmm_id` and restored to the exact input sequence order
after alignment and enrichment. Feature values never determine row membership.

With conditioning enabled, `MetadataDataset.dataset` remains the canonical
trajectory dataset. Its road tensors and rates are returned directly; its FMM
IDs and lengths remain authoritative. The existing validation selection and
length-bucket sampler receive the same indices, lengths, split seed, and epoch
seed for enabled and disabled runs. Metadata values and availability masks are
additional batch members and enter only the diffusion encoder. They do not enter
the GNN or change graph features, graph preparation, FMM processing, road mapping,
or trajectory loading. With conditioning disabled, construction requires no
metadata artifacts and uses the original denoiser projection.

Missing weather, mode, temperature, or optional POI values preserve trajectories.
Unknown categories and unavailable continuous values have false masks and exact
zero field contributions, including projection bias. Even an entirely missing
metadata batch leaves trajectory tensors and batch order intact. Categorical
vocabularies and temperature normalization fit only the selected training rows;
validation, test, and raw generation requests use those frozen definitions.
Checkpoint schema checks continue to protect fitted preprocessing and provenance.

The paired configurations share source artifacts, graph and training settings,
validation split seed, and batch-order seeds. Only the conditioning switch and
experiment name differ. Run independently trained on/off models with:

```bash
PYTHONPATH=src python -m urban_syn.experiment train --config configs/experiments/metadata_conditioning/disabled.yaml
PYTHONPATH=src python -m urban_syn.experiment train --config configs/experiments/metadata_conditioning/enabled.yaml
```

The GNN remains trainable in both runs. Identical trajectory inputs do not imply
identical learned weights: the conditioning changes diffusion predictions and
the gradients flowing back through the joint model.

For an additional diagnostic, make two generation configs from the enabled
configuration. Set the same `generation.checkpoint`, `experiment.seed`,
`generation.num_trajectories`, `generation.batch_size`, and
`generation.metadata_requests` in both. Use `generation.mask_metadata: false`
for the known config and `true` for the masked config, with different output
names. Then run:

```bash
PYTHONPATH=src python -m urban_syn.experiment generate --config YOUR_KNOWN_CONFIG.yaml
PYTHONPATH=src python -m urban_syn.experiment generate --config YOUR_MASKED_CONFIG.yaml
```

This suppresses metadata on one conditioned checkpoint with identical sampling
seeds. It measures sensitivity of that checkpoint; it differs from independently
training an unconditional model with the disabled configuration above.

The original fixed-embedding implementation remains under
`src/urban_syn/diffusion/models/`. Its legacy regression tests remain in
`conditional/tests/test_pilot.py` and `conditional/tests/test_sparse.py`. They
exercise the original model and encoder contracts and remain separate from
canonical joint-model construction. The canonical compatibility checks are in
`conditional/tests/test_prepared.py`. Run all three together with:

```bash
PYTHONPATH=src python -m pytest src/urban_syn/diffusion/conditional/tests -q
```

These checks use synthetic fixtures for exact identity/order assertions,
multiple lengths and remainder batches, three shuffled epochs, training and
validation membership, all-missing metadata, train-only preprocessing, finite
joint updates, frozen GNN behavior, sampling, and checkpoint reload/resume.
They isolate compatibility properties with known expected identities. Real-data
verification and new real-data artifact generation remain deferred because the
required prepared train/test Parquets and matching FMM observations are not
available in this checkout. Existing pilot artifacts remain preserved.

## Standalone fixed-embedding diffusion

`python -m urban_syn.diffusion.conditional.standalone` provides `check`, `train`,
`generate`, and `evaluate`. This runner adapts the retained denoiser in
`diffusion/models/`; it never constructs a GNN. The existing joint experiment
entrypoint, graph preparation, FMM algorithms, models, and artifacts remain
separate. Fixed embeddings are supplied data buffers, excluded from the
optimizer. A frozen GNN still constructs an encoder and computes embeddings
from graph features and weights; this standalone runner loads the embedding
table directly and has no such encoder.

Data flow is prepared sequence Parquets → explicit raw-road embedding vocabulary
→ fixed vectors plus normalized rates → retained diffusion denoiser. Trajectory
FMM IDs must be unique positive integers. Raw road zero is valid and maps to a
positive model ID; model zero is padding and is excluded from generated road
candidates. Neither absent roads nor absent features filter trajectories. Missing
roads produce `missing_roads.json` and stop the run; missing metadata produces
false masks. The unchanged canonical dataset and length-bucket loader preserve
FMM IDs, rates, trajectory lengths, and ordering across the conditioning toggle.

Use the exact FMM preparation command documented above to produce fresh v3
metadata. The standalone configurations read those same sequence files, identity
map, and `outputs/prepared_metadata_v3`. Disabled training does not read metadata
artifacts. Enabled training fits vocabularies, normalization, and temperature
quartile boundaries exclusively on its training indices. Validation is selected
from the upstream training split, grouped by parent trip; the test split remains
reserved for final reporting, with train/test user overlap rejected.

### Matching fixed embedding inputs

Supply an NPZ with these arrays (no pickled objects):

| Array | Meaning |
| --- | --- |
| `edge_ids` | Unique nonnegative raw canonical road integers, one per embedding row |
| `embeddings` | Finite `[N,D]` float matrix in exactly that vocabulary order |
| `transitions` | Integer `[E,2]` directed valid transitions, expressed in those raw road IDs |
| `network_sha256` | Scalar string containing SHA-256 of the exact FMM network file |

`paths.network` must be the exact network used for those FMM inputs. The runner
checks its hash against the embedding bundle, vocabulary uniqueness, transition
coverage, finite embeddings, and complete train/test sequence road coverage. It
does not infer network identity from coincident road numbers. Optional
`paths.geometry` must have that same network hash and supply all vocabulary road
geometries; `paths.geometry_layer` defaults to `roads`.

For an existing word2vec text table and explicit JSON vocabulary, package only
with verified provenance from the embedding producer:

```bash
PYTHONPATH=src python -m urban_syn.diffusion.conditional.standalone.pack_embeddings \
  --embedding-text MATCHING/road_embed.txt \
  --vocabulary MATCHING/new2raw_rid.json --vocabulary-index-base 1 \
  --transitions MATCHING/raw_transitions.csv \
  --network fmm_scripts/data/road_network/nyc.gpkg \
  --provenance MATCHING/embedding_producer_provenance.json \
  --output outputs/fixed_embeddings/matching_network.npz
```

The transitions CSV has raw integer `source,target` columns. The producer JSON
must contain `embedding_text_sha256`, `vocabulary_sha256`, `transitions_sha256`,
and `network_sha256` for those exact files. Text row IDs must be zero-based;
the vocabulary index base specifies the explicit row-to-raw-ID association.
This utility verifies and packages an existing table; it does not train one.
Do not manufacture new-network provenance for an old table. The table in
`data/NYC/graph/road_embed.txt` and `data/NYC/extra_file/new2raw_rid.json` has not
been verified against the new FMM network or sequence coverage. Automatic reuse
is disabled. If its mapping/network is incompatible or provenance unavailable,
a matching fixed embedding table and producer provenance must be generated
using the actual new FMM network before proceeding.

### Commands and paired runs

Run from the repository root in an environment containing the repository's
Python dependencies, including pandas/pyarrow, PyTorch and PyYAML:

```bash
export PYTHONPATH="$PWD/src"
python -m urban_syn.diffusion.conditional.standalone check --config configs/experiments/metadata_conditioning/standalone/smoke_enabled.yaml --device cpu
python -m urban_syn.diffusion.conditional.standalone train --config configs/experiments/metadata_conditioning/standalone/smoke_disabled.yaml --device cuda:0
python -m urban_syn.diffusion.conditional.standalone train --config configs/experiments/metadata_conditioning/standalone/smoke_enabled.yaml --device cuda:0
python -m urban_syn.diffusion.conditional.standalone train --config configs/experiments/metadata_conditioning/standalone/full_enabled.yaml --resume outputs/standalone/full_enabled/last.pt
```

Use the paired `full_disabled.yaml` and `full_enabled.yaml` for full training.
Configurations differ only in `conditioning.enabled` and output directory.
Shared denoiser parameters initialize identically. Metadata initialization uses
an isolated CPU RNG stream; the shared trajectory projection weights and bias
are copied into the wider conditioned projection. Batch-order, split, training
noise, evaluation noise, and generation seeds are explicit separate settings.
Noise MSE plus road-vector reconstruction MSE drives updates. Connectivity is a
separate evaluation diagnostic, never part of the optimization objective.

Submit standalone jobs separately from joint-GNN jobs:

```bash
PYTHON_BIN=/path/to/venv/bin/python sbatch hpc/standalone_metadata/run.sbatch check configs/experiments/metadata_conditioning/standalone/smoke_enabled.yaml
PYTHON_BIN=/path/to/venv/bin/python sbatch hpc/standalone_metadata/run.sbatch train configs/experiments/metadata_conditioning/standalone/full_disabled.yaml
PYTHON_BIN=/path/to/venv/bin/python sbatch hpc/standalone_metadata/run.sbatch train configs/experiments/metadata_conditioning/standalone/full_enabled.yaml
PYTHON_BIN=/path/to/venv/bin/python sbatch hpc/standalone_metadata/run.sbatch train configs/experiments/metadata_conditioning/standalone/full_enabled.yaml --resume outputs/standalone/full_enabled/last.pt
```

Slurm controls visible GPUs; `DEVICE=cuda:0` addresses the first allocated GPU.
Locally, use `CUDA_VISIBLE_DEVICES=1` and `--device cuda:0` to select physical
GPU 1. Checkpoints contain standalone model settings, fitted schema, source,
embedding, vocabulary and network fingerprints, split IDs, optimizer, and RNG
state. `best.pt` is selected solely by validation loss; `last.pt` supports epoch
boundary resume. Increase only the epoch budget to extend a run. Other
compatibility settings and artifact contents must match. A fresh training run
refuses to overwrite an existing `last.pt`.

```bash
python -m urban_syn.diffusion.conditional.standalone evaluate --config configs/experiments/metadata_conditioning/standalone/full_enabled.yaml --checkpoint outputs/standalone/full_enabled/best.pt --split test
python -m urban_syn.diffusion.conditional.standalone generate --config configs/experiments/metadata_conditioning/standalone/full_enabled.yaml --checkpoint outputs/standalone/full_enabled/best.pt --split test --metadata-mode correct
python -m urban_syn.diffusion.conditional.standalone generate --config configs/experiments/metadata_conditioning/standalone/full_enabled.yaml --checkpoint outputs/standalone/full_enabled/best.pt --split test --metadata-mode masked
python -m urban_syn.diffusion.conditional.standalone evaluate --config configs/experiments/metadata_conditioning/standalone/full_disabled.yaml --checkpoint outputs/standalone/full_disabled/best.pt --split test --report-metadata outputs/prepared_metadata_v3
```

Use `--split validation` for development. Evaluation performs correct, all-masked,
and globally shuffled metadata diagnostics using identical sampled noise and
timesteps. Whole metadata records, including masks, shuffle together. Generation
uses a separate identical seed for each variant, excludes padding, and chunks
both query vectors and road candidates during cosine decoding. Output sequence
Parquets contain canonical raw `edge_ids`, `[0,1]` rates, lengths, and the
held-out reference `fmm_id` used for that generation request. Matching geometry
also enables GPS observation Parquet export.

### Evaluation interpretation and compatibility

Pooled generation uses exactly one request per held-out trajectory with its
original length and metadata composition. Reports include trajectory counts,
per-field unavailable counts (including training-unseen categories), noise and
road reconstruction MSE, invalid directed-transition fraction, road-visitation
Jensen–Shannon divergence in natural-log units, unique-road-sequence fraction,
and road token entropy. Connectivity counts **all consecutive observation pairs**
within each trajectory in its denominator, including repeated-road pairs. A pair
is valid when both observations are on the same road, or when the directed road
change occurs in the supplied transition table. Only disconnected road changes
count in the numerator; no cross-trajectory pairs are formed. For a set with no
observation pairs the fraction is unavailable (`null`). Metrics are also stratified by the original requested
categories and training-defined temperature bins. Temperature bins extend the
outer training quantile intervals to cover held-out extremes. Missing or unseen
values form an explicit group. Disabled checkpoints can use `--report-metadata`
for the same strata without requiring metadata during their training.

When matching geometry is available, distance reports compare means and
quantile-based Wasserstein distance in metres for the generated versus reference
GPS observation-to-observation path lengths. These are displacement-based path
lengths, not complete routed road distances. Without geometry, distance metrics
are explicitly unavailable. MSE averages within each trajectory and then across trajectories. Tiny group counts and finite smoke losses are
compatibility evidence; they do not establish conditional quality. Correct/
masked/shuffled diagnostics on one conditioned checkpoint differ from the
separate independently trained on/off model comparison.

| Artifact | Accepted path |
| --- | --- |
| Legacy binary trajectories, fixed-embedding checkpoints | Original `diffusion/models/` loader and legacy tests only; preserved |
| Joint-GNN checkpoints | Existing `urban_syn.experiment` runner only; preserved |
| Prepared sequence Parquets and v3 metadata | Shared aligned inputs; no conversion to legacy binaries needed |
| Verified matching fixed NPZ table | Standalone input; immutable in both conditioning modes |
| `standalone-fixed-v1` checkpoints | New standalone resume/generation only; no legacy resume claim |

Synthetic end-to-end tests in `conditional/tests/test_standalone.py` exercise
explicit road-zero mapping, positive trajectory IDs, missing-road and network
failures, immutable embeddings, initialization parity, identical on/off inputs,
missing metadata, checkpoint reload and exact resumed-versus-uninterrupted
training, generation, GPS reconstruction, and evaluation:

```bash
PYTHONPATH=src python -m pytest src/urban_syn/diffusion/conditional/tests -q
```

Real-data smoke remains deferred. `check` writes `missing_inputs.json` with exact
missing paths, including missing prepared Parquets, identity map, metadata, and
matching embedding/network artifacts. Missing-input checks exit with status 2. No existing pilot or joint artifacts are
modified, and no real-data training or quality result is claimed.

Standalone smoke YAMLs set `training.max_batches: 2`; full-training YAMLs set
`training.max_batches: null` (all training batches). The loader accepts only
YAML null or a positive integer here; strings, booleans, zero, and negatives fail
configuration validation before training.
