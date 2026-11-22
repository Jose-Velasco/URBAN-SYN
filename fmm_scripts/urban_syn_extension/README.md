# Urban-Syn extension: GPS sampling and road-sequence preparation

This extension prepares real map-matched trajectory sequences for the **joint GNN + Diff-RNTraj** model. It adds optional 15-second sampling and converts **shared FMM** outputs into road-ID/position-rate sequences. It **does not** build another road network, generate UBODT, or implement a separate FMM runner.

Start with the [shared FMM README](../README.md) for network prerequisites, FMM XML settings, and the meaning of FMM outputs.

## Scripts and configuration

| File | Inputs | Outputs / responsibility |
| --- | --- | --- |
| `config/nyc.yaml` | Model-specific sampling settings | Defines sampling interval, minimum/maximum piece length, grouping key, optional mode filter and source-column mapping. **No road-network configuration.** |
| `sample_trajectories.py` | `enriched_moves.parquet` **or** `traj_cleaned.parquet` | A sampled-observation Parquet containing GPS points at a configured interval and non-overlapping trajectory pieces. |
| `../prepare_fmm_gps.py` | Sampled Parquet + shared `../config/nyc.yaml` | `*_fmm_gps_points.csv` and `*_fmm_trip_id_map.csv`. |
| Shared FMM (`fmm` + extension XML copy) | Prepared GPS CSV, existing shared FMM network and UBODT | One raw FMM CSV including `opath`, `cpath`, `offset`, `length`, and diagnostics. This copy changes only variant-specific input/output paths. |
| `../split_fmm_matches.py` | Raw FMM CSV + trip-ID map + original train/test membership | Two split FMM CSVs, unassigned rows, missing-ID diagnostics and JSON summary. |
| `convert_fmm_sequences.py` | Train/test FMM CSVs | Prepared train/test trajectory Parquet files with `edge_ids` and `rates`, length-distribution `.npy` files, and conversion summary. |

**All commands below run from `fmm_scripts/`,** not from `urban_syn_extension/`. Python steps use the development environment with `uv`. The FMM executable runs in the FMM container. The shared road network and UBODT should already exist.

## Run order: `enriched_moves.parquet` (current NYC workflow)

### 1. Configure sampling

In `urban_syn_extension/config/nyc.yaml`:

```yaml
sampling:
  interval_seconds: 15
  min_points: 20
  max_points: 100
  group_columns: [uid, tid, move_id]

  # Optional: restrict transportation labels; omit to keep all modes.
  # allowed_labels: [0, 1, 2, 3, 4, 5, 6]

  # Optional: internal field name -> source column when different.
  # columns:
  #   datetime: recorded_at
```

One **move** is identified by `(uid, tid, move_id)`. The sampler creates `piece` to distinguish non-overlapping pieces of that move. Trailing pieces shorter than `min_points` are dropped.

**Duplicate timestamps:** The sampler intentionally sorts observations and keeps the **last observation** for a repeated timestamp before `np.interp`. This avoids ambiguous interpolation when two coordinates share one timestamp. It differs from a baseline that interpolates duplicated timestamps directly and can slightly change results for affected moves.

### 2. Sample GPS observations

```bash
uv run python -m urban_syn_extension.sample_trajectories \
  --config ./urban_syn_extension/config/nyc.yaml \
  --trajectory_parquet ../data/nyc_output_tabular/output/enriched_moves.parquet \
  --sampled_parquet ./data/nyc_15s_sampled_trajectories.parquet \
  --log_file ./outputs/logs/sample_trajectories.log
```

**Output:** `data/nyc_15s_sampled_trajectories.parquet`

| Column | Meaning |
| --- | --- |
| `uid`, `tid`, `move_id` | Original user, trip, and move identity. |
| `piece` | Index of a non-overlapping sampled segment within the move. |
| `datetime` | Sampled observation timestamp. |
| `lat`, `lng` | Interpolated latitude and longitude. |
| `label` / `split` | Retained **only if present and consistent** within each source move. Other source metadata are not automatically carried into the sampled file. |

### 3. Prepare the single FMM GPS CSV and ID map

```bash
uv run prepare_fmm_gps.py \
  --config ./config/nyc.yaml \
  --trajectory_parquet ./data/nyc_15s_sampled_trajectories.parquet \
  --gps_output ./data/nyc_15s_fmm_gps_points.csv \
  --trip_id_map_output ./data/nyc_15s_fmm_trip_id_map.csv \
  --identity_columns uid tid move_id piece \
  --log_file ./outputs/logs/prepare_fmm_gps.log
```

| Output | Important fields |
| --- | --- |
| `data/nyc_15s_fmm_gps_points.csv` | `fmm_id;x;y;timestamp` (`;` delimiter); longitude `x`, latitude `y`, integer Unix seconds, sorted by `fmm_id` and time. |
| `data/nyc_15s_fmm_trip_id_map.csv` | `fmm_id,uid,tid,move_id,piece` (`,` delimiter). Each sampled piece receives a positive integer `fmm_id` for traceability. |

`fmm_id` identifies a **GPS trajectory piece**, not a road. The road identifier is `edge_id`, which may validly be `0`.

### 4. Run FMM once against the existing shared road network

For this model-specific 15-second variant, **copy the existing shared XML once** (if not already copied) so the baseline can use its own GPS and output paths without changing the shared default:


Request all required FMM output fields, `offset`, `length`, `opath`, `cpath`. Run **inside the FMM container** from `fmm_scripts/`:

```bash
fmm ./config/fmm_config_csv_point_urban_syn.xml
```

**Output:** `outputs/nyc_15s_fmm_match.csv` (raw, usually semicolon-delimited). Key columns:

- `id`: trajectory identifier returned by FMM; not always preserved for fully unmatched trajectories.
- `opath`: the road `edge_id` associated with each sampled GPS observation.
- `offset` and `length`: matched position along each road and that road's length; needed for position-rate calculation.
- `cpath`: complete path including intermediate roads; used later by the PyG graph builder for training-only transition features.
- `mgeom`, `tpath`, `spdist`, `duration`, `speed`, `error`: additional retained FMM information/diagnostics.

This raw CSV should stay **unchanged**. There is no need to redownload OSM roads or rerun UBODT merely to process this sampled GPS variant. The copied XML is not inherited or automatically kept in sync with the shared XML: if shared FMM fields or matching parameters change intentionally, review the extension copy too. Other baselines may use `enriched_moves.parquet` **without** 15-second resampling; see [the shared FMM README](../README.md#2-prepare-gps-observations-one-input-variant-at-a-time).

### 5. Split the FMM result using original trip identity

Use the existing `data/nyc_test.csv` containing `uid` and `tid` (other columns may be present). Split membership is by **parent `(uid, tid)`**, not by `move_id` or `piece`.

```bash
uv run split_fmm_matches.py \
  --config ./config/nyc.yaml \
  --fmm_matches ./outputs/nyc_15s_fmm_match.csv \
  --trip_id_map ./data/nyc_15s_fmm_trip_id_map.csv \
  --test_trips ./data/nyc_test.csv \
  --output_dir ./outputs/nyc_15s_fmm_enriched_moves \
  --prefix nyc_15s \
  --log_file ./outputs/logs/split_fmm_matches.log
```

**Outputs** in `outputs/nyc_15s_fmm_enriched_moves/`:

| File | Purpose |
| --- | --- |
| `nyc_15s_train_fmm_matches.csv` | FMM matches belonging to training trips; includes every FMM result field. |
| `nyc_15s_test_fmm_matches.csv` | Held-out trip matches, same schema. |
| `nyc_15s_unassigned_fmm_matches.csv` | Unresolvable FMM IDs, including `id=0` failures. |
| `nyc_15s_fmm_ids_without_match.csv` | Prepared input `fmm_id` values with no identifiable output result. |
| `nyc_15s_fmm_split_summary.json` | Counts of input/raw/train/test/unassigned trajectories and matching outcomes. |

The split CSVs are **comma-delimited with quoted path fields**, unlike the raw FMM file. Known-ID empty/partial matches are retained here for diagnostics, then filtered by the sequence converter.

The optional `--split_assignments` CSV can replace `--test_trips` when explicit train/test membership for every trip is available. See the shared README.

### 6. Convert FMM matches to prepared road sequences

```bash
uv run python -m urban_syn_extension.convert_fmm_sequences \
  --train_fmm_matches ./outputs/nyc_15s_enriched_moves/nyc_15s_train_fmm_matches.csv \
  --test_fmm_matches ./outputs/nyc_15s_enriched_moves/nyc_15s_test_fmm_matches.csv \
  --output_dir ./outputs/urban_syn_ready \
  --prefix nyc_15s_enriched_moves \
  --log_file ./outputs/logs/convert_fmm_sequences.log
```

**Outputs** in `outputs/urban_syn_ready/`:

| File | Purpose / fields |
| --- | --- |
| `nyc_15s_enriched_moves_train_prepared_road_sequences.parquet` | Training dataset containing `fmm_id`, `edge_ids`, `rates`, `num_points`. |
| `nyc_15s_enriched_moves_test_prepared_road_sequences.parquet` | Held-out dataset with the same columns. |
| `nyc_15s_enriched_moves_train_length_distribution.npy` | Count of accepted **training** trajectories at each length; index = number of observations. Used to choose generation lengths. |
| `nyc_15s_enriched_moves_test_length_distribution.npy` | Equivalent test distribution, useful for analysis; do not use test lengths to drive training-time generation settings. |
| `nyc_15s_enriched_moves_sequence_preparation_summary.json` | Per-split accepted observations/trajectories and skipped-row reasons (for example, empty `opath`). |

**Prepared Parquet schema:**

| Column | Meaning |
| --- | --- |
| `fmm_id` | Traceability back to FMM result and trip-ID map. |
| `edge_ids` | Road IDs from `opath`, one per matched observation. These are **shared road-network `edge_id` values**, not PyG node indices or model IDs; `edge_id=0` remains valid. |
| `rates` | `[0, 1]` position rates aligned with `edge_ids`. On a new road, rate is offset/length; for consecutive observations on the same road, rate represents the fraction of the **remaining road** traversed. |
| `num_points` | Number of road observations in the sequence, equal to the lengths of `edge_ids` and `rates`. |

The converter skips unusable rows (empty `opath`, mismatched `opath`/`offset`/`length` counts, invalid IDs). The raw and split FMM CSVs are retained. The length-distribution files belong to preprocessing.

## Expected directory layout (current NYC run)

```text
fmm_scripts/
├── urban_syn_extension/config/fmm_config_csv_point.xml  # FMM XML copy for 15s variant
├── data/
│   ├── road_network/nyc.gpkg                  # One pre-existing shared road network
│   ├── fmm/fmm_nyc.shp                       # Same road IDs; keep sidecars
│   ├── nyc_ubodt.bin                         # Pre-existing shortest-path table
│   ├── nyc_test.csv                          # Held-out uid/tid list
│   ├── nyc_15s_sampled_trajectories.parquet
│   ├── nyc_15s_fmm_gps_points.csv
│   └── nyc_15s_fmm_trip_id_map.csv
└── outputs/
    ├── nyc_15s_fmm_match.csv                 # Raw FMM results
    ├── nyc_15s_test_fmm_matches/
    │   ├── nyc_15s_train_fmm_matches.csv
    │   ├── nyc_15s_test_fmm_matches.csv
    │   ├── nyc_15s_unassigned_fmm_matches.csv
    │   ├── nyc_15s_fmm_ids_without_match.csv
    │   └── nyc_15s_fmm_split_summary.json
    └── urban_syn_ready/
        ├── nyc_15s_enriched_moves_train_prepared_road_sequences.parquet
        ├── nyc_15s_enriched_moves_test_prepared_road_sequences.parquet
        ├── nyc_15s_enriched_moves_train_length_distribution.npy
        ├── nyc_15s_enriched_moves_test_length_distribution.npy
        └── nyc_15s_enriched_moves_sequence_preparation_summary.json
```

## Switching to `traj_cleaned.parquet`

The same **15-second extension workflow** works with the **cleaned trajectory GPS Parquet**; no new road network or FMM implementation is needed. For any baseline that wants to map-match original GPS observations **without** 15-second sampling, skip this extension's sampler and use the direct `prepare_fmm_gps.py` examples in the [shared FMM README](../README.md#2-prepare-gps-observations-one-input-variant-at-a-time).

| Setting / argument | `enriched_moves.parquet` | `traj_cleaned.parquet` |
| --- | --- | --- |
| Sampling `group_columns` | `[uid, tid, move_id]` | `[uid, tid]` |
| Sampling `--trajectory_parquet` | `../data/nyc_output_tabular/output/enriched_moves.parquet` | `../data/nyc_output_tabular/output/traj_cleaned.parquet` |
| FMM GPS `--identity_columns` | `uid tid move_id piece` | `uid tid piece` |
| Train/test identity | `uid, tid` | `uid, tid` |
| Road-network/UBODT artifacts | Same shared artifacts | **Same shared artifacts** |

For example, change `group_columns` in `urban_syn_extension/config/nyc.yaml` to `[uid, tid]`, then run:

```bash
uv run python -m urban_syn_extension.sample_trajectories \
  --config ./urban_syn_extension/config/nyc.yaml \
  --trajectory_parquet ../data/nyc_output_tabular/output/traj_cleaned.parquet \
  --sampled_parquet ./data/nyc_15s_cleaned_sampled_trajectories.parquet

uv run prepare_fmm_gps.py \
  --config ./config/nyc.yaml \
  --trajectory_parquet ./data/nyc_15s_cleaned_sampled_trajectories.parquet \
  --gps_output ./data/nyc_15s_cleaned_fmm_gps_points.csv \
  --trip_id_map_output ./data/nyc_15s_cleaned_fmm_trip_id_map.csv \
  --identity_columns uid tid piece
```

Then update the **extension's copied FMM XML** GPS input/output paths, keeping the road-network and UBODT paths unchanged. For example:

- Raw match output: `outputs/nyc_15s_cleaned_fmm_match.csv`
- Split destination: `outputs/nyc_15s_cleaned/`, `--prefix nyc_15s_cleaned`
- Converter destination: `outputs/urban_syn_ready/`, `--prefix nyc_15s_cleaned`
- Split input map: `data/nyc_15s_cleaned_fmm_trip_id_map.csv`; same `data/nyc_test.csv` **only if it uses the same `(uid, tid)` identities**.

The same final names are then formed with the `nyc_15s_cleaned_...` prefix. If you instead want to match the original cleaned observations **without** 15-second resampling, skip `sample_trajectories.py` and call `prepare_fmm_gps.py` directly with the original Parquet and default identity `[uid, tid]`. That is a different GPS input variant and should produce separately named FMM results.

**Do not overwrite the enriched-moves artifacts**: the two GPS inputs can yield different matches (`opath`, `cpath`, `offset`, and `length`) even when using the same roads.

## After preprocessing: PyG and training

```text
Train FMM matches (cpath) + shared nyc.gpkg
    -> graph builder -> road_graph.pt

Prepared training Parquet (edge_ids + rates) + road_graph.pt
    -> RoadSequenceDataset -> length-based DataLoader
    -> model road IDs (PyG node index + 1; 0 reserved for padding)
    -> jointly trained GNN + Diff-RNTraj
```

Point the graph builder's `edge_transition_count.match_csv` setting at the **training** FMM CSV, not the combined raw results. The graph builder and DataLoader share road identities through `edge_id`. Neither FMM preprocessing nor the sequence converter needs to know PyG node ordering.

`eid_seqs.bin`, `rate_seqs.bin`, and `rn_dict.json` are **not required** by our current integrated training pipeline. Those files belong to the older baseline's storage conventions. The length-distribution `.npy` files are lightweight preprocessing outputs and can be used for generation without loading the training DataLoader.

### Known compatibility boundaries

`diffrntraj_version/` may share helper imports and road-network YAML values, but its generated artifacts are not automatically compatible with the new full-road PyG graph. See the compatibility notes in the [shared README](../README.md).

The November 7, 2026 NYC `enriched_moves.parquet` preparation run produced **19,974 sampled pieces**, **15,793 training sequences**, and **3,668 test sequences**. These are illustrative output counts from that particular FMM configuration, not guaranteed totals for a new network or changed FMM matching parameters.
