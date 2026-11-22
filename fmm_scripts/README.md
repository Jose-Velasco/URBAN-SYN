# Shared FMM preprocessing

This directory prepares GPS trajectories for **Fast Map Matching (FMM)** using a single **shared road network**. It supports multiple source trajectory datasets and downstream baselines without downloading a different network for each run.

The main stages are separate: **build the road network (only when needed) -> prepare GPS inputs -> run FMM once per GPS variant -> split the FMM output by original trip**

For the 15-second, move-based Diff-RNTraj extension, see [`urban_syn_extension/README.md`](urban_syn_extension/README.md). The existing [`diffrntraj_version/README.md`](diffrntraj_version/README.md) documents a separate, original baseline pipeline.

## Scripts and responsibilities

| Script / configuration | Reads | Writes / does |
| --- | --- | --- |
| `build_road_network.py` | `config/nyc.yaml`; optionally a trajectory Parquet for network bounds | `data/road_network/nyc.gpkg` and `data/fmm/fmm_nyc.shp` (plus Shapefile sidecars). **Build only when deliberately creating/replacing the network.** |
| `prepare_fmm_gps.py` | Any compatible GPS-observation Parquet; `config/nyc.yaml` trajectory column mappings | A semicolon-delimited FMM GPS CSV and a comma-delimited `*_fmm_trip_id_map.csv`. |
| `config/ubodt_config.xml` + `ubodt_gen` | Existing FMM road Shapefile | `data/nyc_ubodt.bin`, a shortest-path lookup for FMM. Regenerate only if the network or required UBODT settings change. |
| `config/fmm_config_csv_point.xml` + `fmm` | Existing road Shapefile, UBODT, and prepared GPS CSV | One **raw** FMM match CSV for the selected GPS variant. A baseline can use its own copied XML while reusing the same network and UBODT. |
| `split_fmm_matches.py` | Raw FMM CSV, its trip-ID map, and either test-trip IDs or explicit split assignments | Train/test match CSVs and failure/coverage diagnostics; leaves raw output intact. |
| `prepare_model_fmm_points.py` | Model-generated GPS observations | Separate FMM-ready inputs for evaluating generated trajectories; not part of the real-trajectory preparation steps below. |
| `utils.py` | Called by the scripts above | Shared network/GPS utilities, configuration, logging, and stable `fmm_id` generation. |
| `prepare_fmm_inputs.py` |  | kept for compatibility. Downloads and builds the road network and prepares gps points to be FMM. Should instead use `build_road_network.py` then `prepare_fmm_gps.py` but the downloaded road network will likely not be compatible with already download ones. |

### Suggested directory layout

Paths shown below are **examples from the 15s Diff-RNTraj our version NYC run**.

```text
fmm_scripts/
├── README.md
├── config/
│   ├── nyc.yaml
│   ├── ubodt_config.xml
│   └── fmm_config_csv_point.xml
├── data/
│   ├── road_network/nyc.gpkg
│   ├── fmm/fmm_nyc.shp            # Also .dbf, .shx, .prj, etc.
│   ├── nyc_ubodt.bin
│   ├── nyc_15s_sampled_trajectories.parquet
│   ├── nyc_15s_fmm_gps_points.csv
│   └── nyc_15s_fmm_trip_id_map.csv
└── outputs/
    ├── nyc_15s_fmm_match.csv     # Complete raw FMM output
    ├── nyc_15s_fmm_enriched_moves/   # Post-FMM train/test + diagnostics
    └── urban_syn_ready/         # Model-specific prepared sequences
```

## Run the shared FMM stages

Commands below assume your working directory is **`fmm_scripts/`**. Run Python steps in the project development environment with `uv`; run `ubodt_gen` and `fmm` in the FMM Docker container.

### 0. Build a network only if one does not already exist

**Do not repeat this step when changing input trajectory datasets.** A new OSMnx download or changed road filter can change `edge_id` assignments, making existing FMM results, UBODT files, and PyG graphs incompatible.

```bash
uv run build_road_network.py \
  --config ./config/nyc.yaml \
  --shared_road_network_output ./data/road_network/nyc.gpkg \
  --shared_road_network_layer roads \
  --fmm_road_network_output ./data/fmm/fmm_nyc.shp \
  --trajectory_parquet ../data/nyc_output_tabular/output/traj_cleaned.parquet \
  --log_file ./outputs/logs/build_road_network.log
```

`--trajectory_parquet` is required here **only** when `road_network.coverage.type: trajectory_bbox` is selected in the YAML. It determines the bounding box; it does not mean this dataset must also be used for map matching. The builder refuses to replace existing network files unless `--overwrite` is explicitly supplied.

- **GeoPackage (`nyc.gpkg`, layer `roads`)**: full road geometry and attributes including `edge_id`, `u`, `v`, OSM `key`, `length`, and available OSM tags. Used by the PyG graph builder, later geometry-based GPS reconstruction and any baseline needing road network metadata.
- **FMM Shapefile (`fmm_nyc.shp` + sidecars)**: the same road segments, with minimal `edge_id`, `u`, `v`, and `geometry` columns. Used by FMM and `ubodt_gen`. Keep all associated Shapefile files together.
- **Important:** `edge_id=0` is a **valid road segment**. It is unrelated to FMM's failed trajectory `id=0` behavior.

The OSM mode/filter is whatever is **currently enabled** in `config/nyc.yaml`.

### 1. Generate UBODT if needed

In the FMM container, with the existing network referenced by `config/ubodt_config.xml`:

```bash
ubodt_gen ./config/ubodt_config.xml
```

`data/nyc_ubodt.bin` caches upper-bounded shortest paths for the **same** FMM road network. The `delta` setting controls maximum precomputed path distance (in map units). Reusing the existing network normally means reusing this UBODT. Keep the network and UBODT file paths, ID fields, and relevant settings consistent with the matching XML.

### 2. Prepare GPS observations (one input variant at a time)

Example using **`traj_cleaned.parquet` without sampling**:

```bash
uv run prepare_fmm_gps.py \
  --config ./config/nyc.yaml \
  --trajectory_parquet ../data/nyc_output_tabular/output/traj_cleaned.parquet \
  --gps_output ./data/nyc_original_fmm_gps_points.csv \
  --trip_id_map_output ./data/nyc_original_fmm_trip_id_map.csv
```

Example using **`enriched_moves.parquet` directly, without 15-second sampling**:

```bash
uv run prepare_fmm_gps.py \
  --config ./config/nyc.yaml \
  --trajectory_parquet ../data/nyc_output_tabular/output/enriched_moves.parquet \
  --gps_output ./data/nyc_enriched_moves_fmm_gps_points.csv \
  --trip_id_map_output ./data/nyc_enriched_moves_fmm_trip_id_map.csv \
  --identity_columns uid tid move_id
```

Here `(uid, tid, move_id)` identifies **one original move** for FMM. This keeps separate moves from the same trip from being concatenated into a single map-matching sequence, which could otherwise create artificial road connections across gaps. The input GPS timestamps and coordinates are **not resampled**. The output trip-ID map contains `fmm_id,uid,tid,move_id` and the GPS CSV contains `fmm_id;x;y;timestamp`.

> **Model-specific exception:** To run the joint GNN + Diff-RNTraj **15-second sampled** workflow, first run `urban_syn_extension/sample_trajectories.py`, then use `--identity_columns uid tid move_id piece`. See [urban_syn_extension/README.md](urban_syn_extension/README.md) for the full order, its sampling YAML. The sampling procedure is **not required** by the shared FMM pipeline or other baselines.

Other source datasets can be prepared in the same way by changing the input path, output paths, and, when needed, the source-column mapping and trajectory identity columns. The existing `traj_cleaned.parquet` example above uses `(uid, tid)` without sampling.

The shared `config/nyc.yaml` has a `trajectories` section:

```yaml
trajectories:
  identity_columns: [uid, tid]  # Default original trip identity; CLI may override.
  # split_identity_columns: [uid, tid]  # Original trip key for post-FMM splitting.
  columns:                       # Shared field name: source Parquet column name.
    uid: uid
    tid: tid
    datetime: datetime
    lat: lat
    lng: lng
```

Change the right-hand names when source datasets use different column names. This mapping preserves additional existing fields, such as `move_id` and `piece`.

**Prepared GPS CSV** (`*_fmm_gps_points.csv`; `;` delimiter):

| Column | Meaning |
| --- | --- |
| `fmm_id` | Positive integer identifying one original trajectory or sampled piece. IDs are deterministic for a given set of identities, but can change when that set changes. |
| `x`, `y` | Longitude and latitude, respectively, in EPSG:4326. |
| `timestamp` | Integer Unix timestamp in seconds. |

Rows are ordered by `fmm_id`, then `timestamp`.

**Trip-ID map** (`*_fmm_trip_id_map.csv`; `,` delimiter): `fmm_id` plus the configured identity columns, for example `uid`, `tid`, `move_id`, `piece`. It lets downstream processing resolve an FMM result back to the original trip. Preserve the map alongside the matching result; do not regenerate it independently with a different GPS population/order.

### 3. Run FMM once for the prepared GPS variant

Select the FMM XML configuration for this input variant. The shared default is `config/fmm_config_csv_point.xml`; each baseline may instead keep a **copy** of it (for example `config/fmm_config_csv_point_urban_syn.xml`) so input/output paths can differ without editing the shared default. These XML are not automatically synchronized so make sure to update all if critical parameters such as delta radius, k, r or , etc are changed to update all of them for each models config. also make sure the ubodt_config.xml also are in sync with fmm_config_csv_point files.

In the XML you choose, set:

- `<input><gps><file>`: the chosen `*_fmm_gps_points.csv`.
- `<output><file>`: a unique filename for this variant, such as `./outputs/nyc_15s_fmm_match.csv`.
- `<output><fields>`: retain all **ten** requested fields, including `offset`.

Then, from the FMM container in `fmm_scripts/`, run the XML you selected:

```bash
fmm ./config/fmm_config_csv_point.xml
```

For the 15-second extension, use `fmm ./config/fmm_config_csv_point_urban_syn.xml`. Both must reference the *same* existing road network and UBODT.

Do not overwrite the combined **raw FMM result**: it is the source for train/test splitting, diagnostics, and future preprocessing changes.

**Raw FMM result fields** (normally `;` delimiter):

| Column | Meaning |
| --- | --- |
| `id` | FMM's output trajectory ID. FMM may emit `0` for completely candidate-unmatched trajectories even when input `fmm_id` was positive. |

The full raw output fields requested in the xml **plus** the trajectory identifier. A raw FMM CSV normally uses `id`, even when the GPS input column is `fmm_id`.

### 4. Split the one FMM result into train/test

`split_fmm_matches.py` works the same for **unsampled `enriched_moves.parquet`**, unsampled `traj_cleaned.parquet`, 15-second sampled variants, and other compatible GPS datasets. It does **not** need a sampling flag or know how the GPS coordinates were produced. It only needs the **matching** raw FMM output and trip-ID map from the **same GPS input run**, plus valid train/test identities.

By default, split membership uses the original `(uid, tid)` even when the FMM trajectory identity also includes `move_id` or `piece`. Thus, all moves/pieces from a parent trip stay in the same partition. The `nyc_test.csv` must use those same original `(uid, tid)` values. For another dataset whose split identity differs, configure `trajectories.split_identity_columns` and supply a matching test list or split-assignments file.

With an existing `nyc_test.csv` containing `uid` and `tid`:

```bash
uv run split_fmm_matches.py \
  --config ./config/nyc.yaml \
  --fmm_matches ./outputs/nyc_15s_fmm_match.csv \
  --trip_id_map ./data/nyc_15s_fmm_trip_id_map.csv \
  --test_trips ./data/nyc_test.csv \
  --output_dir ./outputs/nyc_15s_enriched_moves \
  --prefix nyc_15s
```

For **unsampled enriched moves**, run the same splitter with `--fmm_matches ./outputs/nyc_enriched_moves_fmm_match.csv`, `--trip_id_map ./data/nyc_enriched_moves_fmm_trip_id_map.csv`, and a separate `--output_dir` / `--prefix` such as `./outputs/nyc_enriched_moves` / `nyc_enriched_moves`. For **unsampled cleaned trajectories**, use the matching `nyc_original_*` files instead. Keep each GPS variant's outputs distinct.

Alternative: use **`--split_assignments ./data/trajectory_split_assignments.csv`** instead of `--test_trips`. That CSV must provide the configured original identity columns plus a `split` column (`train` or `test`) for **every prepared trip**. With `--test_trips`, remaining known trips are assigned to training.

| Output from the example | Contents |
| --- | --- |
| `outputs/nyc_15s_enriched_moves/nyc_15s_train_fmm_matches.csv` | All raw FMM fields, rows belonging to training trips. |
| `outputs/nyc_15s_enriched_moves/nyc_15s_test_fmm_matches.csv` | Same schema for held-out test trips. |
| `outputs/nyc_15s_enriched_moves/nyc_15s_unassigned_fmm_matches.csv` | Rows with unresolvable IDs, including FMM output `id=0`. |
| `outputs/nyc_15s_enriched_moves/nyc_15s_fmm_ids_without_match.csv` | Input `fmm_id` values without an identifiable FMM output row. Includes original identity columns and split. |
| `outputs/nyc_15s_enriched_moves/nyc_15s_fmm_split_summary.json` | Counts for raw/train/test/unassigned rows and empty/single/multiple `opath` outcomes. |

The splitter **preserves every match column**, including empty/partial matches, and writes **comma-delimited CSV with standard quoting** (path fields containing commas remain safe). It does not modify the raw `;`-delimited file. Downstream graph features can read `train_fmm_matches.csv` with ordinary `pd.read_csv(...)`.

**FMM `id=0` caveat:** An output `id=0` can represent an unmatched trajectory for which FMM lost the original input ID. The splitter cannot reliably determine which trip produced each such row. Those rows are unassigned, not included in train/test. This is separate from valid road `edge_id=0`.

## Downstream usage

- **Graph building:** Configure `edge_transition_count.match_csv` in the graph YAML to the **training** FMM match CSV. The graph builder reads `cpath` and the existing shared GeoPackage.
- **Diff-RNTraj extension:** `urban_syn_extension/convert_fmm_sequences.py` reads train/test FMM matches and writes prepared Parquet road sequences and length distributions. Its output retains shared road-network `edge_id` values. See the [extension README](urban_syn_extension/README.md).
- **Other baselines:** May consume the shared road artifacts and FMM matches but keep their own downstream converters.

## FMM container

From `fmm_scripts/`, the existing image can be built and opened with a bind mount:

```bash
docker build -t urban-syn-fmm:ubuntu22 .
```

Linux/macOS:

```bash
docker run -it --rm --mount type=bind,source="${PWD}",target=/workspace urban-syn-fmm:ubuntu22
```

Windows Git Bash:

```bash
docker run -it --rm --mount type=bind,source="$(pwd -W)",target=/workspace urban-syn-fmm:ubuntu22
```

Run FMM commands from `/workspace` inside the container so the relative paths in the XML resolve correctly. The container must have access to `data/` and `outputs/` through this mount.

## Compatibility notes (older baseline)

- The FMM shared pipeline now uses `fmm_id` in FMM input than the older `id` convention.
- The shared road network must be reused for new comparable experiments. Changing `config/nyc.yaml` and redownloading OSM data can change road coverage and `edge_id` values.
