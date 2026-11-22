# Road Graph and GNN YAML Guide

A graph YAML defines **which roads exist, what each road looks like to the model, and how its GNN is built**. It is referenced by both GNN pretraining and joint diffusion experiments through `paths.graph_config`.

Each **road segment becomes a PyTorch Geometric node**. A directed link joins roads that physically connect in the correct direction. The source GeoPackage's `edge_id` identifies roads; the model's road IDs use `edge_id + 1` to reserve zero for padding.

## 1. Start with a graph configuration

```yaml
dataset:
  network_path: fmm_scripts/data/road_network/nyc.gpkg
  layer: roads
  root: outputs/pyg/nyc_gnn

features:
  numeric: [log_length, speed_kph, lanes, width]
  categorical: [highway, surface, access, junction]
  boolean: [oneway, bridge, tunnel]
  geometry: [mid_lon, mid_lat, bearing_sin, bearing_cos, straightness]

graph:
  allow_self_loops: false
  transforms:
    degree_features: {enabled: true, log_scale: true}
    edge_transition_count:
      enabled: true
      match_csv: fmm_scripts/outputs/nyc_15s_enriched_moves/nyc_15s_train_fmm_matches.csv
      smoothing: {method: additive, alpha: 1.0}

model:
  node_features: [x, degree_features]
  edge_features: [edge_transition_prob]
  gnn:
    architecture: gatv2
    hidden_dim: 128
    output_dim: 64
    num_layers: 2
    dropout: 0.1
    activation: relu
    normalization: layer
    residual: true
    gatv2: {heads: 4, concat: true, final_concat: false, attention_dropout: 0.1, add_self_loops: false}
```

This is a **small example**, not a replacement for the existing full graph YAML. `dataset.root` stores `processed/road_graph.pt` and `processed/preprocessor.joblib`. The graph builder uses the GeoPackage; it does **not** perform map matching. `edge_transition_count` requires a **training-only** FMM CSV with `cpath`. If that file is unavailable, disable the transform **and remove `edge_transition_prob` from `model.edge_features`**.

## 2. Raw road features: supported names

Choose lists of features under `features`. The selected raw features are processed together into graph attribute `x`. Numeric/geographic columns are scaled, categorical OSM tags are one-hot encoded, and missing values are handled by the preprocessor.

| Group | Supported feature names | What they describe |
|---|---|---|
| `numeric` | `log_length`, `speed_kph`, `lanes`, `width` | Road distance, speed, lanes, width |
| `categorical` | `highway`, `surface`, `access`, `junction` | Road type, paving, access, junction tag |
| `boolean` | `oneway`, `bridge`, `tunnel` | Road restrictions and structures |
| `geometry` | `start_lon`, `start_lat`, `mid_lon`, `mid_lat`, `end_lon`, `end_lat`, `relative_x_km`, `relative_y_km`, `bearing_sin`, `bearing_cos`, `straightness` | Location, orientation, and shape |

A group can be `[]`, but not **all** groups at once. These are feature **column names**, not values such as `highway: residential`. For a simpler graph, start with a few road features and add more only when they support your experiment.

## 3. Optional graph features

Add features under `graph.transforms`; use `enabled: true/false` (omitted means disabled). Then add their output attribute to `model.node_features` or `model.edge_features` **if the GNN should consume it**.

| Transform | Options (besides `enabled`) | Produces / why it helps |
|---|---|---|
| `degree_features` | `log_scale: true/false` | Node in/out degree; basic connectivity |
| `local_degree_profile` | `log_scale: true/false` | Neighbor degree statistics |
| `random_walk_pe` | `walk_length: 3`, `drop_first: true/false` | Node random-walk position features |
| `laplacian_pe` | `k: 16` | Node Laplacian position features; can be costly |
| `betweenness` | `k: 100`, `seed: 42`, `normalized: true`, `parallel: false` | Approximate node centrality; can be costly |
| `pagerank_features` | `alpha: 0.85`, `tolerance: 1e-8` | Directed importance of a road |
| `clustering_features` | `turbo: false` | Local clustering coefficient |
| `constraint_features` | No other options | Structural-hole constraint |
| `edge_transition_count` | `match_csv`, `smoothing: {method: additive, alpha: 1.0}` | Observed training transitions: `edge_transition_count`, `edge_transition_prob` |
| `adamic_adar_edge_features` | `batch_size: 25000` | Shared-neighborhood link score |
| `random_walk_edge_features` | No other options | Relative random-walk features for connected roads |
| `laplacian_edge_features` | No other options | Relative Laplacian position features for connected roads |

Only **`additive`** smoothing is implemented for transition probabilities. Use only training matches; using test matches would leak information. Random-walk/Laplacian **edge** features require their corresponding node positional transform to run first. Order transforms accordingly.

## 4. Choose GNN inputs and architecture

```yaml
model:
  node_features: [x, degree_features]  # Node attribute groups to concatenate
  edge_features: [edge_transition_prob]  # Edge attribute groups; [] is allowed for GATv2
  gnn:
    architecture: gatv2               # gatv2 | gine
```

`x` contains only the raw groups selected under `features`. Other group names come from enabled transforms. **Selecting an attribute does not compute it**—its transform must be enabled and the cache rebuilt.

| GNN setting | Supported values / purpose |
|---|---|
| `architecture` | **`gatv2` / `gine`**; attention-based versus edge-aware message passing |
| `hidden_dim`, `output_dim` | Positive integers; internal width and exported road embedding width |
| `num_layers` | Positive integer; message-passing depth |
| `dropout` | Number from 0 up to (not including) 1 |
| `activation` | Examples: `relu`, `leaky_relu`, `elu`, `gelu`, `silu`, `tanh` (resolved by installed PyG) |
| `normalization` | **`layer` / `layer_norm` / `batch` / `batch_norm` / `none` / `null`** |
| `residual` | `true/false`; residual connections between layers |

**GATv2-specific** (`model.gnn.gatv2`): `heads` (positive integer), `concat`, `final_concat`, `attention_dropout` (0 to below 1), `add_self_loops` (all three toggles are booleans). If `concat: true` and there are multiple layers, `hidden_dim` must be divisible by `heads`. If `final_concat: true`, `output_dim` must be divisible by `heads`.

**GINE-specific** (`model.gnn.gine`): `train_eps: true/false`, `eps: <finite number>`, and `mlp` settings: `hidden_dim`, `num_layers`, `activation`, `dropout`, `normalization`, `plain_last`. For the MLP, common normalization choices are `layer_norm`, `batch_norm`, or `null` (resolved by PyG). **GINE requires a nonempty `model.edge_features`**, so keep or enable an edge transform when using it.

## 5. Useful combinations

| Goal | Transforms to enable | Model feature selection |
|---|---|---|
| Simple attention GNN (no FMM matches needed) | `degree_features` | `node_features: [x, degree_features]`, `edge_features: []`, `architecture: gatv2` |
| Include observed road usage | `degree_features`, `edge_transition_count` | Add `edge_transition_prob` as an edge feature |
| Add positional information | `degree_features`, `random_walk_pe` | Add `random_walk_pe` to node features; optionally enable its edge counterpart |
| Edge-aware GINE | `degree_features`, `edge_transition_count` | `architecture: gine`, `edge_features: [edge_transition_prob]` |

Start with fewer transforms when iterating. Some graph statistics can require substantial CPU/RAM for a city-scale graph. Every selected group affects the GNN input layout, so **pretraining and joint diffusion must use matching ordered groups and GNN architecture** for encoder transfer.

## 6. Cache and rebuild rules

The experiment pipeline builds the processed graph **if cache files are missing**. It doesn't automatically detect changed YAML settings. Changing road features, topology, source network, or transforms requires rebuilding the cache (or using a fresh `dataset.root`). Keep `road_graph.pt` and `preprocessor.joblib` together.

Changing only `model.node_features` / `model.edge_features` can reuse a cache **if** the requested attributes already exist there. Selecting an uncomputed attribute causes an error.

For the source data paths and example HPC commands, see [Slurm setup](../../hpc/urban_syn/README_SLURM.md). Training and generation settings belong in the [experiment YAML](../experiments/README.md), not this file.
