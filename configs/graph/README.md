# Road graph and GNN YAML configuration guide

This guide describes `configs/graph/nyc_graph.yaml`, which controls **road graph preparation, feature selection, and the GNN architecture** for URBAN-SYN. It complements the [experiment YAML guide](../experiments/README.md), which controls diffusion, training, checkpointing, generation, and MLflow.

A physical road segment becomes one **PyG node**. Directed links represent physically possible road-to-road transitions (road A's end junction equals road B's start junction). The graph uses the shared road network's `edge_id` field to preserve road identities; PyG node indices are not interchangeable with road IDs. A road `edge_id` of `0` is valid.

## 1. How the file is used

An experiment references this graph YAML:

```yaml
# configs/experiments/nyc_joint.yaml
paths:
  graph_config: configs/graph/nyc_graph.yaml
```

The experiment runner's `prepare-graph` stage generates the processed graph and fitted sklearn preprocessor **only when the cache files are missing**:

```text
outputs/pyg/nyc_gnn/processed/
├── road_graph.pt
└── preprocessor.joblib
```

Both are outputs, not prerequisites for an initial build. If a transform needs observed road-transition frequencies, it also requires a **training-only** FMM `cpath` CSV. The source road GeoPackage must already exist; this stage does not download roads or run FMM.

**Important cache rule:** The current stage checks for cache **existence**, not whether the YAML still matches the cache. If you change selected raw features, transforms, transition CSV, or topology, deliberately rebuild the cached graph, or use a separate `dataset.root` for that graph variant. Deleting just one file triggers rebuilding, but keeping separate cache roots is cleaner when comparing graph configurations. Do not mix a stale `preprocessor.joblib` with a new graph.

## 2. Example graph YAML

This is a practical example showing **every supported graph/GNN attribute**. Disabled transforms do not execute. The `model.gnn.gine` options can remain in the file when `architecture: gatv2` and vice versa.

```yaml
dataset:
  root: outputs/pyg/nyc_gnn          # Directory for PyG graph cache
  network_path: fmm_scripts/data/road_network/nyc.gpkg
  layer: roads                       # GeoPackage road layer

features:
  numeric: [log_length, speed_kph, lanes, width]
  categorical: [highway, surface, access, junction]
  boolean: [oneway, bridge, tunnel]
  geometry: [mid_lon, mid_lat, bearing_sin, bearing_cos, straightness]

graph:
  allow_self_loops: false           # Road-to-itself physical transitions
  transforms:
    degree_features: {enabled: true, log_scale: true}
    local_degree_profile: {enabled: false, log_scale: true}
    random_walk_pe: {enabled: false, walk_length: 3, drop_first: true}
    laplacian_pe: {enabled: false, k: 16}
    betweenness: {enabled: false, k: 100, seed: 42, normalized: true, parallel: false}
    pagerank_features: {enabled: false, alpha: 0.85, tolerance: 1.0e-8}
    clustering_features: {enabled: false, turbo: false}
    constraint_features: {enabled: false}
    edge_transition_count:
      enabled: true
      match_csv: fmm_scripts/outputs/nyc_15s_enriched_moves/nyc_15s_train_fmm_matches.csv
      smoothing:
        method: additive             # Only supported smoothing method
        alpha: 1.0
    adamic_adar_edge_features: {enabled: false, batch_size: 25000}
    random_walk_edge_features: {enabled: false}
    laplacian_edge_features: {enabled: false}

model:
  node_features: [x, degree_features] # Output attributes to concatenate as node input
  edge_features: [edge_transition_prob]  # Output attributes to concatenate as edge input
  gnn:
    architecture: gatv2             # gatv2 | gine
    hidden_dim: 128
    output_dim: 64
    num_layers: 2
    dropout: 0.1
    activation: relu                # e.g. relu | leaky_relu | elu | gelu | silu | tanh
    normalization: layer            # layer | layer_norm | batch | batch_norm | none | null
    residual: true
    gatv2:
      heads: 4
      concat: true
      final_concat: false
      attention_dropout: 0.1
      add_self_loops: false
    gine:
      train_eps: true
      eps: 0.0
      mlp:
        hidden_dim: 128
        num_layers: 2
        activation: relu
        dropout: 0.1
        normalization: layer_norm
        plain_last: true
```

This example includes a training-transition feature and selects it as an edge attribute. If you do not have the matching **training** FMM CSV, disable `edge_transition_count` and remove `edge_transition_prob` from `model.edge_features`. With `architecture: gine`, **some edge feature group is required**, because GINE needs `edge_attr`.

## 3. Graph source and raw road features

### Dataset and graph structure

| Attribute | Supported values / default | Meaning |
|---|---|---|
| `dataset.root` | Directory path; **required** | Where processed graph and fitted preprocessor are cached. |
| `dataset.network_path` | File path; **required** | Existing road GeoPackage with road geometry and fields such as `edge_id`, `u`, `v`, `length`, and `speed_kph`. |
| `dataset.layer` | Nonempty GeoPackage layer name; default `roads` | Which layer to read from the GeoPackage. |
| `graph.allow_self_loops` | `true` \| `false`; default `false` | Whether topology construction can include transitions from a road to itself. |
| `graph.transforms` | Mapping of registered transform name to `{enabled: ..., options...}`; default empty | Extra structural node/edge features computed once when processing the graph. |

Paths inside the graph YAML are resolved against the repository root when the experiment runner loads them. The underlying standalone graph builder may follow the current working directory instead, so run it from the repository root for consistent paths.

### Selected raw road features

Feature groups contain **lists of field names**, not booleans. The preprocessor handles missing values; numeric/geometric features are scaled, categorical features are one-hot encoded. At least one feature across all four groups must be selected, and names must not repeat.

| Attribute | Currently supported names | Meaning |
|---|---|---|
| `features.numeric` | `log_length` \| `speed_kph` \| `lanes` \| `width` | Road length, speed limit, lane count, and width. `log_length` is computed from road length. |
| `features.categorical` | `highway` \| `surface` \| `access` \| `junction` | OSM categorical road tags. |
| `features.boolean` | `oneway` \| `bridge` \| `tunnel` | Binary OSM road properties. |
| `features.geometry` | `mid_lon` \| `mid_lat` \| `relative_x_km` \| `relative_y_km` \| `bearing_sin` \| `bearing_cos` \| `straightness` | Midpoint, relative position, orientation, and shape of a segment. |

All four groups can be omitted or use `[]` individually, but **not all at once**. The resulting encoded node feature matrix is stored in `x`.

## 4. Optional graph transforms

All registered transforms use `enabled: true | false`. Omitted transforms or transforms set to `false` do not run. When a transform is enabled, the listed options are accepted, and any other options cause an error.

| Transform under `graph.transforms` | Available options and defaults | Produced feature / purpose |
|---|---|---|
| `degree_features` | `log_scale: true` | `degree_features`, road-node degree statistics. |
| `local_degree_profile` | `log_scale: true` | `local_degree_profile`, local neighborhood degree statistics. |
| `random_walk_pe` | `walk_length: 3`, `drop_first: true` | `random_walk_pe`, random-walk positional encoding. |
| `laplacian_pe` | `k: 16` | `laplacian_pe`, Laplacian eigenvector positional encoding. |
| `betweenness` | `k: 100`, `seed: 42`, `normalized: true`, `parallel: false` | `betweenness`, approximate directed centrality; `k` controls samples. |
| `pagerank_features` | `alpha: 0.85`, `tolerance: 1.0e-8` | `pagerank_features`, directed PageRank centrality. |
| `clustering_features` | `turbo: false` | `clustering_features`, local clustering coefficient. |
| `constraint_features` | No additional options | `constraint_features`, structural-hole constraint. |
| `edge_transition_count` | `match_csv` (required when enabled), `smoothing` | `edge_transition_count` and `edge_transition_prob`, observed training-path frequencies and smoothed probabilities on **existing** physical transitions. |
| `adamic_adar_edge_features` | `batch_size: 25000` | `adamic_adar_edge_features`, edge similarity score. |
| `random_walk_edge_features` | No additional options | `random_walk_edge_features`, relative random-walk positional information on edges. Requires the corresponding node positional encoding. |
| `laplacian_edge_features` | No additional options | `laplacian_edge_features`, relative Laplacian positional information on edges. Requires the corresponding node positional encoding. |

### Transform option reference

All options below live under `graph.transforms.<transform_name>`. The `enabled` switch applies to **every** transform and accepts `true` or `false`.

| Transform option | Supported values / default | Meaning |
|---|---|---|
| `degree_features.log_scale` | Boolean; default `true` | Apply `log1p` to directed degree counts. |
| `local_degree_profile.log_scale` | Boolean; default `true` | Apply `log1p` to local neighborhood degree statistics. |
| `random_walk_pe.walk_length` | Integer `>= 2` with `drop_first: true`, or `>= 1` otherwise; default `3` | Maximum random-walk length for return-probability features. |
| `random_walk_pe.drop_first` | Boolean; default `true` | Drop the one-step return-probability column. |
| `laplacian_pe.k` | Positive integer; default `16` | Number of nontrivial Laplacian eigenvectors to compute. |
| `betweenness.k` | Positive integer; default `100` | Number of samples for approximate betweenness centrality. |
| `betweenness.seed` | Integer; default `42` | Seed for NetworKit's sampling algorithm. |
| `betweenness.normalized` | Boolean; default `true` | Normalize centrality scores. |
| `betweenness.parallel` | Boolean; default `false` | Enable NetworKit parallel sampling, which can use more memory and change reproducibility. |
| `pagerank_features.alpha` | Float strictly between `0` and `1`; default `0.85` | PageRank damping probability. |
| `pagerank_features.tolerance` | Positive float; default `1.0e-8` | PageRank convergence tolerance. |
| `clustering_features.turbo` | Boolean; default `false` | Use a faster triangle-counting mode with higher memory use. |
| `adamic_adar_edge_features.batch_size` | Positive integer; default `25000` | Number of existing edge pairs scored per batch. |
| `edge_transition_count.match_csv` | Path; required when enabled | Training-only FMM split CSV, containing the `cpath` column. |
| `edge_transition_count.smoothing.method` | **`additive` only**; default `additive` | Select add-alpha smoothing for outgoing transition probabilities. |
| `edge_transition_count.smoothing.alpha` | Positive finite float; default `1.0` | Pseudocount for every physically available outgoing road transition. |

`constraint_features`, `random_walk_edge_features`, and `laplacian_edge_features` have no options beyond `enabled`.

Only `edge_transition_count` takes the nested smoothing mapping:

| Attribute | Supported values / default | Meaning |
|---|---|---|
| `graph.transforms.edge_transition_count.enabled` | `true` \| `false`; default `false` when omitted | Enable observed-path transition features. |
| `graph.transforms.edge_transition_count.match_csv` | Path to **training** FMM matches | Comma-separated split CSV with `cpath` column. Never point this at test matches or unsplit FMM data. |
| `graph.transforms.edge_transition_count.smoothing.method` | **`additive` only**; default `additive` | Smoothing formula for physical-road transition probabilities. |
| `graph.transforms.edge_transition_count.smoothing.alpha` | Positive finite number; default `1.0` | Additive smoothing strength for physically available outgoing transitions. |

Transform order follows the YAML mapping order. When an edge-derived feature depends on a node transform, list the node transform **first**. Feature computations such as betweenness, Laplacian decomposition, or training transition counts can be expensive on a large road graph. They are preprocessing operations, not per-epoch GNN calculations.

## 5. Selecting what the GNN actually receives

The graph cache can contain more feature groups than the model uses. The following keys control which groups are concatenated and passed to the GNN:

| Attribute | Supported values / default | Meaning |
|---|---|---|
| `model.node_features` | Nonempty list of unique graph node attributes; default `[x]` | Node feature groups concatenated to form GNN input `x`. Examples: `[x]`, `[x, degree_features, pagerank_features]`. |
| `model.edge_features` | List of unique graph edge attributes; default `[]` | Edge feature groups concatenated to form GNN input `edge_attr`. Examples: `[edge_transition_prob]`, `[edge_transition_count, edge_transition_prob]`. |
| `model.gnn` | GNN architecture mapping, **required by experiment loader** | Configures the trainable road encoder. |

Valid choices for `model.node_features` or `model.edge_features` depend on which transforms actually ran. A feature name is not valid merely because it appears in the registry; it must exist on the processed graph with the expected tensor shape. Graph selection raises an error otherwise. Selecting feature groups does not automatically enable their transforms.

## 6. GNN architecture fields

### Shared GNN settings

| Attribute | Supported values / default | Meaning |
|---|---|---|
| `model.gnn.architecture` | **`gatv2` \| `gine`**; default `gatv2` | Attention-based GATv2 or edge-aware GINE message-passing layers. |
| `model.gnn.hidden_dim` | Positive integer; default `128` | Width of intermediate GNN layers. |
| `model.gnn.output_dim` | Positive integer; default `64` | Width of learned road embeddings passed to diffusion. |
| `model.gnn.num_layers` | Positive integer; default `2` | Number of GNN convolution layers. |
| `model.gnn.dropout` | Float `0 <= p < 1`; default `0.1` | Dropout on intermediate GNN activations during training. |
| `model.gnn.activation` | PyG-resolved activation string; default `relu` | Intermediate-layer nonlinearity. Common accepted examples: `relu` \| `leaky_relu` \| `elu` \| `gelu` \| `silu` \| `tanh` \| `sigmoid` \| `softplus`. |
| `model.gnn.normalization` | **`layer` \| `layer_norm` \| `batch` \| `batch_norm` \| `none` \| `null`**; default `layer` | Normalization between GNN layers, not after the final output layer. |
| `model.gnn.residual` | Boolean; default `true` | Add skip connections, with a learned projection if dimensions differ. |

**Activation note:** The project does not define a closed activation enum. `model.gnn.activation` is passed to PyTorch Geometric's `activation_resolver`, so the precise complete list depends on the installed PyG version. The names above are common examples, **not** a guarantee that arbitrary activation strings are accepted. An unsupported name raises an error at model creation. The same principle applies to GINE's internal MLP activation.

`model.gnn.normalization` is different: it is a **fixed list in our own code**, and `none`/`null` mean no intermediate normalization.

### GATv2-only settings

These keys live under `model.gnn.gatv2`. They are ignored by the GINE convolution factory, but validated as a settings object.

| Attribute | Supported values / default | Meaning |
|---|---|---|
| `heads` | Positive integer; default `4` | Number of attention heads. |
| `concat` | Boolean; default `true` | Concatenate attention-head outputs in non-final layers; `false` averages heads. |
| `final_concat` | Boolean; default `false` | Concatenate heads in final layer; `false` averages them. |
| `attention_dropout` | Float `0 <= p < 1`; default `0.1` | Dropout applied to attention coefficients. |
| `add_self_loops` | Boolean; default `false` | Let GATv2 add self-loop attention messages. Separate from `graph.allow_self_loops`. |

If `concat: true` and `num_layers > 1`, `hidden_dim` must be divisible by `heads`. If `final_concat: true`, `output_dim` must be divisible by `heads`.

### GINE-only settings

These keys live under `model.gnn.gine`. **GINE requires `model.edge_features` to produce an `edge_attr` tensor.**

| Attribute | Supported values / default | Meaning |
|---|---|---|
| `train_eps` | Boolean; default `true` | Make the GINE epsilon coefficient trainable. |
| `eps` | Finite float; default `0.0` | Initial epsilon for GINE's skip/aggregation weighting. |
| `mlp.hidden_dim` | Positive integer; default `128` | Hidden width of the MLP inside each GINE layer. |
| `mlp.num_layers` | Positive integer; default `2` | Depth of each GINE layer's MLP. |
| `mlp.activation` | PyG-resolved activation string; default `relu` | MLP nonlinearity. Examples: `relu` \| `leaky_relu` \| `elu` \| `gelu` \| `silu` \| `tanh`. |
| `mlp.dropout` | Float `0 <= p < 1`; default `0.1` | Dropout inside the GINE MLP. |
| `mlp.normalization` | PyG MLP normalization name or `null`; default `layer_norm` | Normalization inside the GINE MLP. Common options include `layer_norm`, `batch_norm`, and `null`, subject to the installed PyG resolver. |
| `mlp.plain_last` | Boolean; default `true` | If true, the MLP's final layer omits activation, normalization, and dropout. |

Unlike `model.gnn.normalization`, `model.gnn.gine.mlp.normalization` is passed directly to PyG's MLP normalization resolver. Use `null` to disable it; don't assume all of the outer encoder's aliases work identically here.

## 7. Recommended workflow for graph experiments

1. Copy your graph YAML to a new filename and use a **new `dataset.root`** if changing raw features, topology, or transforms. This avoids accidentally reusing incompatible cached graph files.
2. Set `model.node_features` and `model.edge_features` to groups that actually exist after your transforms run. Keep the training-only FMM split for any transition-count features.
3. Point `paths.graph_config` in your experiment YAML to the new graph YAML.
4. Use `--validate-only` to check required source inputs, then run the experiment to build the cache and train. Remember that `--validate-only` does not execute expensive graph transforms.
5. Keep diffusion/training settings in the **experiment** YAML. The experiment loader ignores legacy graph YAML `model.diffusion` and `training` settings if those happen to remain from older scripts.

**Source of truth:** `src/urban_syn/gnn/config.py`, `src/urban_syn/gnn/encoder.py`, `src/urban_syn/gnn/factory.py`, `src/urban_syn/graph/config.py`, `src/urban_syn/graph/features.py`, and `src/urban_syn/graph/transforms.py`. Update this guide when their supported fields or behavior change.
