# Joint road GNN and trajectory diffusion

The canonical pipeline uses `model.py`, `diff_module.py`, `training.py`,
`sampling/`, and the `urban_syn.experiment` configuration/training/generation
entrypoint. Road embeddings come from the joint GNN; model ID zero is padding
and is excluded from generated roads. Raw road zero remains a valid physical ID.

Optional sparse metadata is documented in [conditional/README.md](conditional/README.md),
including preparation, paired training, raw generation requests, and masked
checkpoint comparisons. Conditioning defaults to disabled and retains the
original projection and model path. Enabled metadata is concatenated before
projection; the joint GNN, road mapping, and configured loss composition remain
part of the same canonical model. `mask_mat` remains the denoiser's third
positional argument; metadata is keyword-only.

```bash
PYTHONPATH=src python -m urban_syn.experiment train --config configs/experiments/metadata_conditioning/enabled.yaml
PYTHONPATH=src python -m urban_syn.experiment generate --config YOUR_GENERATION_CONFIG.yaml
PYTHONPATH=src python -m urban_syn.diffusion.conditional.sparse_pilot --output outputs/metadata_joint_smoke_NEW
```

The files under `models/` and binary-sidecar helpers are retained for historical
reference. Current joint checkpoints include GNN weights and cannot silently
reuse those legacy pilot weights. See [experiment/README.md](../experiment/README.md)
for experiment runtime and tracking details.
