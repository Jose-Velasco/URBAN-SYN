# URBAN-SYN connectivity ablation matrix

Both road-GNN architectures (**GINE** and **GATv2**) use the same six joint-training variants, each with a full preset and a matching `_smoke` preset.

| Variant | Joint GNN initialization | Joint GNN training | Pretraining source | Connectivity scorer | Added joint loss |
|---|---|---|---|---|---|
| `baseline` | Random | Jointly trainable | None | None | None |
| `baseline_warmup` | Random | Jointly trainable with gradient warmup | None | None | None |
| `bce_frozen` | BCE-pretrained | Frozen | BCE | BCE-pretrained, frozen | BCE |
| `bpr_frozen` | BPR-pretrained | Frozen | BPR | BPR-pretrained, frozen | BPR |
| `bce_warmup` | BCE-pretrained | Warmup fine-tuning | BCE | BCE-pretrained, frozen | BCE |
| `bpr_warmup` | BPR-pretrained | Warmup fine-tuning | BPR | BPR-pretrained, frozen | BPR |

`baseline_warmup` applies the **existing normalized-sigmoid gradient schedule** to a randomly initialized, trainable GNN during joint diffusion training. Neither baseline uses a pretrained encoder, a connectivity scorer, or a BCE/BPR connectivity regularizer. The warmup schedule parameters are configurable in `encoder/warmup_random.yaml`.

## Presets and outputs

- Four existing road-link pretraining sources (GINE/GATv2 x BCE/BPR).
- Twelve full joint presets, followed by twelve matching smoke presets at the bottom of `study.yaml`.
- Joint `metrics.parquet` export and centralized MLflow behavior are retained unchanged.
- No changes to checkpoint saving, graph preparation, pipeline stages, Slurm submission, or local workflow execution.

The removed `bce_scratch` and `bpr_scratch` names previously meant a random GNN with a **pretrained frozen scorer**, not the intended no-pretraining ablation. They are no longer registered or supplied as Hydra presets.
