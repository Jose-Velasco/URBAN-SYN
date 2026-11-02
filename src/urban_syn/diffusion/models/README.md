# Diff-RNTraj models

This directory contains the copied model implementation. See the [package
overview](../README.md) and [conditional contracts](../conditional/README.md).

## Modules and forward path

model.py implements Diff_RNTraj. forward looks up SE[road IDs], concatenates each
road embedding with its normalized rate, and invokes the loss helper. generate_data
runs reverse sampling, separates D road-embedding features from the rate, and
decodes each generated embedding with chunked cosine similarity against SE.
Rates are postprocessed by (clamp(rate, -1, 1) + 1) / 2 by the caller.

diff_module.py implements diff_CSDI. It concatenates a broadcast [B,C] condition
to [B,L,D+1] noisy trajectories, applies a 1x1 projection and ReLU, adds a
sinusoidal timestep embedding projected through two SiLU layers, then runs gated
dilated-convolution residual blocks with dilations 1, 2, 4, 8 repeating.
Residual and skip 1x1 projections are used; skips are summed and divided by the
square root of the number of blocks before the output layer. Gating is
tanh(first half) times sigmoid(second half), and residual returns are scaled by
sqrt(0.5).

diff_util.py implements forward noising, reconstruction, connectivity counting,
and the reverse DDPM update.

## Equations

For batch B, length L, embedding dimension D, road embedding e, and normalized
rate r:

~~~text
x₀[b,l] = concat(e[road_id[b,l]], r[b,l]) ∈ R^(D+1)
αₜ      = 1 − βₜ
ᾱₜ     = product(s=0..t) αₛ
xₜ      = sqrt(ᾱₜ) x₀ + sqrt(1−ᾱₜ) ε,  ε ~ N(0,I)
~~~

The network predicts εhat. The executable losses are:

~~~text
L_noise = MSE(εhat, ε)
xhat₀    = (xₜ − sqrt(1−ᾱₜ) εhat) / sqrt(ᾱₜ)
L_x0     = MSE(xhat₀[:, :, :D], x₀[:, :, :D])
~~~

The reconstructed road embedding is mapped to its maximum dot-product road ID.
Consecutive IDs form sparse keys road_id_t * 2^32 + road_id_(t+1):

~~~text
L_connectivity = sum 1[key_t is absent from graph_edges]
L_pilot        = L_noise + L_connectivity + L_x0
~~~

Connectivity is an unchanged, nondifferentiable count; only the MSE terms train
the model. Reverse sampling uses:

~~~text
muₜ = (1 / sqrt(αₜ)) [xₜ − ((1−αₜ) / sqrt(1−ᾱₜ)) εhat(xₜ,t,c)]
σₜ² = ((1−ᾱₜ₋₁) / (1−ᾱₜ)) (1−αₜ)
xₜ₋₁ = muₜ + σₜ z, when t > 0
x₀   = mu₀,          when t = 0
~~~

The final step draws no additional noise.

## Shapes and API

| Object | Shape/type | Meaning |
| --- | --- | --- |
| noisy trajectory | [B,L,D+1] | D embedding features plus normalized rate |
| road table SE | [N,D] | indexed by road IDs |
| v1 weather | [B], torch.long | one vocabulary code per trajectory |
| v2 values/masks | per-field [B] and bool [B] | sparse metadata dictionary |
| condition | [B,C] | broadcast to [B,L,C] |
| prediction | [B,L,D+1] | unchanged output dimensions |
| generated IDs/rates | [B,L], [B,L] | decoded IDs and continuous rates |

The legacy final argument is named weather. V1 accepts a weather tensor or None
(None means all code zero). V2 accepts a metadata dictionary with values and
masks or None for the shared all-missing path. Condition and trajectory must be
on the same device, and validation preserves prediction dimensions.

Inherited definitions unused by this trajectory forward path include
get_torch_trans, the residual block upsample_conv2d and mel_conv, diff_CSDI’s
GRU, BatchNorm1d, LayerNorm, and the mask_mat argument.

## Runtime and checkpoints

Schedule tensors follow parameter device and dtype. Pilot generation uses
model.eval(), disabling metadata dropout. A supplied torch.Generator controls
initial and non-final reverse draws; otherwise PyTorch’s default RNG is used.
Diagnostics call model.train() before their one optimizer update.

To restore a checkpoint, load it, construct make_model(checkpoint["config"]), and
load its state dictionary with strict=True. Checkpoints carry configuration,
schema/field descriptors, and hashes for sidecars, road mappings, and embeddings;
they store state and configuration rather than import paths, so reorganization
requires no checkpoint migration.

