"""Forward diffusion, denoising, and DDPM sampling operations.

Training objectives and road-connectivity calculations are handled
separately by the loss implementations.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .road_ids import encode_transition_keys
from .sampling.ddpm import cal_x0_from_noise_ddpm

# Changed for our model version
# Temporary compatibility for the original UTGraph connectivity check.
# This will be revisited when adapting the connectivity loss.
# This avoids copying build_graph_nyc.py just to satisfy that import.
# EDGE_KEY_BASE = 1 << 32

@dataclass(frozen=True)
class DiffusionForwardOutput:
    """Intermediate tensors from forward diffusion and denoising.

    Attributes
    ----------
    clean : Tensor
        Original clean trajectory representations [B, T, F].
    noisy : Tensor
        Noisy representations at sampled timesteps [B, T, F].
    sampled_noise : Tensor
        Gaussian noise added to the clean trajectories [B, T, F].
    predicted_noise : Tensor
        Denoiser's predicted noise [B, T, F].
    reconstructed : Tensor
        Clean representations estimated from predicted noise [B, T, F].
    timesteps : Tensor
        Sampled diffusion timestep per trajectory [B, 1].
    """

    clean: Tensor
    noisy: Tensor
    sampled_noise: Tensor
    predicted_noise: Tensor
    reconstructed: Tensor
    timesteps: Tensor


def std_normal(
    size: tuple[int, ...],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None
) -> Tensor:
    """Draw standard normal noise.

    Parameters
    ----------
    size : tuple[int, ...]
        Shape of the generated tensor.
    device : torch.device | None, default=None
        Device for the generated tensor.
    dtype : torch.dtype | None, default=None
        Tensor dtype.

    Returns
    -------
    Tensor
        Gaussian noise of the requested shape and dtype.
    """
    return torch.randn(size, device=device, dtype=dtype)


def diffuse_and_denoise(
    net: nn.Module,
    clean: Tensor,
    diffusion_hyperparams: dict
) -> DiffusionForwardOutput:
    """Apply forward diffusion, predict noise, and reconstruct clean input.

    Parameters
    ----------
    net : nn.Module
        Denoiser taking (noisy_trajectories, timesteps).
    clean : Tensor
        Clean trajectories with shape [B, T, F].
    diffusion_hyperparams : dict
        Diffusion schedule with T and alpha_bar.

    Returns
    -------
    DiffusionForwardOutput
        Shared forward and denoising tensors for independent losses.
    """
    if clean.ndim != 3:
        raise ValueError("Expected clean trajectories with shape [B, T, F].")

    batch_size = clean.size(0)
    diffusion_steps = int(diffusion_hyperparams["T"])
    alpha_bar = diffusion_hyperparams["alpha_bar"]

    timesteps = torch.randint(
        diffusion_steps,
        (batch_size, 1),
        device=clean.device
    )
    alpha_bar_t = alpha_bar[timesteps].unsqueeze(-1)
    sampled_noise = torch.randn_like(clean)

    sqrt_alpha = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha = torch.sqrt(1 - alpha_bar_t)
    noisy = sqrt_alpha * clean + sqrt_one_minus_alpha * sampled_noise

    predicted_noise = net(noisy, timesteps)
    reconstructed = (
        noisy - sqrt_one_minus_alpha * predicted_noise
    ) / sqrt_alpha

    return DiffusionForwardOutput(
        clean=clean,
        noisy=noisy,
        sampled_noise=sampled_noise,
        predicted_noise=predicted_noise,
        reconstructed=reconstructed,
        timesteps=timesteps
    )

def diff_forward_x0_constraint(
        net: nn.Module,
        x: torch.Tensor,
        diffusion_hyperparams: dict,
        road_embeddings: torch.Tensor,
        valid_transition_keys: torch.Tensor,
        road_id_base: int,
        similarity_chunk_size: int = 128
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute noise prediction, connectivity, and reconstruction losses.

    Parameters
    ----------
    net : nn.Module
        Diffusion noise-prediction network.
    x : Tensor
        Embedded road trajectories and position rates [B, T, D + 1].
    diffusion_hyperparams : dict
        Diffusion schedule parameters.
    road_embeddings : Tensor
        GNN-generated road embeddings [N + 1, D], including padding.
    valid_transition_keys : Tensor
        Encoded valid directed road transitions.
    road_id_base : int
        Number of model road IDs, including padding (N + 1).
    similarity_chunk_size : int, default=128
        Number of trajectory positions processed per similarity batch.

    x: vectorized RNTraj

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Noise prediction, connectivity, and reconstruction losses.
    """
    _dh = diffusion_hyperparams
    T, Alpha_bar, Alpha = _dh["T"], _dh["alpha_bar"], _dh["alpha"]
    
    # audio = x
    B, C, L = x.shape  # B is batchsize, C=1, L is audio length
    diffusion_steps = torch.randint(T, size=(B,1,1), device=x.device)  # randomly sample diffusion steps from 1~T
    
    z = std_normal(x.shape)
    
    xt = torch.sqrt(Alpha_bar[diffusion_steps]) * x + torch.sqrt(1-Alpha_bar[diffusion_steps]) * z  # compute x_t from q(x_t|x_0)
    #这里得到的是噪声, loss函数之一
    pred_noise = net(xt, diffusion_steps.view(B,1))  # predict \epsilon according to \epsilon_\theta
    
    noise_func = nn.MSELoss()
    xt_noise = noise_func(pred_noise, z)  #将预测的x0与真实的x0计算loss
    
    x0_hat = (xt - torch.sqrt(1-Alpha_bar[diffusion_steps]) * pred_noise) / torch.sqrt(Alpha_bar[diffusion_steps])
    x0_hat = x0_hat[:,:,:L-1]  # B, T, F
    x0_loss = noise_func(x0_hat, x[:,:,:L-1])

    # computes the dot product between each predicted road vector and every road embedding.
    # id_sim = torch.einsum('btf,nf->btn',x0_hat,SE)

    # selects the road with the highest dot-product similarity.
    # id_sim = id_sim.argmax(-1)

    # Change for our model version
    # Find predicted road IDs using chunked dot-product similarity (what the original code did).
    # This avoids allocating the complete [B, C, num_roads] matrix.
    queries = x0_hat.reshape(-1, x0_hat.size(-1))

    # The original connectivity check and in the Diff_RNTraj paper uses argmax, so gradients are not needed here
    # since it should not produce gradients anyways
    with torch.no_grad():
        predicted_ids = []

        for query_batch in queries.split(similarity_chunk_size):
            similarities = query_batch @ road_embeddings.T
            predicted_ids.append(similarities.argmax(dim=-1))

        id_sim = torch.cat(predicted_ids).reshape(B, C)

    # mask = spatial_A_trans[id_sim[:, :-1], id_sim[:, 1:]] == 1e-10
    # Change for Diff-RNTraj (NYC): spatial_A_trans is a sorted tensor of edge keys, so look up
    # each (road_t, road_t+1) pair in it. mask is True for transitions that are not
    # UTGraph edges, exactly like the dense "== 1e-10" test above.
    # Encode consecutive predicted road IDs into integer transition keys.
    # Model road IDs are PyG node indices + 1, reserving 0 for padding.
    # pair_keys = id_sim[:, :-1].long() * road_id_base + id_sim[:, 1:].long()
    pair_keys = encode_transition_keys(
        id_sim[:, :-1],
        id_sim[:, 1:],
        road_id_base,
    )

    # Original connectivity loss: count transitions absent from the
    # supplied valid road connections.
    mask = ~torch.isin(pair_keys, valid_transition_keys)
    loss = torch.sum(mask).float()
    
    return xt_noise, loss, x0_loss 

# def cal_x0_from_noise_ddpm(net, diffusion_hyperparams, batchsize, length,  feature):
#     _dh = diffusion_hyperparams
#     T, Alpha_bar, Alpha = _dh["T"], _dh["alpha_bar"], _dh["alpha"]
    
#     diff_input = std_normal((batchsize, length, feature))
    
#     with torch.no_grad():
#         for t in range(T-1, -1, -1):
            
#             predict_noise = net(diff_input, t)  
            
#             coeff1 = 1 / (Alpha[t] ** 0.5)
#             coeff2 = (1 - Alpha[t]) / ((1 - Alpha_bar[t]) ** 0.5)
#             diff_input = coeff1 * (diff_input - coeff2 * predict_noise)
            
#             if t>0:
#                 noise = std_normal(diff_input.shape)
#                 sigma = ( (1 - Alpha_bar[t-1]) / (1 - Alpha_bar[t]) * (1 - Alpha[t]) ) ** 0.5
#                 diff_input += sigma * noise
                
#     return diff_input
