"""Copied from the baseline model module; supports weather and sparse metadata conditioning."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy
import copy
from ..conditional.conditioning import SparseMetadataEncoder

def get_torch_trans(heads=8, layers=1, channels=64, key_padding_mask=None):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=channels, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

def Conv1d_with_init(in_channels, out_channels, kernel_size, padding=0):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
    nn.init.kaiming_normal_(layer.weight)
    return layer

class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table

class diff_CSDI(nn.Module):
    """Predict noise for [B, L, D+1] trajectories with an additive [B, C] condition."""
    def __init__(self, channels, inputdim, num_steps, embedding_dim, pre_dim, rdcl,
                 weather_categories=None, condition_dim=8, metadata_fields=None,
                 all_dropout=0.1, field_dropout=0.1):
        super().__init__()
        self.channels = channels
        self.pre_dim = pre_dim
        self.rdcl = rdcl
        self.metadata_encoder = None
        if metadata_fields is not None:
            self.metadata_encoder = SparseMetadataEncoder(
                metadata_fields, condition_dim, all_dropout, field_dropout)
        else:
            if weather_categories is None or weather_categories < 2 or condition_dim < 1:
                raise ValueError("Need at least two weather codes and a positive condition dimension")
            self.weather_embedding = nn.Embedding(weather_categories, condition_dim, padding_idx=0)
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps,
            embedding_dim,
        )

        self.input_projection = Conv1d_with_init(self.pre_dim + 1 + condition_dim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_layer = nn.Linear(self.channels, self.pre_dim + 1)
        
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                Residual_block(
                    res_channels=self.channels, skip_channels=self.channels, 
                    dilation=2 ** (n % 4),
                    diffusion_step_embed_dim_out=embedding_dim,
                )
                for n in range(self.rdcl)
            ]
        )
        self.rnn = nn.GRU(inputdim, inputdim)
        self.bn = nn.BatchNorm1d(inputdim)
        self.out_bn = nn.BatchNorm1d(inputdim)
        self.layernorm = nn.LayerNorm(inputdim)
    def forward(self, x, diffusion_step, weather=None, mask_mat=None):
        """
        Denoise [B, L, D+1] input while preserving the output feature shape.

        In schema v2, the parameter named weather carries the complete metadata
        dictionary, including transportation mode and temperature. The
        metadata_encoder branch is the multi-field path; _encode_weather() is
        the legacy weather-only path. The selected [B, C] vector is broadcast
        to [B, L, C] and concatenated to the noisy trajectory before the input
        projection.
        """
        
        B, step_length, c = x.shape
        if c != self.pre_dim + 1:
            raise ValueError("Noisy trajectory feature dimension must remain pre_dim + 1")
        # Multi-field schema-v2 path: weather is the complete sparse dictionary.
        if self.metadata_encoder is not None:
            condition = self.metadata_encoder(weather, B, x.device, x.dtype)
        else:
            # Legacy schema-v1 weather-only path.
            condition = self._encode_weather(weather, B, x.device)
        condition = condition.unsqueeze(1).expand(-1, step_length, -1)
        x = torch.cat((x, condition), dim=-1)
        x = x.permute(0, 2, 1)   # B, F, T

        x = self.input_projection(x)
        x = F.relu(x).permute(0, 2, 1)  # B, T, F

        diffusion_emb = self.diffusion_embedding(diffusion_step)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_emb)
            skip.append(skip_connection)
        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        return self.output_layer(x)

    def _encode_weather(self, weather, batch_size, device):
        # Compatibility for the v1 pilot API; unknown now has exact-zero influence.
        if weather is None:
            weather = torch.zeros(batch_size, dtype=torch.long, device=device)
        if weather.shape != (batch_size,):
            raise ValueError("Weather must have one categorical code per trajectory: shape [B]")
        if weather.dtype != torch.long:
            raise TypeError("Weather codes must be torch.long")
        if weather.device != device:
            raise ValueError("Weather and trajectory must be on the same device")
        if torch.any((weather < 0) | (weather >= self.weather_embedding.num_embeddings)):
            raise ValueError("Weather code outside saved vocabulary")
        return self.weather_embedding(weather) * (weather != 0).unsqueeze(-1)
        
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        out = self.conv(x)
        return out

class Residual_block(nn.Module):
    def __init__(self, res_channels, skip_channels, dilation, 
                 diffusion_step_embed_dim_out):
        super(Residual_block, self).__init__()
        self.res_channels = res_channels

        # the layer-specific fc for diffusion step embedding
        self.fc_t = nn.Linear(diffusion_step_embed_dim_out, self.res_channels)

        # dilated conv layer
        self.dilated_conv_layer = Conv(self.res_channels, 2 * self.res_channels, kernel_size=3, dilation=dilation)

        # add mel spectrogram upsampler and conditioner conv1x1 layer
        self.upsample_conv2d = torch.nn.ModuleList()
        for s in [16, 16]:
            conv_trans2d = torch.nn.ConvTranspose2d(1, 1, (3, 2 * s), padding=(1, s // 2), stride=(1, s))
            conv_trans2d = torch.nn.utils.weight_norm(conv_trans2d)
            torch.nn.init.kaiming_normal_(conv_trans2d.weight)
            self.upsample_conv2d.append(conv_trans2d)
        self.mel_conv = Conv(80, 2 * self.res_channels, kernel_size=1)  # 80 is mel bands

        # residual conv1x1 layer, connect to next residual layer
        self.res_conv = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.res_conv = nn.utils.weight_norm(self.res_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)

        # skip conv1x1 layer, add to all skip outputs through skip connections
        self.skip_conv = nn.Conv1d(res_channels, skip_channels, kernel_size=1)
        self.skip_conv = nn.utils.weight_norm(self.skip_conv)
        nn.init.kaiming_normal_(self.skip_conv.weight)

    def forward(self, x, diffusion_step_embed):
        # x, mel_spec, diffusion_step_embed = input_data
        h = x
        

        # add in diffusion step embedding
        part_t = self.fc_t(diffusion_step_embed)
        # h += part_t
        # Change for Diff-RNTraj (NYC): out-of-place add. h is x, so the in-place "+=" also
        # changed x (the ReLU output in the first block), which newer PyTorch rejects during
        # backward. Same math as before: x and h both include part_t, and x is used in the return.
        x = x + part_t
        h = x

        h = h.permute(0, 2, 1)
        B, C, L = h.shape
        assert C == self.res_channels
        
        # dilated conv layer
        h = self.dilated_conv_layer(h)


        # gated-tanh nonlinearity
        out = torch.tanh(h[:,:self.res_channels,:]) * torch.sigmoid(h[:,self.res_channels:,:])

        # residual and skip outputs
        res = self.res_conv(out).permute(0, 2, 1)
        assert x.shape == res.shape
        skip = self.skip_conv(out).permute(0, 2, 1)

        return (x + res) * math.sqrt(0.5), skip  # normalize for training stability
