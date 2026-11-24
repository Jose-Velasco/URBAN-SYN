"""Validated fixed road inputs and an adapter around the retained denoiser."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from ...data import RoadSequenceDataset
from ...models.diff_module import diff_CSDI
from ...models.diff_util import cal_x0_from_noise_ddpm
from ..conditioning import SparseMetadataEncoder
from ..metadata import sha256


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def positive_ids(frame):
    ids = pd.to_numeric(frame.fmm_id, errors='raise')
    if frame.fmm_id.map(lambda value: isinstance(value, bool)).any() or ids.isna().any() or (ids <= 0).any() or (ids % 1 != 0).any() or ids.duplicated().any():
        raise ValueError('Trajectory fmm_id values must be unique positive integers')


class FixedRoads:
    """NPZ vocabulary order defines model ID i+1; row zero is padding only.

    The bundle binds embeddings AND raw-ID transitions to a network file hash.
    This explicit provenance is required even if every road happens to overlap.
    """
    def __init__(self, bundle, network, sequence_paths, report):
        with np.load(bundle, allow_pickle=False) as data:
            ids, embeddings = data['edge_ids'], data['embeddings']
            pairs = data['transitions']
            if str(data['network_sha256'].item()) != sha256(network):
                raise ValueError('Embedding/network identity mismatch; generate matching fixed embeddings')
        if ids.ndim != 1 or ids.dtype.kind not in 'iu' or (ids < 0).any() or len(set(ids.tolist())) != len(ids):
            raise ValueError('Embedding road mapping must contain unique nonnegative integer edge_ids')
        if embeddings.dtype.kind not in 'fiu' or embeddings.ndim != 2 or embeddings.shape[0] != len(ids) or not len(ids) or not embeddings.shape[1] or not np.isfinite(embeddings).all():
            raise ValueError('Invalid or nonfinite fixed embeddings')
        embeddings32 = embeddings.astype(np.float32)
        if not np.isfinite(embeddings32).all():
            raise ValueError('Fixed embeddings are not finite at model precision')
        if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in 'iu':
            raise ValueError('Transitions must be raw integer road ID pairs')
        if not set(pairs.ravel()).issubset(set(ids)):
            raise ValueError('Transition mapping differs from embedding vocabulary')
        vocabulary = set(ids.tolist())
        missing = {}
        frames = []
        for path in sequence_paths:
            frame = pd.read_parquet(path)
            positive_ids(frame)
            if (frame.num_points <= 0).any() or (frame.num_points % 1 != 0).any() or frame.num_points.isna().any():
                raise ValueError('Prepared trajectories must have positive integer lengths; no filtering allowed')
            frames.append(frame)
            for row in frame.itertuples():
                raw = np.asarray(row.edge_ids)
                if raw.dtype.kind not in 'iu' or (raw < 0).any():
                    raise ValueError('Sequence road IDs must be nonnegative integers')
                absent = sorted(set(map(int, raw)) - vocabulary)
                if absent:
                    missing.setdefault(str(path), {})[str(row.fmm_id)] = absent
        if set(frames[0].fmm_id) & set(frames[1].fmm_id):
            raise ValueError('Train/test FMM IDs overlap')
        if missing:
            Path(report).parent.mkdir(parents=True, exist_ok=True)
            Path(report).write_text(json.dumps(missing, indent=2)+'\n')
            raise ValueError(f'Missing roads; trajectories preserved. See {report}')
        self.edge_ids = torch.from_numpy(ids.astype(np.int64))
        self.embeddings = torch.cat((torch.zeros(1, embeddings.shape[1]), torch.tensor(embeddings32, dtype=torch.float32)))
        self.pairs = {tuple(map(int, pair)) for pair in pairs}
        self.fingerprints = dict(bundle=sha256(bundle), network=sha256(network),
                                 vocabulary=digest(ids.tolist()), embeddings=hashlib.sha256(embeddings.tobytes()).hexdigest())

    def dataset(self, path):
        return RoadSequenceDataset(path, self.edge_ids)


class FixedDiffusion(nn.Module):
    """Use retained fixed-embedding architecture with controlled initialization."""
    def __init__(self, settings, roads, fields, enabled, seed, metadata_seed):
        super().__init__()
        dim = roads.embeddings.shape[1]
        # Shared initialization runs on CPU in its own RNG scope in both modes.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            net = diff_CSDI(settings['channels'], settings['channels'], settings['steps'],
                           settings['channels'], dim, settings['residual_layers'],
                           metadata_fields=[], condition_dim=settings['condition_dim'])
            projection = nn.Conv1d(dim+1, settings['channels'], 1)
            nn.init.kaiming_normal_(projection.weight)
            net.input_projection = projection
            net.metadata_encoder = None
            net.skip_conditioning = True
        if enabled:
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(metadata_seed)
                net.metadata_encoder = SparseMetadataEncoder(fields, settings['condition_dim'],
                                                              settings['all_dropout'], settings['field_dropout'])
                wider = nn.Conv1d(dim+1+settings['condition_dim'], settings['channels'], 1)
                nn.init.kaiming_normal_(wider.weight)
                with torch.no_grad():
                    wider.weight[:, :dim+1].copy_(projection.weight)
                    wider.bias.copy_(projection.bias)
                net.input_projection = wider
                net.skip_conditioning = False
        self.net = net
        self.register_buffer('fixed_embeddings', roads.embeddings.clone(), persistent=False)
        beta = torch.linspace(settings['beta_start']**.5, settings['beta_end']**.5, settings['steps']).square()
        self.register_buffer('alpha', 1-beta)
        self.register_buffer('alpha_bar', (1-beta).cumprod(0))
        self.chunk_size = settings['comparison_chunk']
        self.steps = settings['steps']
        self.edge_ids = roads.edge_ids

    def losses(self, ids, rates, metadata=None, generator=None):
        clean = torch.cat((self.fixed_embeddings[ids], (rates*2-1).unsqueeze(-1)), -1)
        t = torch.randint(self.steps, (len(ids),1), device=ids.device, generator=generator)
        noise = torch.randn(clean.shape, device=ids.device, generator=generator)
        a = self.alpha_bar[t].unsqueeze(-1)
        noisy = a.sqrt()*clean + (1-a).sqrt()*noise
        predicted = self.net(noisy, t, metadata)
        reconstructed = (noisy-(1-a).sqrt()*predicted)/a.sqrt()
        noise_loss = (predicted-noise).square().mean((1,2))
        road_loss = (reconstructed[...,:-1]-clean[...,:-1]).square().mean((1,2))
        return noise_loss, road_loss

    def decode(self, states):
        # Padding is never a candidate. Chunk both query and vocabulary axes.
        vectors = nn.functional.normalize(states[...,:-1].reshape(-1, self.fixed_embeddings.shape[1]), dim=-1)
        candidates = nn.functional.normalize(self.fixed_embeddings[1:], dim=-1)
        result = []
        for query in vectors.split(self.chunk_size):
            best = query.new_full((len(query),), -float('inf'))
            chosen = torch.zeros(len(query), dtype=torch.long, device=query.device)
            for start in range(0,len(candidates),self.chunk_size):
                values, indices = (query @ candidates[start:start+self.chunk_size].T).max(-1)
                update = values > best
                chosen = torch.where(update, indices+start+1, chosen)
                best = torch.maximum(values,best)
            result.append(chosen)
        return torch.cat(result).reshape(states.shape[:2])

    @torch.no_grad()
    def generate(self, count, length, metadata, generator):
        states = cal_x0_from_noise_ddpm(self.net, dict(T=self.steps, alpha=self.alpha, alpha_bar=self.alpha_bar),
                                       count,length,self.fixed_embeddings.shape[1]+1,metadata,generator=generator)
        ids = self.decode(states)
        raw = self.edge_ids.to(ids.device)[ids-1]
        return raw.cpu(), ((states[...,-1].clamp(-1,1)+1)/2).cpu()
