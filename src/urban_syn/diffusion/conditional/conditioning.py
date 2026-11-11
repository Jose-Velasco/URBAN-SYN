"""Encode default weather/mode/temperature and optional POI fields additively."""
from __future__ import annotations

import torch
from torch import nn


class SparseMetadataEncoder(nn.Module):
    """Map manifest fields to an additive [B, condition_dim] vector.

    The input has this structure::

        {"values": {
            "weather_category": ...,     # categorical torch.long [B]
            "transportation_mode": ...,  # categorical torch.long [B]
            "temperature": ...,          # normalized floating [B]
        }, "masks": { ... }                # corresponding bool [B] tensors
        }

    Manifest construction creates encoders["weather_category"] and
    encoders["transportation_mode"] as nn.Embedding layers, and
    encoders["temperature"] as an nn.Linear layer. The optional POI field uses
    the same categorical nn.Embedding pattern. Each contribution is masked
    first, including temperature bias, then added to the shared condition.
    """
    def __init__(self, fields, condition_dim=8, all_dropout=0.1, field_dropout=0.1):
        super().__init__()
        if condition_dim < 1 or not 0 <= all_dropout <= 1 or not 0 <= field_dropout <= 1:
            raise ValueError("Positive condition dimension and dropout probabilities in [0, 1] required")
        self.fields = fields
        self.condition_dim = condition_dim
        self.all_dropout, self.field_dropout = all_dropout, field_dropout
        self.skip_conditioning = False
        names = [field["name"] for field in fields]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate metadata field names")
        self.encoders = nn.ModuleDict()
        for field in fields:
            name = field["name"]
            if field["kind"] == "categorical":
                # weather_category, transportation_mode, and optional POI:
                # categorical codes become one field-specific embedding each.
                codes = sorted(field["vocabulary"].values())
                if codes != list(range(len(codes))) or field["vocabulary"].get("<unknown>") != 0:
                    raise ValueError("Categorical vocabulary must reserve code 0 for <unknown>")
                self.encoders[name] = nn.Embedding(len(codes), condition_dim, padding_idx=0)
            elif field["kind"] == "continuous":
                # temperature is already training-normalized before this layer.
                # A bias lets a known mean-valued observation differ from a missing field.
                # The entire result, including bias, is masked for missing observations.
                self.encoders[name] = nn.Linear(1, condition_dim)
            else:
                raise ValueError(f"Unsupported metadata kind: {field['kind']}")

    def forward(self, metadata, batch_size, device, dtype):
        """Return contributions with exact-zero output for every unavailable field."""
        if self.skip_conditioning:
            return torch.zeros(batch_size, self.condition_dim, device=device, dtype=dtype)
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict) or set(metadata) - {"values", "masks"}:
            raise ValueError("Metadata must contain optional 'values' and 'masks' dictionaries")
        values, masks = metadata.get("values", {}), metadata.get("masks", {})
        if set(values) - set(self.encoders) or set(masks) - set(self.encoders):
            raise ValueError("Unknown metadata field name")
        if set(masks) - set(values):
            raise ValueError("A mask requires a corresponding field value")
        condition = torch.zeros(batch_size, self.condition_dim, device=device, dtype=dtype)
        keep_all = torch.ones(batch_size, dtype=torch.bool, device=device)
        if self.training and self.all_dropout:
            keep_all = torch.rand(batch_size, device=device) >= self.all_dropout
        for field in self.fields:
            name = field["name"]
            if name not in values:
                continue
            value = values[name]
            if value.shape != (batch_size,) or value.device != device:
                raise ValueError(f"{name}: values must have shape [B] on the trajectory device")
            mask = masks.get(name, torch.ones(batch_size, dtype=torch.bool, device=device))
            if mask.shape != (batch_size,) or mask.dtype != torch.bool or mask.device != device:
                raise ValueError(f"{name}: availability mask must be bool [B] on the trajectory device")
            if field["kind"] == "categorical":
                if value.dtype != torch.long:
                    raise TypeError(f"{name}: category codes must be torch.long")
                mask = mask & (value != 0)
                if torch.any(mask & ((value < 0) | (value >= self.encoders[name].num_embeddings))):
                    raise ValueError(f"{name}: category code outside vocabulary")
                safe_value = torch.where(mask, value, 0)
                contribution = self.encoders[name](safe_value)
            else:
                if not value.is_floating_point():
                    raise TypeError(f"{name}: continuous values must be floating point")
                if torch.any(mask & ~torch.isfinite(value)):
                    raise ValueError(f"{name}: known continuous values must be finite")
                safe_value = torch.where(mask, value, 0).to(dtype=dtype)
                contribution = self.encoders[name](safe_value.unsqueeze(-1))
            keep = mask & keep_all
            if self.training and self.field_dropout:
                keep = keep & (torch.rand(batch_size, device=device) >= self.field_dropout)
            # Mask each field contribution, then add it to the shared condition;
            # enforce zero even if a checkpoint contains a nonzero row-0 embedding.
            condition = condition + torch.where(keep.unsqueeze(-1), contribution, 0)
        return condition


def encode_requests(requests, fields, device="cpu"):
    """Encode raw categorical/Celsius requests into values and availability masks.

    Categorical values use saved training vocabularies; temperatures are
    normalized with saved training statistics. Omitted, None, and unknown
    values are masked. POI identities, coordinates, dates, distances, and
    context gaps are provenance and never enter the encoder.
    """
    names = {field["name"] for field in fields}
    if any(not isinstance(row, dict) or set(row) - names for row in requests):
        raise ValueError("Generation request contains an unsupported metadata field")
    values, masks = {}, {}
    for field in fields:
        name = field["name"]
        raw = [row.get(name) for row in requests]
        if field["kind"] == "categorical":
            codes = [field["vocabulary"].get(value, 0) if value is not None else 0 for value in raw]
            values[name] = torch.tensor(codes, dtype=torch.long, device=device)
            masks[name] = values[name] != 0
        else:
            array = torch.tensor([float(value) if value is not None else float("nan") for value in raw],
                                 dtype=torch.float32, device=device)
            stats = field["normalization"]
            known = torch.isfinite(array) & (stats["observed_training_rows"] > 0)
            normalized = (torch.where(known, array, stats["mean"]) - stats["mean"]) / stats["scale"]
            values[name], masks[name] = normalized, known
    return {"values": values, "masks": masks}


def slice_metadata(metadata, indices):
    """Select a batch while preserving every sparse field's values and masks."""
    return {group: {name: value[indices] for name, value in fields.items()}
            for group, fields in metadata.items()}
