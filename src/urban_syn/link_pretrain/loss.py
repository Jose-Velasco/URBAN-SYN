"""Loss functions for binary link classification and pairwise ranking."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from .validation import validate_link_logits


def calculate_link_loss(
    positive_logits: Tensor,
    negative_logits: Tensor,
    objective: Literal["bce", "bpr"]
) -> Tensor:
    """Compute a link objective without depending on the model output type.

    Args:
        positive_logits: Raw logits for observed links, shape ``[P]``.
        negative_logits: Raw logits for source-aligned nonedges, shape
            ``[P, amount]``; each row belongs to the corresponding positive.
        objective: ``"bce"`` to classify positive and negative links or
            ``"bpr"`` to rank each positive above its sampled negatives.

    Returns:
        Differentiable scalar tensor. BCE averages over individual link
        examples; BPR averages over all positive/negative comparisons.

    Raises:
        ValueError: If the objective is unknown or the logits have invalid
            shapes, dtypes, or devices.

    Regularization belongs to the training optimizer, not this loss function.
    Both objectives can use the same configured optimizer weight decay.
    """
    validate_link_logits(positive_logits, negative_logits)

    if objective == "bce":
        # BCEWithLogits applies sigmoid internally and avoids log(0) issues.
        logits = torch.cat((positive_logits, negative_logits.reshape(-1)))
        targets = torch.zeros_like(logits)
        targets[:positive_logits.numel()] = 1
        return F.binary_cross_entropy_with_logits(logits, targets)

    if objective == "bpr":
        # -log(sigmoid(pos - neg)) equals softplus(neg - pos). The latter
        # avoids numerical instability when the score difference is extreme.
        differences = negative_logits - positive_logits.unsqueeze(1)
        return F.softplus(differences).mean()

    raise ValueError(f"Unsupported link objective: {objective!r}.")
