"""
Layer Normalization for KartikAI.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)

        x = (x - mean) / torch.sqrt(variance + self.eps)

        return self.weight * x + self.bias