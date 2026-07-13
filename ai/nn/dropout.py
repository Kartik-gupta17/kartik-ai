"""
Dropout layer for KartikAI.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Dropout(nn.Module):

    def __init__(
        self,
        p: float = 0.1,
    ) -> None:

        super().__init__()

        if not 0 <= p < 1:
            raise ValueError("Dropout probability must be between 0 and 1.")

        self.p = p

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if self.training:
            return torch.nn.functional.dropout(
                x,
                p=self.p,
                training=True,
            )

        return x