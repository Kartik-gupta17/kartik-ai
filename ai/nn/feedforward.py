"""
Feed Forward Network for KartikAI.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ai.nn.linear import Linear
from ai.nn.dropout import Dropout


class FeedForward(nn.Module):
    """
    Transformer Feed Forward Network.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:

        super().__init__()

        self.linear1 = Linear(
            embedding_dim,
            hidden_dim,
        )

        self.activation = nn.GELU()

        self.dropout = Dropout(dropout)

        self.linear2 = Linear(
            hidden_dim,
            embedding_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.linear1(x)

        x = self.activation(x)

        x = self.dropout(x)

        x = self.linear2(x)

        return x