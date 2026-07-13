"""
Linear layer for KartikAI.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Linear(nn.Module):
    """
    Fully connected linear layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return torch.nn.functional.linear(
            x,
            self.weight,
            self.bias,
        )