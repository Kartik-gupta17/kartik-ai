"""
Causal scaled dot-product self-attention for KartikAI.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ai.nn import Dropout, Linear


class SelfAttention(nn.Module):
    """
    Single-head causal self-attention.

    Input shape:
        (batch_size, sequence_length, embedding_dim)

    Output shape:
        (batch_size, sequence_length, embedding_dim)
    """

    def __init__(
        self,
        embedding_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than zero.")

        self.embedding_dim = embedding_dim

        self.query_projection = Linear(
            in_features=embedding_dim,
            out_features=embedding_dim,
        )

        self.key_projection = Linear(
            in_features=embedding_dim,
            out_features=embedding_dim,
        )

        self.value_projection = Linear(
            in_features=embedding_dim,
            out_features=embedding_dim,
        )

        self.attention_dropout = Dropout(dropout)

    @staticmethod
    def create_causal_mask(
        sequence_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create lower-triangular causal mask.

        True means attention is allowed.
        False means token is hidden.
        """
        return torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=device,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        if x.ndim != 3:
            raise ValueError(
                "Input must have shape "
                "(batch_size, sequence_length, embedding_dim)."
            )

        if x.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, "
                f"but received {x.shape[-1]}."
            )

        sequence_length = x.shape[1]

        query = self.query_projection(x)
        key = self.key_projection(x)
        value = self.value_projection(x)

        attention_scores = torch.matmul(
            query,
            key.transpose(-2, -1),
        )

        attention_scores = attention_scores / math.sqrt(
            self.embedding_dim
        )

        causal_mask = self.create_causal_mask(
            sequence_length=sequence_length,
            device=x.device,
        )

        attention_scores = attention_scores.masked_fill(
            ~causal_mask,
            float("-inf"),
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1,
        )

        attention_weights = self.attention_dropout(
            attention_weights
        )

        output = torch.matmul(
            attention_weights,
            value,
        )

        if return_attention:
            return output, attention_weights

        return output