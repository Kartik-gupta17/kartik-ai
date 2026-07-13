"""
Token embedding layer for KartikAI.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Embedding(nn.Module):
    """
    Converts token IDs into dense vectors.

    Args:
        num_embeddings: Vocabulary size.
        embedding_dim: Size of each token vector.
        padding_idx: Optional token ID used for padding.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
    ) -> None:
        super().__init__()

        if num_embeddings <= 0:
            raise ValueError("num_embeddings must be greater than 0.")

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than 0.")

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(
            self.weight,
            mean=0.0,
            std=0.02,
        )

        if self.padding_idx is not None:
            if not 0 <= self.padding_idx < self.num_embeddings:
                raise ValueError(
                    "padding_idx must be within vocabulary range."
                )

            with torch.no_grad():
                self.weight[self.padding_idx].zero_()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                "token_ids must contain integer token IDs."
            )

        if token_ids.numel() > 0:
            minimum_id = int(token_ids.min().item())
            maximum_id = int(token_ids.max().item())

            if minimum_id < 0 or maximum_id >= self.num_embeddings:
                raise IndexError(
                    "Token ID is outside the vocabulary range."
                )

        return torch.nn.functional.embedding(
            token_ids,
            self.weight,
            padding_idx=self.padding_idx,
        )