import math
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import nn

from ai.nn import Dropout, Linear


class MultiHeadAttention(nn.Module):
    """
    Causal Multi-Head Self-Attention.

    Input shape:
        (batch_size, sequence_length, embedding_dim)

    Output shape:
        (batch_size, sequence_length, embedding_dim)
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than 0.")

        if num_heads <= 0:
            raise ValueError("num_heads must be greater than 0.")

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be between 0 and 1.")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Q, K and V ko ek hi projection se generate karenge.
        self.qkv_projection = Linear(
            embedding_dim,
            3 * embedding_dim,
            bias=bias,
        )

        self.output_projection = Linear(
            embedding_dim,
            embedding_dim,
            bias=bias,
        )

        self.attention_dropout = Dropout(dropout)
        self.output_dropout = Dropout(dropout)

    def _split_heads(self, tensor: Tensor) -> Tensor:
        """
        Convert:

        (batch, sequence, embedding_dim)

        into:

        (batch, num_heads, sequence, head_dim)
        """

        batch_size, sequence_length, _ = tensor.shape

        tensor = tensor.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        return tensor.transpose(1, 2)

    def _merge_heads(self, tensor: Tensor) -> Tensor:
        """
        Convert:

        (batch, num_heads, sequence, head_dim)

        into:

        (batch, sequence, embedding_dim)
        """

        batch_size, _, sequence_length, _ = tensor.shape

        tensor = tensor.transpose(1, 2).contiguous()

        return tensor.view(
            batch_size,
            sequence_length,
            self.embedding_dim,
        )

    def _create_causal_mask(
        self,
        sequence_length: int,
        device: torch.device,
    ) -> Tensor:
        """
        Lower triangular mask create karta hai.

        Token sirf current aur previous tokens ko dekh sakta hai.
        Future tokens masked rahenge.
        """

        return torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=device,
                dtype=torch.bool,
            )
        )

    def forward(
        self,
        x: Tensor,
        return_attention: bool = False,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        if x.ndim != 3:
            raise ValueError(
                "Input must have shape "
                "(batch_size, sequence_length, embedding_dim)."
            )

        batch_size, sequence_length, embedding_dim = x.shape

        if embedding_dim != self.embedding_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embedding_dim}, "
                f"but received {embedding_dim}."
            )

        # Shape:
        # (batch, sequence, 3 * embedding_dim)
        qkv = self.qkv_projection(x)

        # Q, K, V alag karna
        query, key, value = qkv.chunk(3, dim=-1)

        # Shape:
        # (batch, heads, sequence, head_dim)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        # Scaled dot-product attention
        attention_scores = torch.matmul(
            query,
            key.transpose(-2, -1),
        )

        attention_scores = attention_scores / math.sqrt(self.head_dim)

        causal_mask = self._create_causal_mask(
            sequence_length=sequence_length,
            device=x.device,
        )

        # Future positions ko -infinity denge
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

        # Shape:
        # (batch, heads, sequence, head_dim)
        attention_output = torch.matmul(
            attention_weights,
            value,
        )

        # Sabhi heads ko combine karna
        attention_output = self._merge_heads(
            attention_output
        )

        output = self.output_projection(
            attention_output
        )

        output = self.output_dropout(output)

        if return_attention:
            return output, attention_weights

        return output