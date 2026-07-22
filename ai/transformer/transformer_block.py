from torch import Tensor, nn

from ai.nn import FeedForward, LayerNorm
from .multi_head_attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer block.

    Architecture:

        Input
          ↓
        LayerNorm
          ↓
        Multi-Head Attention + RoPE
          ↓
        Residual Connection
          ↓
        LayerNorm
          ↓
        Feed Forward Network
          ↓
        Residual Connection
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        ffn_hidden_dim: int,
        dropout: float = 0.1,
        bias: bool = True,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be greater than 0."
            )

        if ffn_hidden_dim <= 0:
            raise ValueError(
                "ffn_hidden_dim must be greater than 0."
            )

        self.norm1 = LayerNorm(embedding_dim)

        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            rope_base=rope_base,
        )

        self.norm2 = LayerNorm(embedding_dim)

        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim,
            hidden_dim=ffn_hidden_dim,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(
                "Input must have shape "
                "(batch_size, sequence_length, embedding_dim)."
            )

        # Pre-Norm Attention + Residual
        x = x + self.attention(
            self.norm1(x)
        )

        # Pre-Norm Feed Forward + Residual
        x = x + self.feed_forward(
            self.norm2(x)
        )

        return x