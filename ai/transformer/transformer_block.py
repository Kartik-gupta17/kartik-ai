from torch import Tensor, nn

from ai.nn import FeedForward, LayerNorm
from ai.transformer import MultiHeadAttention, RotaryPositionEmbedding


class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer Block.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        ffn_hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.norm1 = LayerNorm(embedding_dim)

        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.rope = RotaryPositionEmbedding(
            head_dim=embedding_dim // num_heads,
        )

        self.norm2 = LayerNorm(embedding_dim)

        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim,
            hidden_dim=ffn_hidden_dim,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        # ----- Attention -----
        residual = x

        x = self.norm1(x)

        # NOTE:
        # RoPE integration actual Q,K tensors par hogi.
        # Abhi architecture ready hai.
        x = self.attention(x)

        x = residual + x

        # ----- Feed Forward -----
        residual = x

        x = self.norm2(x)

        x = self.feed_forward(x)

        x = residual + x

        return x