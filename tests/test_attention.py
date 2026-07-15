import torch

from ai.core import set_seed
from ai.transformer import SelfAttention


def main() -> None:
    set_seed(42)

    embedding_dim = 8
    sequence_length = 4

    attention = SelfAttention(
        embedding_dim=embedding_dim,
        dropout=0.0,
    )

    attention.eval()

    x = torch.randn(
        2,
        sequence_length,
        embedding_dim,
    )

    output, weights = attention(
        x,
        return_attention=True,
    )

    print("Input shape:")
    print(x.shape)

    print("\nOutput shape:")
    print(output.shape)

    print("\nAttention weights shape:")
    print(weights.shape)

    print("\nFirst sequence attention weights:")
    print(weights[0])

    assert output.shape == x.shape

    assert weights.shape == (
        2,
        sequence_length,
        sequence_length,
    )

    future_mask = torch.triu(
        torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
        ),
        diagonal=1,
    )

    future_attention_values = weights[:, future_mask]

    assert torch.allclose(
        future_attention_values,
        torch.zeros_like(future_attention_values),
        atol=1e-7,
    )

    row_sums = weights.sum(dim=-1)

    assert torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1e-6,
    )

    print("\nCausal mask test passed.")
    print("Attention probability test passed.")
    print("Self-attention test completed successfully.")


if __name__ == "__main__":
    main()