import pytest
import torch

from ai.transformer import MultiHeadAttention


def test_multi_head_attention_output_shape() -> None:
    batch_size = 2
    sequence_length = 8
    embedding_dim = 32
    num_heads = 4

    attention = MultiHeadAttention(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        dropout=0.0,
    )

    x = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim,
    )

    output = attention(x)

    assert output.shape == (
        batch_size,
        sequence_length,
        embedding_dim,
    )


def test_attention_weights_shape() -> None:
    batch_size = 2
    sequence_length = 6
    embedding_dim = 24
    num_heads = 3

    attention = MultiHeadAttention(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        dropout=0.0,
    )

    x = torch.randn(
        batch_size,
        sequence_length,
        embedding_dim,
    )

    output, weights = attention(
        x,
        return_attention=True,
    )

    assert output.shape == (
        batch_size,
        sequence_length,
        embedding_dim,
    )

    assert weights.shape == (
        batch_size,
        num_heads,
        sequence_length,
        sequence_length,
    )


def test_causal_mask_blocks_future_tokens() -> None:
    attention = MultiHeadAttention(
        embedding_dim=16,
        num_heads=4,
        dropout=0.0,
    )

    x = torch.randn(1, 5, 16)

    _, weights = attention(
        x,
        return_attention=True,
    )

    # Upper triangular area future tokens hai.
    future_weights = torch.triu(
        weights,
        diagonal=1,
    )

    assert torch.allclose(
        future_weights,
        torch.zeros_like(future_weights),
        atol=1e-7,
    )


def test_attention_rows_sum_to_one() -> None:
    attention = MultiHeadAttention(
        embedding_dim=16,
        num_heads=4,
        dropout=0.0,
    )

    x = torch.randn(2, 5, 16)

    _, weights = attention(
        x,
        return_attention=True,
    )

    row_sums = weights.sum(dim=-1)

    assert torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1e-6,
    )


def test_backward_pass() -> None:
    attention = MultiHeadAttention(
        embedding_dim=32,
        num_heads=4,
        dropout=0.0,
    )

    x = torch.randn(
        2,
        6,
        32,
        requires_grad=True,
    )

    output = attention(x)

    loss = output.mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_invalid_head_configuration() -> None:
    with pytest.raises(ValueError):
        MultiHeadAttention(
            embedding_dim=30,
            num_heads=4,
        )
def test_attention_contains_rope() -> None:
    attention = MultiHeadAttention(
        embedding_dim=32,
        num_heads=4,
        dropout=0.0,
    )

    assert hasattr(attention, "rope")
    assert attention.rope.head_dim == 8


def test_invalid_odd_rope_head_dimension() -> None:
    with pytest.raises(
        ValueError,
        match="even head dimension",
    ):
        MultiHeadAttention(
            embedding_dim=15,
            num_heads=3,
        )


def test_rope_attention_output_is_finite() -> None:
    attention = MultiHeadAttention(
        embedding_dim=32,
        num_heads=4,
        dropout=0.0,
    )

    x = torch.randn(2, 12, 32)

    output, weights = attention(
        x,
        return_attention=True,
    )

    assert torch.isfinite(output).all()
    assert torch.isfinite(weights).all()