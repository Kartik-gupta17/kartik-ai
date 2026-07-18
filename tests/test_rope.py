import pytest
import torch

from ai.transformer import RotaryPositionEmbedding


def test_rope_output_shape() -> None:
    rope = RotaryPositionEmbedding(head_dim=8)

    query = torch.randn(2, 4, 6, 8)
    key = torch.randn(2, 4, 6, 8)

    rotated_query, rotated_key = rope(query, key)

    assert rotated_query.shape == query.shape
    assert rotated_key.shape == key.shape


def test_rope_preserves_vector_norm() -> None:
    rope = RotaryPositionEmbedding(head_dim=8)

    query = torch.randn(2, 4, 6, 8)
    key = torch.randn(2, 4, 6, 8)

    rotated_query, rotated_key = rope(query, key)

    original_query_norm = torch.linalg.vector_norm(
        query,
        dim=-1,
    )

    rotated_query_norm = torch.linalg.vector_norm(
        rotated_query,
        dim=-1,
    )

    original_key_norm = torch.linalg.vector_norm(
        key,
        dim=-1,
    )

    rotated_key_norm = torch.linalg.vector_norm(
        rotated_key,
        dim=-1,
    )

    assert torch.allclose(
        original_query_norm,
        rotated_query_norm,
        atol=1e-5,
    )

    assert torch.allclose(
        original_key_norm,
        rotated_key_norm,
        atol=1e-5,
    )


def test_first_position_remains_unchanged() -> None:
    rope = RotaryPositionEmbedding(head_dim=8)

    query = torch.randn(1, 2, 5, 8)
    key = torch.randn(1, 2, 5, 8)

    rotated_query, rotated_key = rope(query, key)

    assert torch.allclose(
        rotated_query[:, :, 0],
        query[:, :, 0],
        atol=1e-6,
    )

    assert torch.allclose(
        rotated_key[:, :, 0],
        key[:, :, 0],
        atol=1e-6,
    )


def test_backward_pass() -> None:
    rope = RotaryPositionEmbedding(head_dim=8)

    query = torch.randn(
        2,
        4,
        6,
        8,
        requires_grad=True,
    )

    key = torch.randn(
        2,
        4,
        6,
        8,
        requires_grad=True,
    )

    rotated_query, rotated_key = rope(query, key)

    loss = rotated_query.mean() + rotated_key.mean()
    loss.backward()

    assert query.grad is not None
    assert key.grad is not None

    assert torch.isfinite(query.grad).all()
    assert torch.isfinite(key.grad).all()


def test_invalid_odd_head_dimension() -> None:
    with pytest.raises(ValueError):
        RotaryPositionEmbedding(head_dim=7)


def test_invalid_query_key_shape() -> None:
    rope = RotaryPositionEmbedding(head_dim=8)

    query = torch.randn(2, 4, 6, 8)
    key = torch.randn(2, 4, 5, 8)

    with pytest.raises(ValueError):
        rope(query, key)