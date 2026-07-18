import torch

from ai.transformer import TransformerBlock


def test_transformer_block_output_shape():
    block = TransformerBlock(
        embedding_dim=32,
        num_heads=4,
        ffn_hidden_dim=128,
    )

    x = torch.randn(2, 10, 32)

    y = block(x)

    assert y.shape == x.shape


def test_backward():
    block = TransformerBlock(
        embedding_dim=32,
        num_heads=4,
        ffn_hidden_dim=128,
    )

    x = torch.randn(
        2,
        10,
        32,
        requires_grad=True,
    )

    y = block(x)

    loss = y.mean()

    loss.backward()

    assert x.grad is not None