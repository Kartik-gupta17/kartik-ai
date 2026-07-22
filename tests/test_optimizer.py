import torch

from ai.training.optimizer import build_optimizer
from ai.transformer.gpt import GPTModel
from configs.model_config import ModelConfig


def test_optimizer():

    model = GPTModel(
        ModelConfig(
            vocab_size=100,
            embedding_dim=32,
            num_heads=4,
            num_layers=2,
            ffn_hidden_dim=128,
        )
    )

    optimizer = build_optimizer(model)

    assert isinstance(
        optimizer,
        torch.optim.AdamW,
    )