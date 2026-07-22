from pathlib import Path

from ai.training import (
    save_checkpoint,
    load_checkpoint,
)
from ai.training.optimizer import build_optimizer
from ai.transformer.gpt import GPTModel
from configs.model_config import ModelConfig


def build_model():

    config = ModelConfig(
        vocab_size=100,
        embedding_dim=32,
        num_heads=4,
        num_layers=2,
        ffn_hidden_dim=128,
    )

    return GPTModel(config)


def test_checkpoint(tmp_path: Path):

    model = build_model()

    optimizer = build_optimizer(model)

    checkpoint_path = (
        tmp_path / "checkpoint.pt"
    )

    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=5,
    )

    loaded_epoch = load_checkpoint(
        checkpoint_path,
        model,
        optimizer,
    )

    assert loaded_epoch == 5