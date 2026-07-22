import torch
from torch.utils.data import DataLoader

from ai.datasets import TextDataset
from ai.training import Trainer
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


def test_validation():

    model = build_model()

    optimizer = build_optimizer(model)

    dataset = TextDataset(
        list(range(100)),
        sequence_length=16,
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
    )

    trainer = Trainer(
        model,
        optimizer,
        "cpu",
    )

    loss = trainer.validate(loader)

    assert loss > 0