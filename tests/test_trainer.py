import torch
from torch.utils.data import DataLoader

from ai.datasets import TextDataset
from ai.training import Trainer

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


def test_trainer():

    model = build_model()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

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

    loss = trainer.train_epoch(loader)

    assert loss > 0