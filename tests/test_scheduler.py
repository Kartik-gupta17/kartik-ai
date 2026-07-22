from ai.training.optimizer import build_optimizer
from ai.training.scheduler import build_scheduler
from ai.transformer.gpt import GPTModel
from configs.model_config import ModelConfig


def test_scheduler():

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

    scheduler = build_scheduler(
        optimizer,
        epochs=10,
    )

    assert scheduler is not None