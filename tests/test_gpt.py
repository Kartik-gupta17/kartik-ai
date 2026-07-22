import torch

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


def test_gpt_output_shape():

    model = build_model()

    x = torch.randint(
        0,
        100,
        (2, 16),
    )

    logits = model(x)

    assert logits.shape == (2, 16, 100)


def test_parameter_count():

    model = build_model()

    assert model.num_parameters > 0


def test_weight_tying():

    model = build_model()

    assert (
        model.token_embedding.weight.data_ptr()
        ==
        model.lm_head.weight.data_ptr()
    )
def test_gpt_training_forward():

    model = build_model()

    x = torch.randint(
        0,
        100,
        (2, 16),
    )

    targets = torch.randint(
        0,
        100,
        (2, 16),
    )

    logits, loss = model(
        x,
        targets,
    )

    assert logits.shape == (
        2,
        16,
        100,
    )

    assert loss.ndim == 0

    assert torch.isfinite(loss)
def test_generate():

    model = build_model()

    x = torch.randint(
        0,
        100,
        (1, 5),
    )

    output = model.generate(
        x,
        max_new_tokens=10,
    )

    assert output.shape == (
        1,
        15,
    )
def test_generate_temperature():

    model = build_model()

    x = torch.randint(
        0,
        100,
        (1, 5),
    )

    output = model.generate(
        input_ids=x,
        max_new_tokens=5,
        temperature=0.7,
    )

    assert output.shape == (1, 10)


def test_generate_top_k():

    model = build_model()

    x = torch.randint(
        0,
        100,
        (1, 5),
    )

    output = model.generate(
        input_ids=x,
        max_new_tokens=5,
        top_k=10,
    )

    assert output.shape == (1, 10)


def test_generate_greedy_mode():

    model = build_model()

    x = torch.randint(
        0,
        100,
        (1, 5),
    )

    output = model.generate(
        input_ids=x,
        max_new_tokens=5,
        do_sample=False,
    )

    assert output.shape == (1, 10)