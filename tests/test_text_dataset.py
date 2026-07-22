import torch

from ai.datasets import TextDataset


def test_dataset_length():

    tokens = list(range(20))

    dataset = TextDataset(
        token_ids=tokens,
        sequence_length=5,
    )

    assert len(dataset) == 15


def test_dataset_item():

    tokens = list(range(20))

    dataset = TextDataset(
        token_ids=tokens,
        sequence_length=5,
    )

    x, y = dataset[0]

    assert torch.equal(
        x,
        torch.tensor([0,1,2,3,4])
    )

    assert torch.equal(
        y,
        torch.tensor([1,2,3,4,5])
    )


def test_shapes():

    tokens = list(range(100))

    dataset = TextDataset(
        token_ids=tokens,
        sequence_length=16,
    )

    x, y = dataset[10]

    assert x.shape == (16,)
    assert y.shape == (16,)