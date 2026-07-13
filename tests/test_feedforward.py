import torch

from ai.nn import FeedForward


def main():

    model = FeedForward(
        embedding_dim=8,
        hidden_dim=32,
    )

    x = torch.randn(
        2,
        5,
        8,
    )

    y = model(x)

    print("Input Shape :", x.shape)
    print("Output Shape:", y.shape)

    print()

    print(y)


if __name__ == "__main__":
    main()