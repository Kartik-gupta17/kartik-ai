import torch

from ai.nn import LayerNorm


def main():

    layer = LayerNorm(8)

    x = torch.randn(2, 4, 8)

    y = layer(x)

    print("Input Shape :", x.shape)
    print("Output Shape:", y.shape)

    print()

    print(y)

    print()

    print("Mean:")
    print(y.mean(dim=-1))

    print()

    print("Variance:")
    print(y.var(dim=-1, unbiased=False))


if __name__ == "__main__":
    main()