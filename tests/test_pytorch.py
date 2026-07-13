import torch

from ai.core import get_device, get_logger, set_seed


def main() -> None:
    logger = get_logger("KartikAI.PyTorchTest")

    set_seed(42)
    device = get_device()

    logger.info("PyTorch foundation test started.")
    logger.info("Selected device: %s", device)

    x = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]],
        device=device,
        requires_grad=True,
    )

    y = torch.tensor(
        [[5.0, 6.0], [7.0, 8.0]],
        device=device,
    )

    addition = x + y
    multiplication = x * y
    matrix_product = x @ y

    loss = matrix_product.mean()
    loss.backward()

    print("\nTensor x:")
    print(x)

    print("\nTensor y:")
    print(y)

    print("\nAddition:")
    print(addition)

    print("\nElement-wise multiplication:")
    print(multiplication)

    print("\nMatrix multiplication:")
    print(matrix_product)

    print("\nLoss:")
    print(loss.item())

    print("\nGradient of x:")
    print(x.grad)

    print("\nDevice:")
    print(device)

    logger.info("Autograd test passed.")
    logger.info("PyTorch foundation test completed.")


if __name__ == "__main__":
    main()