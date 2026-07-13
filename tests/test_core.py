"""
Manual test for KartikAI core utilities.
"""

import random

import numpy as np
import torch

from ai.core import get_device, get_device_info, get_logger, set_seed


def main() -> None:
    logger = get_logger("KartikAI.CoreTest")

    logger.info("Starting core engine test.")

    device = get_device()
    info = get_device_info()

    print("\nSelected device:", device)
    print("\nDevice information:")

    for key, value in info.items():
        print(f"{key}: {value}")

    set_seed(42)

    python_value_1 = random.random()
    numpy_value_1 = np.random.rand()
    torch_value_1 = torch.rand(1).item()

    set_seed(42)

    python_value_2 = random.random()
    numpy_value_2 = np.random.rand()
    torch_value_2 = torch.rand(1).item()

    assert python_value_1 == python_value_2
    assert numpy_value_1 == numpy_value_2
    assert torch_value_1 == torch_value_2

    tensor = torch.tensor([1.0, 2.0, 3.0], device=device)

    print("\nTensor:", tensor)
    print("Tensor device:", tensor.device)

    logger.info("Seed test passed.")
    logger.info("Device test passed.")
    logger.info("Core engine test completed successfully.")


if __name__ == "__main__":
    main()