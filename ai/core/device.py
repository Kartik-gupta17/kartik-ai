"""
Device management utilities for KartikAI.

This module detects the best available PyTorch device:
CUDA GPU, Apple MPS, or CPU.
"""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    """
    Return the best available computing device.

    Priority:
    1. NVIDIA CUDA GPU
    2. Apple MPS
    3. CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_device_info() -> dict[str, object]:
    """
    Return basic information about the selected device.
    """
    device = get_device()

    info: dict[str, object] = {
        "device": str(device),
        "device_type": device.type,
        "cuda_available": torch.cuda.is_available(),
        "pytorch_version": torch.__version__,
    }

    if device.type == "cuda":
        info.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
                "cuda_version": torch.version.cuda,
                "gpu_memory_gb": round(
                    torch.cuda.get_device_properties(0).total_memory
                    / (1024**3),
                    2,
                ),
            }
        )

    return info