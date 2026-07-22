from __future__ import annotations

import torch


def build_optimizer(
    model,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.95),
):
    """
    Build AdamW optimizer.
    """

    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=betas,
        weight_decay=weight_decay,
    )