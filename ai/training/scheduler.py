from __future__ import annotations

from torch.optim.lr_scheduler import CosineAnnealingLR


def build_scheduler(
    optimizer,
    epochs: int,
):
    """
    Cosine learning rate scheduler.
    """

    return CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )