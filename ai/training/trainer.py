from __future__ import annotations

import torch
from torch.utils.data import DataLoader


class Trainer:
    """
    Basic GPT trainer.
    """

    def __init__(
        self,
        model,
        optimizer,
        device,
    ) -> None:

        self.model = model
        self.optimizer = optimizer
        self.device = device

        self.model.to(device)

    def train_epoch(
        self,
        dataloader: DataLoader,
    ) -> float:

        self.model.train()

        total_loss = 0.0

        for inputs, targets in dataloader:

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            _, loss = self.model(
                inputs,
                targets,
            )

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)