from __future__ import annotations

from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader


class Trainer:
    """
    Basic trainer for GPT-style language models.

    Features:
    - Model training
    - Optimizer handling
    - Optional learning-rate scheduler
    - Gradient clipping
    - Average epoch loss calculation
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        device: str | torch.device,
        scheduler: Optional[LRScheduler] = None,
        max_grad_norm: float = 1.0,
    ) -> None:
        """
        Initialize the trainer.

        Args:
            model:
                GPT model that will be trained.

            optimizer:
                PyTorch optimizer, such as AdamW.

            device:
                Training device, for example:
                "cpu"
                "cuda"
                torch.device("cpu")

            scheduler:
                Optional learning-rate scheduler.

            max_grad_norm:
                Maximum gradient norm used for gradient clipping.
        """

        if max_grad_norm <= 0:
            raise ValueError(
                "max_grad_norm must be greater than zero."
            )

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.max_grad_norm = max_grad_norm

        self.model.to(self.device)

    def train_epoch(
        self,
        dataloader: DataLoader,
    ) -> float:
        """
        Train the model for one complete epoch.

        Args:
            dataloader:
                DataLoader containing input and target batches.

        Returns:
            Average training loss for the epoch.
        """

        if len(dataloader) == 0:
            raise ValueError(
                "The dataloader is empty."
            )

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

            if loss is None:
                raise RuntimeError(
                    "Model did not return a training loss."
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.max_grad_norm,
            )

            self.optimizer.step()

            total_loss += loss.item()

        if self.scheduler is not None:
            self.scheduler.step()

        average_loss = total_loss / len(dataloader)

        return average_loss