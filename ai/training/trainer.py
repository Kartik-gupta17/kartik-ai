from __future__ import annotations

from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """
    Professional Trainer for GPT models.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        device: str | torch.device,
        scheduler: Optional[LRScheduler] = None,
        max_grad_norm: float = 1.0,
    ) -> None:

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.max_grad_norm = max_grad_norm

        self.train_losses = []
        self.validation_losses = []

        self.model.to(self.device)

    def train_epoch(
        self,
        dataloader: DataLoader,
    ) -> float:

        self.model.train()

        total_loss = 0.0

        progress = tqdm(
            dataloader,
            desc="Training",
            leave=False,
        )

        for inputs, targets in progress:

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            _, loss = self.model(
                inputs,
                targets,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm,
            )

            self.optimizer.step()

            total_loss += loss.item()

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.6f}",
            )

        if self.scheduler is not None:
            self.scheduler.step()

        average_loss = total_loss / len(dataloader)

        self.train_losses.append(
            average_loss
        )

        return average_loss

    @torch.no_grad()
    def validate(
        self,
        dataloader: DataLoader,
    ) -> float:

        self.model.eval()

        total_loss = 0.0

        for inputs, targets in dataloader:

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            _, loss = self.model(
                inputs,
                targets,
            )

            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)

        self.validation_losses.append(
            average_loss
        )

        return average_loss