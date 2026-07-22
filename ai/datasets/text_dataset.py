from __future__ import annotations

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """
    Character-level language modeling dataset.
    """

    def __init__(
        self,
        token_ids: list[int],
        sequence_length: int,
    ) -> None:

        if sequence_length <= 0:
            raise ValueError(
                "sequence_length must be greater than zero."
            )

        if len(token_ids) <= sequence_length:
            raise ValueError(
                "Not enough tokens."
            )

        self.token_ids = token_ids
        self.sequence_length = sequence_length

    def __len__(self):

        return len(self.token_ids) - self.sequence_length

    def __getitem__(self, index):

        x = self.token_ids[
            index:index + self.sequence_length
        ]

        y = self.token_ids[
            index + 1:index + self.sequence_length + 1
        ]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )