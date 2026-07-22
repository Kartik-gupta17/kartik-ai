from __future__ import annotations

from abc import ABC, abstractmethod


class BaseTokenizer(ABC):

    @abstractmethod
    def encode(self, text: str):
        pass

    @abstractmethod
    def decode(self, token_ids):
        pass

    @abstractmethod
    def vocabulary_size(self):
        pass