from __future__ import annotations

from .base_tokenizer import BaseTokenizer
from .vocabulary import Vocabulary


class CharacterTokenizer(BaseTokenizer):

    def __init__(self):

        self.vocab = Vocabulary()

    def fit(self, text: str):

        for ch in sorted(set(text)):
            self.vocab.add_token(ch)

    def encode(self, text: str):

        ids = []

        for ch in text:

            ids.append(
                self.vocab.token_to_id.get(
                    ch,
                    self.vocab.token_to_id[
                        Vocabulary.UNK
                    ],
                )
            )

        return ids

    def decode(self, token_ids):

        text = ""

        for idx in token_ids:

            token = self.vocab.id_to_token.get(
                idx,
                Vocabulary.UNK,
            )

            if token.startswith("<"):
                continue

            text += token

        return text

    def vocabulary_size(self):

        return len(self.vocab)