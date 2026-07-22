from __future__ import annotations


class Vocabulary:
    """
    Character vocabulary.
    """

    PAD = "<PAD>"
    UNK = "<UNK>"
    BOS = "<BOS>"
    EOS = "<EOS>"

    def __init__(self) -> None:

        self.token_to_id = {
            self.PAD: 0,
            self.UNK: 1,
            self.BOS: 2,
            self.EOS: 3,
        }

        self.id_to_token = {
            0: self.PAD,
            1: self.UNK,
            2: self.BOS,
            3: self.EOS,
        }

    def add_token(self, token: str):

        if token not in self.token_to_id:

            index = len(self.token_to_id)

            self.token_to_id[token] = index

            self.id_to_token[index] = token

    def __len__(self):

        return len(self.token_to_id)