from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from ai.nn import Dropout, Embedding, LayerNorm
from configs.model_config import ModelConfig
from .transformer_block import TransformerBlock


class GPTModel(nn.Module):
    """
    Decoder-only GPT Language Model.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        self.token_embedding = Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
        )

        self.embedding_dropout = Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=config.embedding_dim,
                    num_heads=config.num_heads,
                    ffn_hidden_dim=config.ffn_hidden_dim,
                    dropout=config.dropout,
                    bias=config.bias,
                    rope_base=config.rope_base,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_norm = LayerNorm(config.embedding_dim)

        self.lm_head = nn.Linear(
            config.embedding_dim,
            config.vocab_size,
            bias=False,
        )

        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids: Tensor) -> Tensor:

        x = self.token_embedding(input_ids)

        x = self.embedding_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits

    @property
    def num_parameters(self):

        return sum(
            p.numel()
            for p in self.parameters()
        )