from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ai.nn import Dropout, Embedding, LayerNorm
from configs.model_config import ModelConfig
from .transformer_block import TransformerBlock


class GPTModel(nn.Module):
    """
    Decoder-only GPT language model.
    """

    def __init__(self, config: ModelConfig) -> None:
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

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
    ):
        """
        Inference:
            logits = model(input_ids)

        Training:
            logits, loss = model(input_ids, targets)
        """

        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape "
                "(batch_size, sequence_length)."
            )

        if targets is not None:
            if targets.ndim != 2:
                raise ValueError(
                    "targets must have shape "
                    "(batch_size, sequence_length)."
                )

            if targets.shape != input_ids.shape:
                raise ValueError(
                    "targets and input_ids must have the same shape."
                )

        x = self.token_embedding(input_ids)
        x = self.embedding_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits

        loss = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            targets.reshape(-1),
        )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
    ) -> Tensor:
        """
        Greedy token generation.
        """

        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape "
                "(batch_size, sequence_length)."
            )

        if max_new_tokens < 0:
            raise ValueError(
                "max_new_tokens must be zero or greater."
            )

        self.eval()

        generated_ids = input_ids

        for _ in range(max_new_tokens):
            logits = self(generated_ids)

            # Last position ke vocabulary logits
            next_token_logits = logits[:, -1, :]

            # Highest-logit token select karo
            next_token = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True,
            )

            generated_ids = torch.cat(
                (generated_ids, next_token),
                dim=1,
            )

        return generated_ids

    @property
    def num_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
        )