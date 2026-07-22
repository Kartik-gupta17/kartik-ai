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

    Features:
    - Token embedding
    - Multiple Transformer blocks
    - Rotary Position Embedding through attention
    - Final LayerNorm
    - Weight-tied language-model head
    - Training loss calculation
    - Greedy generation
    - Temperature sampling
    - Top-k sampling
    - EOS token handling
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

        # Token embedding aur output projection same weights use karte hain.
        self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Forward pass.

        Inference:
            logits = model(input_ids)

        Training:
            logits, loss = model(input_ids, targets)
        """

        self._validate_input_ids(input_ids)

        if targets is not None:
            self._validate_targets(
                input_ids=input_ids,
                targets=targets,
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
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
        do_sample: bool = True,
    ) -> Tensor:
        """
        Autoregressive token generation.

        Args:
            input_ids:
                Starting token IDs with shape:
                (batch_size, sequence_length)

            max_new_tokens:
                Maximum number of new tokens to generate.

            temperature:
                Logits scaling value.
                Lower value gives more focused output.
                Higher value gives more random output.

            top_k:
                Only the top-k probable tokens are retained.
                None means top-k filtering is disabled.

            eos_token_id:
                Generation stops when every sequence in the batch
                generates this token.

            do_sample:
                True:
                    multinomial sampling is used.

                False:
                    greedy decoding is used.

        Returns:
            Generated token IDs including the original prompt.
        """

        self._validate_input_ids(input_ids)

        if max_new_tokens < 0:
            raise ValueError(
                "max_new_tokens must be zero or greater."
            )

        if temperature <= 0:
            raise ValueError(
                "temperature must be greater than 0."
            )

        if top_k is not None and top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        if eos_token_id is not None:
            if not 0 <= eos_token_id < self.config.vocab_size:
                raise ValueError(
                    "eos_token_id must be inside the vocabulary range."
                )

        was_training = self.training
        self.eval()

        generated_ids = input_ids

        try:
            for _ in range(max_new_tokens):
                logits = self(generated_ids)

                # Sirf last position ke logits chahiye.
                next_token_logits = logits[:, -1, :]

                if do_sample:
                    next_token_logits = (
                        next_token_logits / temperature
                    )

                    if top_k is not None:
                        effective_top_k = min(
                            top_k,
                            next_token_logits.size(-1),
                        )

                        top_values, _ = torch.topk(
                            next_token_logits,
                            effective_top_k,
                            dim=-1,
                        )

                        threshold = top_values[:, -1].unsqueeze(-1)

                        next_token_logits = (
                            next_token_logits.masked_fill(
                                next_token_logits < threshold,
                                float("-inf"),
                            )
                        )

                    probabilities = torch.softmax(
                        next_token_logits,
                        dim=-1,
                    )

                    next_token = torch.multinomial(
                        probabilities,
                        num_samples=1,
                    )

                else:
                    next_token = torch.argmax(
                        next_token_logits,
                        dim=-1,
                        keepdim=True,
                    )

                generated_ids = torch.cat(
                    (generated_ids, next_token),
                    dim=1,
                )

                if eos_token_id is not None:
                    all_sequences_finished = (
                        next_token.squeeze(-1)
                        == eos_token_id
                    ).all()

                    if bool(all_sequences_finished):
                        break

        finally:
            if was_training:
                self.train()

        return generated_ids

    def _validate_input_ids(
        self,
        input_ids: Tensor,
    ) -> None:
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape "
                "(batch_size, sequence_length)."
            )

        if input_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise TypeError(
                "input_ids must contain integer token IDs."
            )

        if input_ids.numel() == 0:
            raise ValueError(
                "input_ids cannot be empty."
            )

        minimum_token_id = int(input_ids.min().item())
        maximum_token_id = int(input_ids.max().item())

        if minimum_token_id < 0:
            raise ValueError(
                "input_ids cannot contain negative token IDs."
            )

        if maximum_token_id >= self.config.vocab_size:
            raise ValueError(
                "input_ids contain token IDs outside "
                "the vocabulary range."
            )

    def _validate_targets(
        self,
        input_ids: Tensor,
        targets: Tensor,
    ) -> None:
        if targets.ndim != 2:
            raise ValueError(
                "targets must have shape "
                "(batch_size, sequence_length)."
            )

        if targets.shape != input_ids.shape:
            raise ValueError(
                "targets and input_ids must have the same shape."
            )

        if targets.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise TypeError(
                "targets must contain integer token IDs."
            )

        minimum_target_id = int(targets.min().item())
        maximum_target_id = int(targets.max().item())

        if minimum_target_id < 0:
            raise ValueError(
                "targets cannot contain negative token IDs."
            )

        if maximum_target_id >= self.config.vocab_size:
            raise ValueError(
                "targets contain token IDs outside "
                "the vocabulary range."
            )

    @property
    def num_parameters(self) -> int:
        """
        Total number of unique trainable and non-trainable parameters.
        """

        return sum(
            parameter.numel()
            for parameter in self.parameters()
        )

    @property
    def trainable_parameters(self) -> int:
        """
        Total number of parameters requiring gradients.
        """

        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )