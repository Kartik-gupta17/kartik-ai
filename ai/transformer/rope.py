import torch
from torch import Tensor, nn


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Query aur Key vectors me token position ki information add karta hai
    bina separate positional embedding add kiye.

    Expected input shape:
        (batch_size, num_heads, sequence_length, head_dim)
    """

    def __init__(
        self,
        head_dim: int,
        base: float = 10000.0,
    ) -> None:
        super().__init__()

        if head_dim <= 0:
            raise ValueError("head_dim must be greater than 0.")

        if head_dim % 2 != 0:
            raise ValueError("head_dim must be an even number.")

        if base <= 0:
            raise ValueError("base must be greater than 0.")

        self.head_dim = head_dim
        self.base = base

        inverse_frequency = 1.0 / (
            base
            ** (
                torch.arange(
                    0,
                    head_dim,
                    2,
                    dtype=torch.float32,
                )
                / head_dim
            )
        )

        self.register_buffer(
            "inverse_frequency",
            inverse_frequency,
            persistent=False,
        )

    def _build_cos_sin(
        self,
        sequence_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        positions = torch.arange(
            sequence_length,
            device=device,
            dtype=self.inverse_frequency.dtype,
        )

        frequencies = torch.outer(
            positions,
            self.inverse_frequency,
        )

        duplicated_frequencies = torch.cat(
            [frequencies, frequencies],
            dim=-1,
        )

        cosine = duplicated_frequencies.cos().to(dtype=dtype)
        sine = duplicated_frequencies.sin().to(dtype=dtype)

        cosine = cosine.unsqueeze(0).unsqueeze(0)
        sine = sine.unsqueeze(0).unsqueeze(0)

        return cosine, sine

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        first_half, second_half = x.chunk(2, dim=-1)

        return torch.cat(
            [-second_half, first_half],
            dim=-1,
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if query.ndim != 4 or key.ndim != 4:
            raise ValueError(
                "Query and key must have shape "
                "(batch_size, num_heads, sequence_length, head_dim)."
            )

        if query.shape != key.shape:
            raise ValueError(
                "Query and key must have identical shapes."
            )

        if query.shape[-1] != self.head_dim:
            raise ValueError(
                f"Expected head dimension {self.head_dim}, "
                f"but received {query.shape[-1]}."
            )

        sequence_length = query.shape[-2]

        cosine, sine = self._build_cos_sin(
            sequence_length=sequence_length,
            device=query.device,
            dtype=query.dtype,
        )

        rotated_query = (
            query * cosine
            + self._rotate_half(query) * sine
        )

        rotated_key = (
            key * cosine
            + self._rotate_half(key) * sine
        )

        return rotated_query, rotated_key