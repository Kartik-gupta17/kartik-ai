from dataclasses import dataclass


@dataclass
class ModelConfig:
    """
    Configuration for the GPT Model.
    """

    vocab_size: int

    embedding_dim: int

    num_heads: int

    num_layers: int

    ffn_hidden_dim: int

    dropout: float = 0.1

    bias: bool = True

    rope_base: float = 10000.0