"""
Global model configuration for KartikAI.
"""

from dataclasses import dataclass


@dataclass(slots=True)
class ModelConfig:
    # Tokenizer
    vocab_size: int = 32000
    max_sequence_length: int = 2048

    # Transformer
    embedding_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    ffn_dim: int = 3072
    dropout: float = 0.1

    # Training
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 10

    # Generation
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95

    # Misc
    seed: int = 42


config = ModelConfig()