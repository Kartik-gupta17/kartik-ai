import torch

from ai.nn import Embedding


def main() -> None:
    vocabulary_size = 10
    embedding_dimension = 8

    layer = Embedding(
        num_embeddings=vocabulary_size,
        embedding_dim=embedding_dimension,
        padding_idx=0,
    )

    token_ids = torch.tensor(
        [
            [1, 2, 3, 0],
            [4, 5, 6, 0],
        ],
        dtype=torch.long,
    )

    output = layer(token_ids)

    print("Token IDs:")
    print(token_ids)

    print("\nInput shape:")
    print(token_ids.shape)

    print("\nEmbedding output shape:")
    print(output.shape)

    print("\nEmbedding output:")
    print(output)

    print("\nPadding vector:")
    print(output[0, 3])

    assert output.shape == (2, 4, 8)
    assert torch.all(output[:, 3] == 0)

    print("\nEmbedding test passed.")


if __name__ == "__main__":
    main()