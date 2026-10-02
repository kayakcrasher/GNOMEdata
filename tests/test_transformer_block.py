"""Tests for GNOME Micro Transformer blocks."""

import numpy as np

from app.micro_llm.transformer.block import (
    TransformerBlock,
    layer_norm,
)
from app.micro_llm.transformer.config import (
    TransformerConfig,
)
from app.micro_llm.transformer.embeddings import (
    TransformerEmbeddings,
)


def test_layer_norm_shape_and_statistics() -> None:
    rng = np.random.default_rng(42)

    x = rng.normal(
        size=(2, 5, 64)
    ).astype(np.float32)

    gamma = np.ones(
        64,
        dtype=np.float32,
    )

    beta = np.zeros(
        64,
        dtype=np.float32,
    )

    output = layer_norm(
        x,
        gamma,
        beta,
    )

    assert output.shape == x.shape

    assert np.allclose(
        output.mean(axis=-1),
        0.0,
        atol=1e-5,
    )


def test_embeddings_preserve_sequence_shape() -> None:
    config = TransformerConfig()

    rng = np.random.default_rng(42)

    embeddings = TransformerEmbeddings(
        config,
        rng,
    )

    tokens = np.array(
        [
            [71, 78, 79, 77, 69],
            [82, 65, 78, 68, 89],
        ],
        dtype=np.int64,
    )

    output = embeddings.forward(tokens)

    assert output.shape == (
        2,
        5,
        config.d_model,
    )


def test_transformer_block_shape() -> None:
    config = TransformerConfig()

    rng = np.random.default_rng(42)

    block = TransformerBlock(
        config,
        rng,
    )

    x = rng.normal(
        size=(
            2,
            8,
            config.d_model,
        )
    ).astype(np.float32)

    output, weights = block.forward(x)

    assert output.shape == x.shape

    assert weights.shape == (
        2,
        config.num_heads,
        8,
        8,
    )


def test_each_attention_head_is_causal() -> None:
    config = TransformerConfig()

    rng = np.random.default_rng(42)

    block = TransformerBlock(
        config,
        rng,
    )

    x = rng.normal(
        size=(
            1,
            6,
            config.d_model,
        )
    ).astype(np.float32)

    _, weights = block.forward(x)

    for head in range(config.num_heads):
        for position in range(6):
            future = weights[
                0,
                head,
                position,
                position + 1:,
            ]

            assert np.allclose(
                future,
                0.0,
                atol=1e-7,
            )
