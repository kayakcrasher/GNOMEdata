"""Tests for GNOME Micro embedding backprop."""

import numpy as np

from app.micro_llm.transformer.config import (
    TransformerConfig,
)
from app.micro_llm.transformer.embeddings import (
    TransformerEmbeddings,
)


def test_embedding_backward_shapes() -> None:
    config = TransformerConfig(
        d_model=8,
        context_size=4,
    )

    rng = np.random.default_rng(42)

    layer = TransformerEmbeddings(
        config,
        rng,
    )

    tokens = np.array(
        [[71, 78, 79, 77]],
        dtype=np.int64,
    )

    output, cache = layer.forward(
        tokens,
        return_cache=True,
    )

    gradients = layer.backward(
        np.ones_like(output),
        cache,
    )

    assert gradients["token"].shape == layer.token.shape
    assert gradients["position"].shape == layer.position.shape


def test_repeated_token_gradients_accumulate() -> None:
    config = TransformerConfig(
        d_model=4,
        context_size=4,
    )

    rng = np.random.default_rng(42)

    layer = TransformerEmbeddings(
        config,
        rng,
    )

    # Byte 65 appears three times.
    tokens = np.array(
        [[65, 65, 66, 65]],
        dtype=np.int64,
    )

    output, cache = layer.forward(
        tokens,
        return_cache=True,
    )

    upstream = np.ones_like(output)

    gradients = layer.backward(
        upstream,
        cache,
    )

    assert np.allclose(
        gradients["token"][65],
        3.0,
    )

    assert np.allclose(
        gradients["token"][66],
        1.0,
    )

    assert np.allclose(
        gradients["position"][:4],
        1.0,
    )


def test_unused_tokens_receive_zero_gradient() -> None:
    config = TransformerConfig(
        d_model=4,
        context_size=4,
    )

    rng = np.random.default_rng(42)

    layer = TransformerEmbeddings(
        config,
        rng,
    )

    tokens = np.array(
        [[65, 66]],
        dtype=np.int64,
    )

    output, cache = layer.forward(
        tokens,
        return_cache=True,
    )

    gradients = layer.backward(
        np.ones_like(output),
        cache,
    )

    assert np.allclose(
        gradients["token"][200],
        0.0,
    )
