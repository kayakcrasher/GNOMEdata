"""Tests for GNOME Micro causal attention."""

import numpy as np

from app.micro_llm.transformer.attention import (
    causal_mask,
    scaled_dot_product_attention,
)


def test_causal_mask_blocks_future() -> None:
    mask = causal_mask(4)

    assert mask.tolist() == [
        [False, True, True, True],
        [False, False, True, True],
        [False, False, False, True],
        [False, False, False, False],
    ]


def test_attention_weights_sum_to_one() -> None:
    rng = np.random.default_rng(42)

    values = rng.normal(
        size=(2, 4, 8)
    ).astype(np.float32)

    _, weights = scaled_dot_product_attention(
        values,
        values,
        values,
    )

    totals = weights.sum(axis=-1)

    assert np.allclose(
        totals,
        1.0,
        atol=1e-6,
    )


def test_attention_cannot_see_future() -> None:
    rng = np.random.default_rng(42)

    values = rng.normal(
        size=(1, 5, 8)
    ).astype(np.float32)

    _, weights = scaled_dot_product_attention(
        values,
        values,
        values,
    )

    for row in range(5):
        assert np.allclose(
            weights[0, row, row + 1:],
            0.0,
            atol=1e-7,
        )
