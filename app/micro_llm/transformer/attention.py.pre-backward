"""Causal self-attention primitives for GNOME Micro."""

import numpy as np


def softmax(
    values: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    shifted = values - np.max(
        values,
        axis=axis,
        keepdims=True,
    )

    exp = np.exp(shifted)

    return exp / np.sum(
        exp,
        axis=axis,
        keepdims=True,
    )


def causal_mask(length: int) -> np.ndarray:
    """Mask future positions from attention."""

    if length < 1:
        raise ValueError(
            "length must be positive."
        )

    return np.triu(
        np.ones(
            (length, length),
            dtype=bool,
        ),
        k=1,
    )


def scaled_dot_product_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform causal scaled dot-product attention."""

    if query.shape != key.shape:
        raise ValueError(
            "query and key shapes must match."
        )

    if key.shape != value.shape:
        raise ValueError(
            "key and value shapes must match."
        )

    depth = query.shape[-1]

    scores = (
        query @ np.swapaxes(key, -1, -2)
    ) / np.sqrt(depth)

    length = query.shape[-2]

    mask = causal_mask(length)

    scores = np.where(
        mask,
        -1e9,
        scores,
    )

    weights = softmax(
        scores,
        axis=-1,
    )

    output = weights @ value

    return output, weights
