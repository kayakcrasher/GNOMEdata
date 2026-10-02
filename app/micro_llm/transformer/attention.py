"""Causal self-attention with explicit NumPy backprop."""

from dataclasses import dataclass

import numpy as np


@dataclass
class AttentionCache:
    """Values required for attention backward."""

    query: np.ndarray
    key: np.ndarray
    value: np.ndarray
    weights: np.ndarray
    scale: float


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
    """Return True where future attention is forbidden."""

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
    return_cache: bool = False,
):
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

    scale = 1.0 / np.sqrt(depth)

    scores = (
        query
        @ np.swapaxes(
            key,
            -1,
            -2,
        )
    ) * scale

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

    if not return_cache:
        return output, weights

    cache = AttentionCache(
        query=query,
        key=key,
        value=value,
        weights=weights,
        scale=scale,
    )

    return output, weights, cache


def scaled_dot_product_attention_backward(
    grad_output: np.ndarray,
    cache: AttentionCache,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Backpropagate through causal attention."""

    query = cache.query
    key = cache.key
    value = cache.value
    weights = cache.weights
    scale = cache.scale

    # output = weights @ value
    grad_weights = (
        grad_output
        @ np.swapaxes(
            value,
            -1,
            -2,
        )
    )

    grad_value = (
        np.swapaxes(
            weights,
            -1,
            -2,
        )
        @ grad_output
    )

    # Softmax Jacobian-vector product:
    #
    # dS = P * (dP - sum(dP * P))
    correction = np.sum(
        grad_weights * weights,
        axis=-1,
        keepdims=True,
    )

    grad_scores = (
        weights
        * (
            grad_weights
            - correction
        )
    )

    # Masked probabilities are exactly zero, so their
    # score gradients should remain zero as well.
    length = query.shape[-2]

    mask = causal_mask(length)

    grad_scores = np.where(
        mask,
        0.0,
        grad_scores,
    )

    # scores = (query @ key.T) * scale
    grad_query = (
        grad_scores
        @ key
    ) * scale

    grad_key = (
        np.swapaxes(
            grad_scores,
            -1,
            -2,
        )
        @ query
    ) * scale

    return (
        grad_query,
        grad_key,
        grad_value,
    )
