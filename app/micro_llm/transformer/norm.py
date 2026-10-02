"""Layer normalization with explicit NumPy backprop."""

from dataclasses import dataclass

import numpy as np


@dataclass
class LayerNormCache:
    """Values required for LayerNorm backward."""

    normalized: np.ndarray
    inv_std: np.ndarray
    gamma: np.ndarray


def layer_norm_forward(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    epsilon: float = 1e-5,
) -> tuple[np.ndarray, LayerNormCache]:
    """Normalize the final dimension of x."""

    mean = x.mean(
        axis=-1,
        keepdims=True,
    )

    centered = x - mean

    variance = np.mean(
        centered * centered,
        axis=-1,
        keepdims=True,
    )

    inv_std = 1.0 / np.sqrt(
        variance + epsilon
    )

    normalized = centered * inv_std

    output = (
        normalized * gamma
        + beta
    )

    cache = LayerNormCache(
        normalized=normalized,
        inv_std=inv_std,
        gamma=gamma,
    )

    return output, cache


def layer_norm_backward(
    grad_output: np.ndarray,
    cache: LayerNormCache,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Backpropagate through LayerNorm."""

    normalized = cache.normalized
    inv_std = cache.inv_std
    gamma = cache.gamma

    feature_count = grad_output.shape[-1]

    grad_normalized = (
        grad_output * gamma
    )

    sum_grad = np.sum(
        grad_normalized,
        axis=-1,
        keepdims=True,
    )

    sum_grad_normalized = np.sum(
        grad_normalized * normalized,
        axis=-1,
        keepdims=True,
    )

    grad_x = (
        inv_std
        / feature_count
        * (
            feature_count * grad_normalized
            - sum_grad
            - normalized
            * sum_grad_normalized
        )
    )

    reduction_axes = tuple(
        range(grad_output.ndim - 1)
    )

    grad_gamma = np.sum(
        grad_output * normalized,
        axis=reduction_axes,
    )

    grad_beta = np.sum(
        grad_output,
        axis=reduction_axes,
    )

    return (
        grad_x,
        grad_gamma,
        grad_beta,
    )
