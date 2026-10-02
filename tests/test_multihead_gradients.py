"""Gradient checks for multi-head attention."""

import numpy as np

from app.micro_llm.transformer.config import (
    TransformerConfig,
)
from app.micro_llm.transformer.multihead import (
    MultiHeadAttention,
)


def relative_error(a: float, b: float) -> float:
    return abs(a - b) / max(
        1e-8,
        abs(a) + abs(b),
    )


def objective(
    layer: MultiHeadAttention,
    x: np.ndarray,
    upstream: np.ndarray,
) -> float:
    output, _ = layer.forward(x)

    return float(
        np.sum(output * upstream)
    )


def test_multihead_backward_matches_numerical_gradient() -> None:
    config = TransformerConfig(
        d_model=8,
        num_heads=2,
        d_ff=16,
        context_size=4,
    )

    rng = np.random.default_rng(42)

    layer = MultiHeadAttention(
        config,
        rng,
    )

    x = rng.normal(
        size=(1, 4, 8)
    ).astype(np.float64)

    # Use float64 weights for accurate finite differences.
    layer.wq = layer.wq.astype(np.float64)
    layer.wk = layer.wk.astype(np.float64)
    layer.wv = layer.wv.astype(np.float64)
    layer.wo = layer.wo.astype(np.float64)

    output, cache = layer.forward(x)

    upstream = rng.normal(
        size=output.shape
    ).astype(np.float64)

    grad_x, grads = layer.backward(
        upstream,
        cache,
    )

    epsilon = 1e-5

    # Check an input gradient.
    position = (0, 2, 3)
    original = x[position]

    x[position] = original + epsilon
    plus = objective(layer, x, upstream)

    x[position] = original - epsilon
    minus = objective(layer, x, upstream)

    x[position] = original

    numerical = (
        plus - minus
    ) / (2.0 * epsilon)

    assert relative_error(
        grad_x[position],
        numerical,
    ) < 1e-5

    # Check each projection matrix.
    matrices = (
        ("wq", layer.wq),
        ("wk", layer.wk),
        ("wv", layer.wv),
        ("wo", layer.wo),
    )

    for name, matrix in matrices:
        position = (2, 5)
        original = matrix[position]

        matrix[position] = original + epsilon
        plus = objective(
            layer,
            x,
            upstream,
        )

        matrix[position] = original - epsilon
        minus = objective(
            layer,
            x,
            upstream,
        )

        matrix[position] = original

        numerical = (
            plus - minus
        ) / (2.0 * epsilon)

        analytical = grads[name][position]

        error = relative_error(
            analytical,
            numerical,
        )

        assert error < 1e-5, (
            f"{name} mismatch: "
            f"analytical={analytical}, "
            f"numerical={numerical}, "
            f"error={error}"
        )


def test_multihead_gradient_shapes() -> None:
    config = TransformerConfig(
        d_model=8,
        num_heads=2,
        d_ff=16,
        context_size=4,
    )

    rng = np.random.default_rng(7)

    layer = MultiHeadAttention(
        config,
        rng,
    )

    x = rng.normal(
        size=(2, 4, 8)
    ).astype(np.float32)

    output, cache = layer.forward(x)

    grad_x, grads = layer.backward(
        np.ones_like(output),
        cache,
    )

    assert grad_x.shape == x.shape

    assert grads["wq"].shape == layer.wq.shape
    assert grads["wk"].shape == layer.wk.shape
    assert grads["wv"].shape == layer.wv.shape
    assert grads["wo"].shape == layer.wo.shape
