"""Numerical gradient checks for a complete Transformer block."""

import numpy as np

from app.micro_llm.transformer.block import (
    TransformerBlock,
)
from app.micro_llm.transformer.config import (
    TransformerConfig,
)


def relative_error(
    actual: float,
    expected: float,
) -> float:
    return abs(
        actual - expected
    ) / max(
        1e-8,
        abs(actual) + abs(expected),
    )


def objective(
    block: TransformerBlock,
    x: np.ndarray,
    upstream: np.ndarray,
) -> float:
    output, _ = block.forward(x)

    return float(
        np.sum(output * upstream)
    )


def test_full_block_backward_matches_numerical_gradient() -> None:
    config = TransformerConfig(
        d_model=8,
        num_heads=2,
        d_ff=12,
        context_size=4,
    )

    rng = np.random.default_rng(2026)

    block = TransformerBlock(
        config,
        rng,
    )

    # Float64 makes finite-difference checks much cleaner.
    arrays = [
        block.attention.wq,
        block.attention.wk,
        block.attention.wv,
        block.attention.wo,
        block.w1,
        block.b1,
        block.w2,
        block.b2,
        block.ln1_gamma,
        block.ln1_beta,
        block.ln2_gamma,
        block.ln2_beta,
    ]

    for array in arrays:
        # Assignment through slicing preserves references.
        if array.dtype != np.float64:
            converted = array.astype(np.float64)

            if array is block.attention.wq:
                block.attention.wq = converted
            elif array is block.attention.wk:
                block.attention.wk = converted
            elif array is block.attention.wv:
                block.attention.wv = converted
            elif array is block.attention.wo:
                block.attention.wo = converted
            elif array is block.w1:
                block.w1 = converted
            elif array is block.b1:
                block.b1 = converted
            elif array is block.w2:
                block.w2 = converted
            elif array is block.b2:
                block.b2 = converted
            elif array is block.ln1_gamma:
                block.ln1_gamma = converted
            elif array is block.ln1_beta:
                block.ln1_beta = converted
            elif array is block.ln2_gamma:
                block.ln2_gamma = converted
            elif array is block.ln2_beta:
                block.ln2_beta = converted

    x = rng.normal(
        size=(1, 4, 8)
    ).astype(np.float64)

    output, cache = block.forward(
        x,
        return_cache=True,
    )

    upstream = rng.normal(
        size=output.shape
    ).astype(np.float64)

    grad_x, grads = block.backward(
        upstream,
        cache,
    )

    epsilon = 1e-5

    # Check input gradient.
    position = (0, 2, 5)
    original = x[position]

    x[position] = original + epsilon
    plus = objective(
        block,
        x,
        upstream,
    )

    x[position] = original - epsilon
    minus = objective(
        block,
        x,
        upstream,
    )

    x[position] = original

    numerical = (
        plus - minus
    ) / (2.0 * epsilon)

    assert relative_error(
        grad_x[position],
        numerical,
    ) < 1e-5

    parameters = {
        "wq": block.attention.wq,
        "wk": block.attention.wk,
        "wv": block.attention.wv,
        "wo": block.attention.wo,
        "w1": block.w1,
        "b1": block.b1,
        "w2": block.w2,
        "b2": block.b2,
        "ln1_gamma": block.ln1_gamma,
        "ln1_beta": block.ln1_beta,
        "ln2_gamma": block.ln2_gamma,
        "ln2_beta": block.ln2_beta,
    }

    for name, parameter in parameters.items():
        index = tuple(
            min(1, size - 1)
            for size in parameter.shape
        )

        original = parameter[index]

        parameter[index] = (
            original + epsilon
        )

        plus = objective(
            block,
            x,
            upstream,
        )

        parameter[index] = (
            original - epsilon
        )

        minus = objective(
            block,
            x,
            upstream,
        )

        parameter[index] = original

        numerical = (
            plus - minus
        ) / (2.0 * epsilon)

        analytical = grads[name][index]

        error = relative_error(
            analytical,
            numerical,
        )

        assert error < 1e-5, (
            f"{name} gradient mismatch: "
            f"analytical={analytical}, "
            f"numerical={numerical}, "
            f"error={error}"
        )


def test_block_backward_shapes() -> None:
    config = TransformerConfig()

    rng = np.random.default_rng(42)

    block = TransformerBlock(
        config,
        rng,
    )

    x = rng.normal(
        size=(
            2,
            6,
            config.d_model,
        )
    ).astype(np.float32)

    output, cache = block.forward(
        x,
        return_cache=True,
    )

    grad_x, gradients = block.backward(
        np.ones_like(output),
        cache,
    )

    assert grad_x.shape == x.shape

    assert gradients["wq"].shape == block.wq.shape
    assert gradients["w1"].shape == block.w1.shape
    assert gradients["w2"].shape == block.w2.shape
    assert gradients["ln1_gamma"].shape == block.ln1_gamma.shape
    assert gradients["ln2_gamma"].shape == block.ln2_gamma.shape
