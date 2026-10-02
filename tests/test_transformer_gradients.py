"""Numerical gradient checks for GNOME Micro."""

import numpy as np

from app.micro_llm.transformer.norm import (
    layer_norm_backward,
    layer_norm_forward,
)


def relative_error(
    actual: float,
    expected: float,
) -> float:
    denominator = max(
        1e-8,
        abs(actual) + abs(expected),
    )

    return abs(
        actual - expected
    ) / denominator


def test_layer_norm_backward_matches_numerical_gradient() -> None:
    rng = np.random.default_rng(42)

    x = rng.normal(
        size=(2, 3, 4)
    ).astype(np.float64)

    gamma = rng.normal(
        size=4
    ).astype(np.float64)

    beta = rng.normal(
        size=4
    ).astype(np.float64)

    upstream = rng.normal(
        size=x.shape
    ).astype(np.float64)

    output, cache = layer_norm_forward(
        x,
        gamma,
        beta,
    )

    grad_x, grad_gamma, grad_beta = (
        layer_norm_backward(
            upstream,
            cache,
        )
    )

    epsilon = 1e-5

    # Check several x entries.
    positions = [
        (0, 0, 0),
        (0, 2, 3),
        (1, 1, 2),
    ]

    for position in positions:
        original = x[position]

        x[position] = original + epsilon
        plus, _ = layer_norm_forward(
            x,
            gamma,
            beta,
        )

        x[position] = original - epsilon
        minus, _ = layer_norm_forward(
            x,
            gamma,
            beta,
        )

        x[position] = original

        numerical = np.sum(
            (plus - minus)
            * upstream
        ) / (2.0 * epsilon)

        analytical = grad_x[position]

        assert relative_error(
            analytical,
            numerical,
        ) < 1e-5

    # Check gamma.
    for index in range(gamma.size):
        original = gamma[index]

        gamma[index] = original + epsilon
        plus, _ = layer_norm_forward(
            x,
            gamma,
            beta,
        )

        gamma[index] = original - epsilon
        minus, _ = layer_norm_forward(
            x,
            gamma,
            beta,
        )

        gamma[index] = original

        numerical = np.sum(
            (plus - minus)
            * upstream
        ) / (2.0 * epsilon)

        assert relative_error(
            grad_gamma[index],
            numerical,
        ) < 1e-5

    # beta derivative has a simple exact form.
    expected_beta = upstream.sum(
        axis=(0, 1)
    )

    assert np.allclose(
        grad_beta,
        expected_beta,
        atol=1e-10,
    )


from app.micro_llm.transformer.attention import (
    scaled_dot_product_attention,
    scaled_dot_product_attention_backward,
)


def test_attention_backward_matches_numerical_gradient() -> None:
    rng = np.random.default_rng(123)

    q = rng.normal(
        size=(1, 2, 3, 4)
    ).astype(np.float64)

    k = rng.normal(
        size=(1, 2, 3, 4)
    ).astype(np.float64)

    v = rng.normal(
        size=(1, 2, 3, 4)
    ).astype(np.float64)

    output, _, cache = (
        scaled_dot_product_attention(
            q,
            k,
            v,
            return_cache=True,
        )
    )

    upstream = rng.normal(
        size=output.shape
    ).astype(np.float64)

    dq, dk, dv = (
        scaled_dot_product_attention_backward(
            upstream,
            cache,
        )
    )

    epsilon = 1e-5

    checks = (
        ("query", q, dq),
        ("key", k, dk),
        ("value", v, dv),
    )

    positions = [
        (0, 0, 0, 0),
        (0, 0, 2, 3),
        (0, 1, 1, 2),
    ]

    for name, tensor, analytical in checks:
        for position in positions:
            original = tensor[position]

            tensor[position] = (
                original + epsilon
            )

            plus, _ = (
                scaled_dot_product_attention(
                    q,
                    k,
                    v,
                )
            )

            tensor[position] = (
                original - epsilon
            )

            minus, _ = (
                scaled_dot_product_attention(
                    q,
                    k,
                    v,
                )
            )

            tensor[position] = original

            numerical = np.sum(
                (plus - minus)
                * upstream
            ) / (2.0 * epsilon)

            error = relative_error(
                analytical[position],
                numerical,
            )

            assert error < 1e-5, (
                f"{name} gradient mismatch "
                f"at {position}: "
                f"analytical="
                f"{analytical[position]}, "
                f"numerical={numerical}, "
                f"error={error}"
            )


def test_masked_future_attention_has_no_gradient() -> None:
    rng = np.random.default_rng(321)

    q = rng.normal(
        size=(1, 1, 4, 3)
    ).astype(np.float64)

    k = rng.normal(
        size=(1, 1, 4, 3)
    ).astype(np.float64)

    v = rng.normal(
        size=(1, 1, 4, 3)
    ).astype(np.float64)

    output, weights, cache = (
        scaled_dot_product_attention(
            q,
            k,
            v,
            return_cache=True,
        )
    )

    assert np.allclose(
        np.triu(
            weights[0, 0],
            k=1,
        ),
        0.0,
        atol=1e-12,
    )

    upstream = np.ones_like(output)

    dq, dk, dv = (
        scaled_dot_product_attention_backward(
            upstream,
            cache,
        )
    )

    assert np.all(np.isfinite(dq))
    assert np.all(np.isfinite(dk))
    assert np.all(np.isfinite(dv))
