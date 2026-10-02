"""End-to-end gradient checks for GNOME Micro."""

import numpy as np

from app.micro_llm.transformer.config import (
    TransformerConfig,
)
from app.micro_llm.transformer.model import (
    TransformerLanguageModel,
)


def relative_error(a: float, b: float) -> float:
    return abs(a - b) / max(
        1e-8,
        abs(a) + abs(b),
    )


def test_end_to_end_lm_head_gradient() -> None:
    config = TransformerConfig(
        d_model=8,
        num_heads=2,
        d_ff=12,
        context_size=4,
    )

    model = TransformerLanguageModel(config)

    # Finite differences need precision.
    model.lm_head = model.lm_head.astype(np.float64)
    model.lm_bias = model.lm_bias.astype(np.float64)

    inputs = np.array(
        [[71, 78, 79, 77]],
        dtype=np.int64,
    )

    targets = np.array(
        [[78, 79, 77, 69]],
        dtype=np.int64,
    )

    _, cache = model.forward_loss(
        inputs,
        targets,
    )

    gradients = model.backward(cache)

    epsilon = 1e-5
    position = (2, 69)

    original = model.lm_head[position]

    model.lm_head[position] = original + epsilon
    plus = model.loss(inputs, targets)

    model.lm_head[position] = original - epsilon
    minus = model.loss(inputs, targets)

    model.lm_head[position] = original

    numerical = (
        plus - minus
    ) / (2.0 * epsilon)

    analytical = gradients[
        "lm_head"
    ][position]

    assert relative_error(
        analytical,
        numerical,
    ) < 1e-4


def test_backward_reaches_embeddings_and_attention() -> None:
    config = TransformerConfig(
        d_model=8,
        num_heads=2,
        d_ff=12,
        context_size=4,
    )

    model = TransformerLanguageModel(config)

    inputs = np.array(
        [[71, 78, 79, 77]],
        dtype=np.int64,
    )

    targets = np.array(
        [[78, 79, 77, 69]],
        dtype=np.int64,
    )

    loss, cache = model.forward_loss(
        inputs,
        targets,
    )

    gradients = model.backward(cache)

    assert np.isfinite(loss)

    assert np.any(
        gradients["embeddings.token"] != 0
    )

    assert np.any(
        gradients["embeddings.position"] != 0
    )

    assert np.any(
        gradients["blocks.0.wq"] != 0
    )

    assert np.any(
        gradients["blocks.0.wk"] != 0
    )

    assert np.any(
        gradients["blocks.0.wv"] != 0
    )

    assert np.any(
        gradients["blocks.0.w1"] != 0
    )

    assert np.any(
        gradients["lm_head"] != 0
    )
