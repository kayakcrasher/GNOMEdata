"""Tests for GNOME Micro Transformer language model."""

import numpy as np

from app.micro_llm.transformer.config import (
    TransformerConfig,
)
from app.micro_llm.transformer.model import (
    TransformerLanguageModel,
)


def test_transformer_language_model_shape() -> None:
    config = TransformerConfig()

    model = TransformerLanguageModel(config)

    tokens = np.array(
        [
            [71, 78, 79, 77, 69],
            [82, 65, 78, 68, 89],
        ],
        dtype=np.int64,
    )

    logits = model.forward(tokens)

    assert logits.shape == (
        2,
        5,
        config.vocab_size,
    )


def test_transformer_loss_is_finite() -> None:
    model = TransformerLanguageModel()

    inputs = np.array(
        [[71, 78, 79, 77, 69]],
        dtype=np.int64,
    )

    targets = np.array(
        [[78, 79, 77, 69, 10]],
        dtype=np.int64,
    )

    loss = model.loss(
        inputs,
        targets,
    )

    assert np.isfinite(loss)
    assert loss > 0.0


def test_parameter_count() -> None:
    model = TransformerLanguageModel()

    assert model.parameter_count == 461056
