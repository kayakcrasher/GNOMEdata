"""Tests for the GNOMEdata MicroLLM."""

from app.micro_llm.model import MicroLanguageModel
from app.micro_llm.tokenizer import Tokenizer


def test_tokenizer_round_trip() -> None:
    tokenizer = Tokenizer.train(
        "gnomes mill data."
    )

    encoded = tokenizer.encode(
        "gnomes mill data."
    )

    assert tokenizer.decode(encoded) == (
        "gnomes mill data."
    )


def test_model_probabilities_sum_to_one() -> None:
    model = MicroLanguageModel(4)

    probabilities = model.probabilities(0)

    assert abs(sum(probabilities) - 1.0) < 1e-9


def test_training_reduces_loss() -> None:
    tokenizer = Tokenizer.train(
        "randy sells pies. "
        "randy sells pies. "
        "randy sells pies."
    )

    model = MicroLanguageModel(
        tokenizer.vocab_size
    )

    losses = model.train(
        tokenizer.encode(
            "randy sells pies. "
            "randy sells pies. "
            "randy sells pies."
        ),
        epochs=100,
    )

    assert losses[-1] < losses[0]


def test_model_generates_tokens() -> None:
    tokenizer = Tokenizer.train(
        "randy sells pies."
    )

    model = MicroLanguageModel(
        tokenizer.vocab_size
    )

    model.train(
        tokenizer.encode(
            "randy sells pies."
        ),
        epochs=50,
    )

    start = tokenizer.encode("randy")[0]

    generated = model.generate(
        start,
        length=8,
    )

    assert len(generated) == 8
    assert generated[0] == start
