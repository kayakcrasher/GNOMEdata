"""Tests for GNOME Micro v0.2."""

from app.micro_llm.byte_tokenizer import ByteTokenizer
from app.micro_llm.context_model import ContextLanguageModel


def test_byte_tokenizer_handles_arbitrary_utf8() -> None:
    tokenizer = ByteTokenizer()

    text = "GNOME 🧠 — café — 日本語"

    assert tokenizer.decode(
        tokenizer.encode(text)
    ) == text


def test_context_model_learns_sequence() -> None:
    tokenizer = ByteTokenizer()

    tokens = tokenizer.encode(
        "randy sells pies in oakville. "
        "randy sells pies in oakville."
    )

    model = ContextLanguageModel(
        context_size=10
    )

    model.train(tokens)

    prompt = tokenizer.encode(
        "randy sells "
    )

    generated = model.generate(
        prompt,
        max_new_tokens=5,
        seed=1,
    )

    text = tokenizer.decode(generated)

    assert text.startswith(
        "randy sells pies"
    )


def test_model_checkpoint_round_trip(tmp_path) -> None:
    tokenizer = ByteTokenizer()

    model = ContextLanguageModel(
        context_size=8
    )

    model.train(
        tokenizer.encode(
            "gnomes mill data locally."
        )
    )

    path = tmp_path / "micro.json"

    model.save(path)

    restored = ContextLanguageModel.load(path)

    assert restored.context_size == 8
    assert restored.counts == model.counts
