"""Tests for GNOMEdata embedding machinery."""

import pytest

from app.rag.embeddings import (
    StaticEmbedder,
    build_embedding,
    validate_compatible,
)


def test_build_embedding() -> None:
    embedding = build_embedding(
        [0.1, 0.2, 0.3],
        "test-model",
    )

    assert embedding.vector == (
        0.1,
        0.2,
        0.3,
    )
    assert embedding.model == "test-model"
    assert embedding.dimensions == 3


def test_values_are_normalized_to_floats() -> None:
    embedding = build_embedding(
        [1, 2, 3],
        "test-model",
    )

    assert embedding.vector == (
        1.0,
        2.0,
        3.0,
    )


def test_empty_vector_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        build_embedding(
            [],
            "test-model",
        )


def test_empty_model_name_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="model name",
    ):
        build_embedding(
            [1.0, 2.0],
            " ",
        )


def test_static_embedder_returns_configured_vector() -> None:
    worker = StaticEmbedder({
        "pump maintenance": [
            0.9,
            0.8,
            0.1,
        ],
    })

    result = worker.embed(
        "pump maintenance"
    )

    assert result.vector == (
        0.9,
        0.8,
        0.1,
    )
    assert result.dimensions == 3
    assert result.model == worker.model_name


def test_static_embedder_rejects_empty_text() -> None:
    worker = StaticEmbedder({})

    with pytest.raises(
        ValueError,
        match="empty text",
    ):
        worker.embed("   ")


def test_static_embedder_rejects_unknown_text() -> None:
    worker = StaticEmbedder({})

    with pytest.raises(
        KeyError,
        match="no static vector",
    ):
        worker.embed("unknown lumber")


def test_same_vector_spaces_are_compatible() -> None:
    first = build_embedding(
        [1.0, 2.0, 3.0],
        "same-model",
    )

    second = build_embedding(
        [4.0, 5.0, 6.0],
        "same-model",
    )

    validate_compatible(
        first,
        second,
    )


def test_different_models_are_rejected() -> None:
    first = build_embedding(
        [1.0, 2.0],
        "oak-model",
    )

    second = build_embedding(
        [1.0, 2.0],
        "hickory-model",
    )

    with pytest.raises(
        ValueError,
        match="different models",
    ):
        validate_compatible(
            first,
            second,
        )


def test_different_dimensions_are_rejected() -> None:
    first = build_embedding(
        [1.0, 2.0],
        "same-model",
    )

    second = build_embedding(
        [1.0, 2.0, 3.0],
        "same-model",
    )

    with pytest.raises(
        ValueError,
        match="different dimensions",
    ):
        validate_compatible(
            first,
            second,
        )
