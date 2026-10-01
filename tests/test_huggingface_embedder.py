"""Tests for the GNOMEdata Hugging Face embedding worker."""

import pytest

from app.rag.huggingface_embedder import (
    HuggingFaceConfig,
    HuggingFaceEmbedder,
)


def make_worker() -> HuggingFaceEmbedder:
    """Build a worker with a fake token for local tests."""
    return HuggingFaceEmbedder(
        HuggingFaceConfig(
            token="fake-test-token",
            model="test-model",
        )
    )


def test_worker_reports_model_name() -> None:
    worker = make_worker()

    assert worker.model_name == "test-model"


def test_empty_token_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="token cannot be empty",
    ):
        HuggingFaceEmbedder(
            HuggingFaceConfig(
                token=" ",
            )
        )


def test_empty_model_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="model cannot be empty",
    ):
        HuggingFaceEmbedder(
            HuggingFaceConfig(
                token="fake-token",
                model=" ",
            )
        )


def test_direct_vector_is_accepted() -> None:
    worker = make_worker()

    vector = worker._extract_vector(
        [0.1, 0.2, 0.3]
    )

    assert vector == pytest.approx(
        [0.1, 0.2, 0.3]
    )


def test_token_vectors_are_mean_pooled() -> None:
    worker = make_worker()

    vector = worker._extract_vector([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    assert vector == pytest.approx(
        [2.0, 3.0]
    )


def test_empty_response_is_rejected() -> None:
    worker = make_worker()

    with pytest.raises(
        ValueError,
        match="empty embedding",
    ):
        worker._extract_vector([])


def test_non_list_response_is_rejected() -> None:
    worker = make_worker()

    with pytest.raises(
        ValueError,
        match="unexpected",
    ):
        worker._extract_vector({
            "error": "bad response",
        })


def test_inconsistent_token_dimensions_are_rejected() -> None:
    worker = make_worker()

    with pytest.raises(
        ValueError,
        match="inconsistent",
    ):
        worker._extract_vector([
            [1.0, 2.0],
            [3.0],
        ])


def test_environment_configuration(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "HUGGINGFACE_TOKEN",
        "fake-environment-token",
    )

    monkeypatch.setenv(
        "GNOMEDATA_EMBEDDING_MODEL",
        "environment-model",
    )

    config = HuggingFaceConfig.from_env()

    assert config.token == "fake-environment-token"
    assert config.model == "environment-model"


def test_missing_environment_token_is_rejected(
    monkeypatch,
) -> None:
    monkeypatch.delenv(
        "HUGGINGFACE_TOKEN",
        raising=False,
    )

    with pytest.raises(
        ValueError,
        match="not configured",
    ):
        HuggingFaceConfig.from_env()
