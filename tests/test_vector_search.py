"""Tests for GNOMEdata Lumber Yard vector search."""

import pytest

from app.rag.embeddings import build_embedding
from app.rag.vector_search import (
    VectorBoard,
    search_vectors,
)


MODEL = "mill-model"


def embedding(values):
    """Build a test embedding in one consistent vector space."""
    return build_embedding(values, MODEL)


def test_closest_board_ranks_first() -> None:
    """The board nearest the order should ship first."""
    query = embedding([1.0, 1.0, 0.0])

    boards = [
        VectorBoard(
            "hydraulics",
            embedding([0.95, 0.90, 0.05]),
        ),
        VectorBoard(
            "invoice",
            embedding([0.0, 0.0, 1.0]),
        ),
    ]

    matches = search_vectors(query, boards)

    assert matches[0].board_id == "hydraulics"
    assert matches[0].similarity > matches[1].similarity


def test_results_are_sorted_by_similarity() -> None:
    """Lumber should leave the yard nearest-first."""
    query = embedding([1.0, 0.0, 0.0])

    boards = [
        VectorBoard(
            "weak",
            embedding([0.2, 0.9, 0.0]),
        ),
        VectorBoard(
            "strong",
            embedding([0.9, 0.1, 0.0]),
        ),
        VectorBoard(
            "medium",
            embedding([0.7, 0.5, 0.0]),
        ),
    ]

    matches = search_vectors(query, boards)

    assert [match.board_id for match in matches] == [
        "strong",
        "medium",
        "weak",
    ]


def test_limit_controls_shipment_size() -> None:
    """Dispatch should receive no more than the requested amount."""
    query = embedding([1.0, 0.0])

    boards = [
        VectorBoard("board-1", embedding([1.0, 0.0])),
        VectorBoard("board-2", embedding([0.9, 0.1])),
        VectorBoard("board-3", embedding([0.8, 0.2])),
    ]

    matches = search_vectors(
        query,
        boards,
        limit=2,
    )

    assert len(matches) == 2


def test_minimum_similarity_filters_weak_lumber() -> None:
    """Boards below the cutoff should stay in the yard."""
    query = embedding([1.0, 0.0])

    boards = [
        VectorBoard(
            "strong",
            embedding([1.0, 0.0]),
        ),
        VectorBoard(
            "weak",
            embedding([0.0, 1.0]),
        ),
    ]

    matches = search_vectors(
        query,
        boards,
        minimum_similarity=0.5,
    )

    assert [match.board_id for match in matches] == [
        "strong",
    ]


def test_empty_yard_returns_no_matches() -> None:
    """An empty Lumber Yard should return an empty shipment."""
    query = embedding([1.0, 0.0])

    assert search_vectors(query, []) == []


def test_invalid_limit_is_rejected() -> None:
    """Shipment size must be positive."""
    query = embedding([1.0, 0.0])

    with pytest.raises(
        ValueError,
        match="limit must be positive",
    ):
        search_vectors(
            query,
            [],
            limit=0,
        )


def test_invalid_similarity_threshold_is_rejected() -> None:
    """Cosine thresholds must stay inside the valid range."""
    query = embedding([1.0, 0.0])

    with pytest.raises(
        ValueError,
        match="between -1 and 1",
    ):
        search_vectors(
            query,
            [],
            minimum_similarity=1.5,
        )


def test_different_embedding_models_are_rejected() -> None:
    """Oak and hickory vector spaces must never be mixed."""
    query = build_embedding(
        [1.0, 0.0],
        "oak-model",
    )

    boards = [
        VectorBoard(
            "wrong-species",
            build_embedding(
                [1.0, 0.0],
                "hickory-model",
            ),
        ),
    ]

    with pytest.raises(
        ValueError,
        match="different models",
    ):
        search_vectors(query, boards)


def test_different_dimensions_are_rejected() -> None:
    """Boards from incompatible vector dimensions cannot mix."""
    query = embedding([1.0, 0.0])

    boards = [
        VectorBoard(
            "wrong-size",
            embedding([1.0, 0.0, 0.0]),
        ),
    ]

    with pytest.raises(
        ValueError,
        match="different dimensions",
    ):
        search_vectors(query, boards)


def test_blank_board_id_is_rejected() -> None:
    """Every piece of vector lumber must remain identifiable."""
    query = embedding([1.0, 0.0])

    boards = [
        VectorBoard(
            " ",
            embedding([1.0, 0.0]),
        ),
    ]

    with pytest.raises(
        ValueError,
        match="board_id cannot be empty",
    ):
        search_vectors(query, boards)
