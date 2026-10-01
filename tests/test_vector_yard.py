"""Tests for the GNOMEdata vector Lumber Yard."""

import pytest

from app.rag.boards import Board
from app.rag.embeddings import build_embedding
from app.rag.grading import Grade, Inspection
from app.rag.yard import LumberYard


MODEL = "mill-model"


def make_board(
    board_id: str,
    text: str,
) -> Board:
    """Build test lumber with valid provenance."""
    return Board(
        board_id=board_id,
        log_id="test-log",
        source_name="test-manual.txt",
        text=text,
        start_offset=0,
        end_offset=len(text),
    )


def embed(values):
    """Build an embedding in the test Lumber Yard space."""
    return build_embedding(
        values,
        MODEL,
    )


def test_embedding_can_be_stored_and_recovered(
    tmp_path,
) -> None:
    """Vector lumber should survive storage intact."""

    with LumberYard(tmp_path / "yard.db") as yard:
        board = make_board(
            "pump",
            "Hydraulic pump maintenance.",
        )

        yard.add_board(
            board,
            Inspection(Grade.ACCEPT, ()),
        )

        yard.add_embedding(
            board.board_id,
            embed([0.9, 0.8, 0.1]),
        )

        stored = yard.get_embedding(
            board.board_id,
            MODEL,
        )

        assert stored is not None
        assert stored.vector == pytest.approx(
            (0.9, 0.8, 0.1)
        )
        assert stored.model == MODEL
        assert stored.dimensions == 3


def test_embedding_survives_reopening_yard(
    tmp_path,
) -> None:
    """Vector placement should be durable."""

    database = tmp_path / "yard.db"

    with LumberYard(database) as yard:
        board = make_board(
            "pump",
            "Hydraulic pump maintenance.",
        )

        yard.add_board(
            board,
            Inspection(Grade.ACCEPT, ()),
        )

        yard.add_embedding(
            "pump",
            embed([1.0, 0.0, 0.0]),
        )

    with LumberYard(database) as yard:
        stored = yard.get_embedding(
            "pump",
            MODEL,
        )

        assert stored is not None
        assert stored.vector == pytest.approx(
            (1.0, 0.0, 0.0)
        )


def test_vector_search_returns_nearest_lumber_first(
    tmp_path,
) -> None:
    """The closest board should be dispatched first."""

    with LumberYard(tmp_path / "yard.db") as yard:
        lumber = [
            (
                make_board(
                    "hydraulics",
                    "Hydraulic hose inspection.",
                ),
                [0.95, 0.90, 0.05],
            ),
            (
                make_board(
                    "engine",
                    "Engine service procedure.",
                ),
                [0.70, 0.75, 0.15],
            ),
            (
                make_board(
                    "invoice",
                    "Invoice payment record.",
                ),
                [0.0, 0.0, 1.0],
            ),
        ]

        for board, vector in lumber:
            yard.add_board(
                board,
                Inspection(Grade.ACCEPT, ()),
            )

            yard.add_embedding(
                board.board_id,
                embed(vector),
            )

        query = embed(
            [1.0, 1.0, 0.0]
        )

        results = yard.vector_search(query)

        assert [
            result.board.board_id
            for result in results
        ] == [
            "hydraulics",
            "engine",
            "invoice",
        ]


def test_vector_search_respects_similarity_cutoff(
    tmp_path,
) -> None:
    """Weak lumber should remain in the yard."""

    with LumberYard(tmp_path / "yard.db") as yard:
        strong = make_board(
            "strong",
            "Relevant maintenance information.",
        )

        weak = make_board(
            "weak",
            "Unrelated accounting information.",
        )

        for board in (strong, weak):
            yard.add_board(
                board,
                Inspection(Grade.ACCEPT, ()),
            )

        yard.add_embedding(
            "strong",
            embed([1.0, 0.0]),
        )

        yard.add_embedding(
            "weak",
            embed([0.0, 1.0]),
        )

        results = yard.vector_search(
            embed([1.0, 0.0]),
            minimum_similarity=0.5,
        )

        assert [
            result.board.board_id
            for result in results
        ] == ["strong"]


def test_review_lumber_is_excluded_by_default(
    tmp_path,
) -> None:
    """Review lumber should not ship unless requested."""

    with LumberYard(tmp_path / "yard.db") as yard:
        board = make_board(
            "review",
            "Pump maintenance information.",
        )

        yard.add_board(
            board,
            Inspection(
                Grade.REVIEW,
                ("needs_review",),
            ),
        )

        yard.add_embedding(
            "review",
            embed([1.0, 0.0]),
        )

        query = embed([1.0, 0.0])

        assert yard.vector_search(query) == []

        results = yard.vector_search(
            query,
            include_review=True,
        )

        assert len(results) == 1
        assert results[0].board.board_id == "review"


def test_rejected_lumber_never_ships(
    tmp_path,
) -> None:
    """Rejected boards remain auditable but unretrievable."""

    with LumberYard(tmp_path / "yard.db") as yard:
        board = make_board(
            "reject",
            "Pump maintenance information.",
        )

        yard.add_board(
            board,
            Inspection(
                Grade.REJECT,
                ("bad_material",),
            ),
        )

        yard.add_embedding(
            "reject",
            embed([1.0, 0.0]),
        )

        results = yard.vector_search(
            embed([1.0, 0.0]),
            include_review=True,
        )

        assert results == []


def test_different_models_stay_in_separate_yards(
    tmp_path,
) -> None:
    """Incompatible vector spaces must never be mixed."""

    with LumberYard(tmp_path / "yard.db") as yard:
        board = make_board(
            "hickory",
            "Information from another vector space.",
        )

        yard.add_board(
            board,
            Inspection(Grade.ACCEPT, ()),
        )

        yard.add_embedding(
            "hickory",
            build_embedding(
                [1.0, 0.0],
                "hickory-model",
            ),
        )

        results = yard.vector_search(
            build_embedding(
                [1.0, 0.0],
                "oak-model",
            )
        )

        assert results == []


def test_unknown_board_cannot_receive_embedding(
    tmp_path,
) -> None:
    """Loose vectors cannot enter the yard without real lumber."""

    with LumberYard(tmp_path / "yard.db") as yard:
        with pytest.raises(
            KeyError,
            match="unknown board",
        ):
            yard.add_embedding(
                "ghost-board",
                embed([1.0, 0.0]),
            )


def test_same_board_can_use_multiple_models(
    tmp_path,
) -> None:
    """One board may be re-milled into multiple vector spaces."""

    with LumberYard(tmp_path / "yard.db") as yard:
        board = make_board(
            "board-1",
            "Hydraulic maintenance instructions.",
        )

        yard.add_board(
            board,
            Inspection(Grade.ACCEPT, ()),
        )

        yard.add_embedding(
            "board-1",
            build_embedding(
                [1.0, 0.0],
                "old-model",
            ),
        )

        yard.add_embedding(
            "board-1",
            build_embedding(
                [0.0, 1.0, 0.0],
                "new-model",
            ),
        )

        old = yard.get_embedding(
            "board-1",
            "old-model",
        )

        new = yard.get_embedding(
            "board-1",
            "new-model",
        )

        assert old is not None
        assert new is not None

        assert old.dimensions == 2
        assert new.dimensions == 3
