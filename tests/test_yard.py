"""Tests for the Lumber Yard."""

import pytest

from app.rag.boards import Board
from app.rag.grading import Grade, Inspection
from app.rag.yard import LumberYard


def make_board(
    board_id: str = "board-1",
    text: str = "Pump maintenance requires a filter replacement.",
) -> Board:
    return Board(
        board_id=board_id,
        log_id="manual-1",
        source_name="pump-manual.txt",
        text=text,
        start_offset=0,
        end_offset=len(text),
    )


def test_store_and_retrieve_board(tmp_path) -> None:
    with LumberYard(tmp_path / "yard.db") as yard:
        board = make_board()
        yard.add_board(board, Inspection(Grade.ACCEPT, ()))

        stored = yard.get_board("board-1")

        assert stored is not None
        assert stored.text == board.text
        assert stored.log_id == "manual-1"
        assert stored.source_name == "pump-manual.txt"
        assert stored.start_offset == 0
        assert stored.end_offset == len(board.text)
        assert stored.grade == Grade.ACCEPT


def test_board_survives_reopening_database(tmp_path) -> None:
    database = tmp_path / "yard.db"

    with LumberYard(database) as yard:
        yard.add_board(
            make_board(),
            Inspection(Grade.ACCEPT, ()),
        )

    with LumberYard(database) as yard:
        assert yard.get_board("board-1") is not None


def test_keyword_search_finds_relevant_board(tmp_path) -> None:
    with LumberYard(tmp_path / "yard.db") as yard:
        yard.add_board(
            make_board(
                "pump",
                "Pump maintenance requires a filter replacement.",
            ),
            Inspection(Grade.ACCEPT, ()),
        )
        yard.add_board(
            make_board(
                "garden",
                "Water the garden when the soil is dry.",
            ),
            Inspection(Grade.ACCEPT, ()),
        )

        results = yard.search("pump filter")

        assert results
        assert results[0].board.board_id == "pump"
        assert results[0].score > 0


def test_review_boards_are_excluded_by_default(tmp_path) -> None:
    with LumberYard(tmp_path / "yard.db") as yard:
        yard.add_board(
            make_board("review"),
            Inspection(Grade.REVIEW, ("very_short_board",)),
        )

        assert yard.search("pump") == []
        assert len(yard.search("pump", include_review=True)) == 1


def test_rejected_boards_are_never_searchable(tmp_path) -> None:
    with LumberYard(tmp_path / "yard.db") as yard:
        yard.add_board(
            make_board("reject"),
            Inspection(Grade.REJECT, ("bad_source",)),
        )

        assert yard.search("pump", include_review=True) == []
        assert yard.count(Grade.REJECT) == 1


def test_same_board_id_updates_existing_record(tmp_path) -> None:
    with LumberYard(tmp_path / "yard.db") as yard:
        yard.add_board(
            make_board(text="Old maintenance instructions."),
            Inspection(Grade.REVIEW, ("very_short_board",)),
        )
        yard.add_board(
            make_board(text="Updated pump maintenance instructions."),
            Inspection(Grade.ACCEPT, ()),
        )

        assert yard.count() == 1
        stored = yard.get_board("board-1")
        assert stored is not None
        assert stored.text.startswith("Updated")
        assert stored.grade == Grade.ACCEPT


def test_empty_query_returns_no_results(tmp_path) -> None:
    with LumberYard(tmp_path / "yard.db") as yard:
        yard.add_board(
            make_board(),
            Inspection(Grade.ACCEPT, ()),
        )

        assert yard.search("   ") == []


def test_invalid_limit_is_rejected(tmp_path) -> None:
    with LumberYard(tmp_path / "yard.db") as yard:
        with pytest.raises(ValueError):
            yard.search("pump", limit=0)


def test_counts_by_grade(tmp_path) -> None:
    with LumberYard(tmp_path / "yard.db") as yard:
        yard.add_board(
            make_board("good"),
            Inspection(Grade.ACCEPT, ()),
        )
        yard.add_board(
            make_board("bad"),
            Inspection(Grade.REJECT, ("empty_board",)),
        )

        assert yard.count() == 2
        assert yard.count(Grade.ACCEPT) == 1
        assert yard.count(Grade.REJECT) == 1
