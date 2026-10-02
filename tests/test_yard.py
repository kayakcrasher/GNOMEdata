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


def test_duplicate_text_different_collections_is_allowed(tmp_path) -> None:
    with LumberYard(tmp_path / "yard.db") as yard:
        yard.create_collection("project-b", "Project B")

        yard.add_board(
            make_board("default-copy", "Randy sells pies."),
            Inspection(Grade.ACCEPT, ()),
            collection_id="default",
        )
        yard.add_board(
            make_board("project-copy", "Randy sells pies."),
            Inspection(Grade.ACCEPT, ()),
            collection_id="project-b",
        )

        assert len(yard.list_boards("default")) == 1
        assert len(yard.list_boards("project-b")) == 1


def test_different_text_same_collection_is_allowed(tmp_path) -> None:
    with LumberYard(tmp_path / "yard.db") as yard:
        yard.add_board(
            make_board("board-a", "Randy sells pies."),
            Inspection(Grade.ACCEPT, ()),
        )
        yard.add_board(
            make_board("board-b", "Randy lives in Denver."),
            Inspection(Grade.ACCEPT, ()),
        )

        assert len(yard.list_boards("default")) == 2

def test_delete_source_removes_its_boards(tmp_path) -> None:
    with LumberYard(tmp_path / "yard.db") as yard:
        yard.add_board(
            make_board("randy-1", "Randy sells pies."),
            Inspection(Grade.ACCEPT, ()),
        )

        other = Board(
            board_id="pump-2",
            log_id="manual-2",
            source_name="other-manual.txt",
            text="Inspect the hydraulic pump.",
            start_offset=0,
            end_offset=len("Inspect the hydraulic pump."),
        )

        yard.add_board(
            other,
            Inspection(Grade.ACCEPT, ()),
        )

        removed = yard.delete_source(
            "pump-manual.txt",
            collection_id="default",
        )

        assert removed == 1
        assert yard.get_board("randy-1") is None
        assert yard.get_board("pump-2") is not None


def test_delete_source_is_collection_scoped(tmp_path) -> None:
    with LumberYard(tmp_path / "yard.db") as yard:
        yard.create_collection("other", "Other")

        first = make_board(
            "default-board",
            "Default collection knowledge.",
        )

        second = Board(
            board_id="other-board",
            log_id="other-log",
            source_name=first.source_name,
            text="Other collection knowledge.",
            start_offset=0,
            end_offset=len("Other collection knowledge."),
        )

        yard.add_board(
            first,
            Inspection(Grade.ACCEPT, ()),
            collection_id="default",
        )

        yard.add_board(
            second,
            Inspection(Grade.ACCEPT, ()),
            collection_id="other",
        )

        removed = yard.delete_source(
            first.source_name,
            collection_id="default",
        )

        assert removed == 1
        assert yard.get_board("default-board") is None
        assert yard.get_board("other-board") is not None
