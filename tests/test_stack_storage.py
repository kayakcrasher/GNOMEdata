"""Tests for persistent Lumber Yard stack assignments."""

from app.rag.boards import Board
from app.rag.grading import Grade, Inspection
from app.rag.stacker import sort_board
from app.rag.yard import LumberYard


def make_board(
    board_id: str,
    text: str,
) -> Board:
    """Create test lumber."""
    return Board(
        board_id=board_id,
        log_id=f"log-{board_id}",
        source_name="manual.txt",
        text=text,
        start_offset=0,
        end_offset=len(text),
    )


def test_stack_assignments_can_be_stored(
    tmp_path,
) -> None:
    text = (
        "Inspect and service the machine during "
        "scheduled maintenance."
    )

    with LumberYard(tmp_path / "yard.db") as yard:
        board = make_board("board-1", text)

        yard.add_board(
            board,
            Inspection(Grade.ACCEPT, ()),
        )

        stacking = sort_board(
            board.board_id,
            board.text,
        )

        yard.set_stacks(stacking)

        stored = yard.get_stacks(board.board_id)

        assert stored.board_id == board.board_id
        assert stored.stacks
        assert stored.stacks[0].name == "maintenance"


def test_multiple_stack_assignments_survive_reopening(
    tmp_path,
) -> None:
    database = tmp_path / "yard.db"

    text = (
        "Warning: inspect and service equipment "
        "before maintenance and repair."
    )

    with LumberYard(database) as yard:
        board = make_board("multi-stack", text)

        yard.add_board(
            board,
            Inspection(Grade.ACCEPT, ()),
        )

        yard.set_stacks(
            sort_board(
                board.board_id,
                board.text,
            )
        )

    with LumberYard(database) as yard:
        stored = yard.get_stacks("multi-stack")

        names = {
            stack.name
            for stack in stored.stacks
        }

        assert "maintenance" in names
        assert "safety" in names


def test_restacking_replaces_old_assignments(
    tmp_path,
) -> None:
    with LumberYard(tmp_path / "yard.db") as yard:
        board = make_board(
            "restack",
            "Inspect equipment during maintenance.",
        )

        yard.add_board(
            board,
            Inspection(Grade.ACCEPT, ()),
        )

        first = sort_board(
            board.board_id,
            "Inspect equipment during maintenance.",
        )

        second = sort_board(
            board.board_id,
            "Warning danger safety injury.",
        )

        yard.set_stacks(first)
        yard.set_stacks(second)

        stored = yard.get_stacks("restack")

        names = {
            stack.name
            for stack in stored.stacks
        }

        assert "safety" in names
        assert "maintenance" not in names


def test_unknown_board_cannot_receive_stacks(
    tmp_path,
) -> None:
    with LumberYard(tmp_path / "yard.db") as yard:
        stacking = sort_board(
            "ghost-board",
            "Inspect equipment during maintenance.",
        )

        try:
            yard.set_stacks(stacking)
        except KeyError as exc:
            assert "ghost-board" in str(exc)
        else:
            raise AssertionError(
                "Expected unknown board to raise KeyError."
            )


def test_board_with_no_matches_has_empty_stack_set(
    tmp_path,
) -> None:
    text = (
        "The blue notebook sits beside the window."
    )

    with LumberYard(tmp_path / "yard.db") as yard:
        board = make_board(
            "unclassified",
            text,
        )

        yard.add_board(
            board,
            Inspection(Grade.ACCEPT, ()),
        )

        yard.set_stacks(
            sort_board(
                board.board_id,
                board.text,
            )
        )

        stored = yard.get_stacks(
            board.board_id
        )

        assert stored.board_id == board.board_id
        assert stored.stacks == ()
