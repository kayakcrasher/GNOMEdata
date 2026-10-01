
"""Tests for the Board Cutter."""

import pytest

from app.rag.boards import Log, cut_boards


def test_short_log_produces_one_board() -> None:
    log = Log("log-1", "manual.txt", "Replace the filter.")

    boards = cut_boards(log, board_size=100, overlap=10)

    assert len(boards) == 1
    assert boards[0].text == log.text
    assert boards[0].log_id == "log-1"
    assert boards[0].source_name == "manual.txt"
    assert boards[0].start_offset == 0
    assert boards[0].end_offset == len(log.text)


def test_large_log_produces_overlapping_boards() -> None:
    text = " ".join(f"word{i}" for i in range(30))
    log = Log("log-2", "large.txt", text)

    boards = cut_boards(log, board_size=50, overlap=10)

    assert len(boards) > 1

    for previous, current in zip(boards, boards[1:]):
        assert previous.end_offset > current.start_offset


def test_board_text_matches_original_offsets() -> None:
    text = "Alpha bravo charlie delta echo foxtrot golf hotel."
    log = Log("log-3", "source.txt", text)

    boards = cut_boards(log, board_size=20, overlap=5)

    for board in boards:
        assert board.text == text[
            board.start_offset:board.end_offset
        ]


def test_board_size_must_be_positive() -> None:
    log = Log("log-4", "source.txt", "Some text")

    with pytest.raises(ValueError):
        cut_boards(log, board_size=0)


def test_overlap_must_be_smaller_than_board_size() -> None:
    log = Log("log-5", "source.txt", "Some text")

    with pytest.raises(ValueError):
        cut_boards(log, board_size=10, overlap=10)


def test_empty_log_produces_no_boards() -> None:
    log = Log("log-6", "empty.txt", "")

    assert cut_boards(log) == []

