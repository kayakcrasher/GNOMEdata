"""Tests for the Grading Station."""

from app.rag.grading import Grade, grade_board, grade_boards


def test_good_board_is_accepted() -> None:
    result = grade_board(
        "The pump requires maintenance every 90 days."
    )
    assert result.grade == Grade.ACCEPT
    assert result.reasons == ()


def test_empty_board_is_rejected() -> None:
    result = grade_board(" \n ")
    assert result.grade == Grade.REJECT
    assert "empty_board" in result.reasons


def test_short_board_needs_review() -> None:
    result = grade_board("Replace filter.")
    assert result.grade == Grade.REVIEW
    assert "very_short_board" in result.reasons


def test_corrupted_text_needs_review() -> None:
    result = grade_board("This manual contains a damaged character \ufffd here.")
    assert result.grade == Grade.REVIEW
    assert "possible_text_corruption" in result.reasons


def test_repeated_lines_need_review() -> None:
    text = "Header\nHeader\nHeader\nUseful maintenance instructions."
    result = grade_board(text)
    assert result.grade == Grade.REVIEW
    assert "repeated_line" in result.reasons


def test_batch_grading_preserves_order() -> None:
    results = grade_boards([
        "A useful board containing enough text for review.",
        "",
        "Short.",
    ])
    assert [item.grade for item in results] == [
        Grade.ACCEPT,
        Grade.REJECT,
        Grade.REVIEW,
    ]
