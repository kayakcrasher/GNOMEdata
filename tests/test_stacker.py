"""Tests for the GNOMEdata Stacker."""

import pytest

from app.rag.stacker import sort_board


def test_board_can_enter_multiple_stacks() -> None:
    """One board may legitimately belong in several stacks."""
    result = sort_board(
        "board-1",
        "Warning: inspect the pump before operation.",
    )

    names = {
        stack.name
        for stack in result.stacks
    }

    assert names == {
        "safety",
        "maintenance",
        "operations",
    }


def test_unknown_material_remains_unstacked() -> None:
    """Unrecognized material should not receive invented labels."""
    result = sort_board(
        "board-2",
        "The oak tree grows beside the river.",
    )

    assert result.stacks == ()


def test_empty_material_remains_unstacked() -> None:
    """An empty board has nothing to sort."""
    result = sort_board(
        "board-3",
        "   ",
    )

    assert result.stacks == ()


def test_missing_board_id_is_rejected() -> None:
    """Every sorting record must remain traceable to a board."""
    with pytest.raises(
        ValueError,
        match="board_id cannot be empty",
    ):
        sort_board(
            "",
            "Inspect the pump.",
        )


def test_matching_is_case_insensitive() -> None:
    """Capitalization must not affect stack assignment."""
    result = sort_board(
        "board-4",
        "WARNING: follow the SAFETY procedure.",
    )

    names = {
        stack.name
        for stack in result.stacks
    }

    assert "safety" in names
    assert "operations" in names


def test_custom_stack_rules_are_supported() -> None:
    """A customer or domain can define specialized stacks."""
    rules = {
        "hydraulics": (
            "pump",
            "hose",
            "pressure",
        ),
    }

    result = sort_board(
        "board-5",
        "Check pump pressure before use.",
        rules=rules,
    )

    assert len(result.stacks) == 1

    stack = result.stacks[0]

    assert stack.name == "hydraulics"
    assert stack.confidence == pytest.approx(2 / 3)
    assert set(stack.matched_terms) == {
        "pump",
        "pressure",
    }


def test_confidence_measures_rule_coverage() -> None:
    """Confidence is deterministic vocabulary coverage."""
    rules = {
        "maintenance": (
            "inspect",
            "repair",
            "replace",
            "service",
        ),
    }

    result = sort_board(
        "board-6",
        "Inspect and replace the filter.",
        rules=rules,
    )

    assert result.stacks[0].confidence == pytest.approx(0.5)
