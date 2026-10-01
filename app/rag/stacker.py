"""The Stacker: sort finished boards into useful information stacks.

At a sawmill, boards leave the edger on a chain. A stacker sorts
those boards into the piles where they belong.

GNOMEdata follows the same model.

The Stacker does not alter a board's contents. It creates sorting
records describing where that board belongs.
"""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Stack:
    """A destination assigned to a board."""

    name: str
    confidence: float
    matched_terms: tuple[str, ...]


@dataclass(frozen=True)
class StackedBoard:
    """The complete sorting result for one board."""

    board_id: str
    stacks: tuple[Stack, ...]


DEFAULT_STACK_RULES: dict[str, tuple[str, ...]] = {
    "maintenance": (
        "maintenance",
        "maintain",
        "service",
        "inspect",
        "inspection",
        "replace",
        "repair",
    ),
    "safety": (
        "warning",
        "danger",
        "hazard",
        "safety",
        "caution",
        "injury",
    ),
    "operations": (
        "operate",
        "operation",
        "procedure",
        "start",
        "stop",
        "startup",
        "shutdown",
    ),
    "finance": (
        "invoice",
        "payment",
        "price",
        "cost",
        "revenue",
        "expense",
        "budget",
    ),
}


def _tokenize(text: str) -> set[str]:
    """Normalize text into unique searchable terms."""
    return set(
        re.findall(
            r"[a-z0-9]+",
            text.casefold(),
        )
    )


def sort_board(
    board_id: str,
    text: str,
    rules: dict[str, tuple[str, ...]] | None = None,
) -> StackedBoard:
    """Sort one board into zero or more stacks.

    A board may belong to multiple stacks.

    Confidence represents the fraction of a stack's configured
    vocabulary found in the board. It is not an AI probability.
    """
    if not board_id.strip():
        raise ValueError("board_id cannot be empty.")

    if not text.strip():
        return StackedBoard(
            board_id=board_id,
            stacks=(),
        )

    active_rules = (
        DEFAULT_STACK_RULES
        if rules is None
        else rules
    )

    tokens = _tokenize(text)
    stacks: list[Stack] = []

    for stack_name, terms in active_rules.items():
        normalized_terms = tuple(
            dict.fromkeys(
                term.casefold().strip()
                for term in terms
                if term.strip()
            )
        )

        if not normalized_terms:
            continue

        matched_terms = tuple(
            term
            for term in normalized_terms
            if term in tokens
        )

        if not matched_terms:
            continue

        confidence = (
            len(matched_terms)
            / len(normalized_terms)
        )

        stacks.append(
            Stack(
                name=stack_name,
                confidence=confidence,
                matched_terms=matched_terms,
            )
        )

    stacks.sort(
        key=lambda stack: (
            -stack.confidence,
            stack.name,
        )
    )

    return StackedBoard(
        board_id=board_id,
        stacks=tuple(stacks),
    )
