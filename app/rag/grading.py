"""The Grading Station: inspect boards before storage."""

from dataclasses import dataclass
from enum import Enum


class Grade(str, Enum):
    ACCEPT = "accept"
    REVIEW = "review"
    REJECT = "reject"


@dataclass(frozen=True)
class Inspection:
    grade: Grade
    reasons: tuple[str, ...]


def grade_board(text: str) -> Inspection:
    """Run basic quality checks, not a factuality assessment."""
    stripped = text.strip()

    if not stripped:
        return Inspection(Grade.REJECT, ("empty_board",))

    reasons: list[str] = []

    if len(stripped) < 30:
        reasons.append("very_short_board")

    if "\ufffd" in stripped:
        reasons.append("possible_text_corruption")

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if lines and len(lines) >= 4:
        counts = {line: lines.count(line) for line in set(lines)}
        if any(count >= 3 for count in counts.values()):
            reasons.append("repeated_line")

    if reasons:
        return Inspection(Grade.REVIEW, tuple(reasons))

    return Inspection(Grade.ACCEPT, ())


def grade_boards(texts: list[str]) -> list[Inspection]:
    """Grade a batch of boards in input order."""
    return [grade_board(text) for text in texts]
