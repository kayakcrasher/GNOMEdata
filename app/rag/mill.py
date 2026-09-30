
"""The Mill: turn raw logs into useful lumber.

A log is extracted source material.
Boards preserve contextual passages.
Railroad ties preserve precise facts.
Every product keeps its source identity.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Log:
    """Raw material delivered from the forest."""

    log_id: str
    source_name: str
    text: str


@dataclass(frozen=True)
class Board:
    """A contextual passage cut from a log."""

    board_id: str
    log_id: str
    source_name: str
    text: str
    start_offset: int
    end_offset: int


@dataclass(frozen=True)
class RailroadTie:
    """A precise fact cut from a log."""

    tie_id: str
    log_id: str
    source_name: str
    subject: str
    relation: str
    value: str


def cut_boards(
    log: Log,
    board_size: int = 800,
    overlap: int = 120,
) -> list[Board]:
    """Cut a log into overlapping contextual boards.

    Size and overlap are measured in characters for now.
    Production versions can use token-aware cutting.
    """
    if board_size < 1:
        raise ValueError("Board size must be positive.")

    if not 0 <= overlap < board_size:
        raise ValueError(
            "Overlap must be non-negative and smaller "
            "than board size."
        )

    boards: list[Board] = []
    start = 0
    step = board_size - overlap

    while start < len(log.text):
        end = min(start + board_size, len(log.text))
        passage = log.text[start:end]

        if passage.strip():
            boards.append(
                Board(
                    board_id=f"{log.log_id}:board:{len(boards)}",
                    log_id=log.log_id,
                    source_name=log.source_name,
                    text=passage,
                    start_offset=start,
                    end_offset=end,
                )
            )

        if end == len(log.text):
            break

        start += step

    return boards
