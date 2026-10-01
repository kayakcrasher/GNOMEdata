
"""The Board Cutter: cut contextual passages from source logs."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Log:
    """Original source material entering the mill."""

    log_id: str
    source_name: str
    text: str


@dataclass(frozen=True)
class Board:
    """A contextual passage traceable to its original log."""

    board_id: str
    log_id: str
    source_name: str
    text: str
    start_offset: int
    end_offset: int


def cut_boards(
    log: Log,
    board_size: int = 800,
    overlap: int = 120,
) -> list[Board]:
    """Cut a log into overlapping passages.

    Sizes and offsets are measured in characters, not tokens.
    Each board retains its source identity and original offsets.
    """
    if board_size < 1:
        raise ValueError("board_size must be positive.")

    if not 0 <= overlap < board_size:
        raise ValueError(
            "overlap must be non-negative and smaller "
            "than board_size."
        )

    boards: list[Board] = []
    start = 0
    step = board_size - overlap

    while start < len(log.text):
        end = min(start + board_size, len(log.text))

        # Prefer a nearby word boundary without exceeding board_size.
        if end < len(log.text):
            boundary = log.text.rfind(" ", start, end)
            if boundary > start + step:
                end = boundary

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

        if end >= len(log.text):
            break

        # Guarantee forward progress even for long unbroken text.
        start += step

    return boards

