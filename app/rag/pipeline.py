"""The Mill Controller: orchestrate GNOMEdata's processing stations.

The controller does not perform each station's job itself.

It moves source material through the existing machinery:

Log
 -> Board Cutter
 -> Grading Station
 -> Stacker
 -> Embedder
 -> Lumber Yard

Rejected boards remain stored for auditing but are not embedded.
"""

from dataclasses import dataclass

from app.rag.boards import (
    Board,
    Log,
    cut_boards,
)
from app.rag.embeddings import Embedder
from app.rag.grading import (
    Grade,
    Inspection,
    grade_board,
)
from app.rag.stacker import (
    StackedBoard,
    sort_board,
)
from app.rag.yard import LumberYard


@dataclass(frozen=True)
class ProcessedBoard:
    """Record of one board's trip through the mill."""

    board: Board
    inspection: Inspection
    stacking: StackedBoard
    embedded: bool


@dataclass(frozen=True)
class MillReport:
    """Summary of one log processed by the mill."""

    log_id: str
    source_name: str
    boards: tuple[ProcessedBoard, ...]

    @property
    def total(self) -> int:
        """Return the number of boards produced."""
        return len(self.boards)

    @property
    def accepted(self) -> int:
        """Count accepted boards."""
        return sum(
            item.inspection.grade == Grade.ACCEPT
            for item in self.boards
        )

    @property
    def review(self) -> int:
        """Count boards requiring review."""
        return sum(
            item.inspection.grade == Grade.REVIEW
            for item in self.boards
        )

    @property
    def rejected(self) -> int:
        """Count rejected boards."""
        return sum(
            item.inspection.grade == Grade.REJECT
            for item in self.boards
        )

    @property
    def embedded(self) -> int:
        """Count boards placed into vector space."""
        return sum(
            item.embedded
            for item in self.boards
        )


class MillPipeline:
    """Move logs through GNOMEdata's processing machinery."""

    def __init__(
        self,
        yard: LumberYard,
        embedder: Embedder,
        board_size: int = 800,
        overlap: int = 120,
        embed_review: bool = True,
        collection_id: str = "default",
    ) -> None:
        if board_size < 1:
            raise ValueError(
                "board_size must be positive."
            )

        if not 0 <= overlap < board_size:
            raise ValueError(
                "overlap must be non-negative and "
                "smaller than board_size."
            )

        collection_id = collection_id.strip()

        if not collection_id:
            raise ValueError(
                "collection_id cannot be empty."
            )

        if yard.get_collection(collection_id) is None:
            raise KeyError(
                f"unknown collection: {collection_id}"
            )

        self.yard = yard
        self.embedder = embedder
        self.board_size = board_size
        self.overlap = overlap
        self.embed_review = embed_review
        self.collection_id = collection_id

    def process(
        self,
        log: Log,
    ) -> MillReport:
        """Process one source log into stored searchable lumber."""

        boards = cut_boards(
            log,
            board_size=self.board_size,
            overlap=self.overlap,
        )

        processed: list[ProcessedBoard] = []

        for board in boards:
            inspection = grade_board(
                board.text
            )

            stacking = sort_board(
                board.board_id,
                board.text,
            )

            self.yard.add_board(
                board,
                inspection,
                collection_id=self.collection_id,
            )

            self.yard.set_stacks(
                stacking
            )

            should_embed = (
                inspection.grade == Grade.ACCEPT
                or (
                    inspection.grade == Grade.REVIEW
                    and self.embed_review
                )
            )

            embedded = False

            if should_embed:
                embedding = self.embedder.embed(
                    board.text
                )

                self.yard.add_embedding(
                    board.board_id,
                    embedding,
                )

                embedded = True

            processed.append(
                ProcessedBoard(
                    board=board,
                    inspection=inspection,
                    stacking=stacking,
                    embedded=embedded,
                )
            )

        return MillReport(
            log_id=log.log_id,
            source_name=log.source_name,
            boards=tuple(processed),
        )
