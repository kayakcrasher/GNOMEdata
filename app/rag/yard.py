"""The Lumber Yard: durable storage and evidence retrieval."""

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from app.rag.boards import Board
from app.rag.grading import Grade, Inspection


@dataclass(frozen=True)
class StoredBoard:
    """A board recovered from the lumber yard."""

    board_id: str
    log_id: str
    source_name: str
    text: str
    start_offset: int
    end_offset: int
    grade: Grade
    reasons: tuple[str, ...]
    created_at: str


@dataclass(frozen=True)
class SearchResult:
    """A retrieved board and its lexical relevance score."""

    board: StoredBoard
    score: float


def _tokens(text: str) -> list[str]:
    """Normalize text into searchable word tokens."""
    return re.findall(r"\b[\w'-]+\b", text.casefold())


class LumberYard:
    """SQLite-backed storage for graded boards.

    Accepted boards are searchable by default. Review boards
    can be included explicitly. Rejected boards remain stored
    for auditing but are never returned by search.
    """

    def __init__(self, database: str | Path = "data/gnomedata.db"):
        self.database = str(database)

        if self.database != ":memory:":
            Path(self.database).parent.mkdir(
                parents=True, exist_ok=True
            )

        self._connection = sqlite3.connect(self.database)
        self._connection.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        """Create the storage table and useful indexes."""
        with self._connection:
            self._connection.execute("""
                CREATE TABLE IF NOT EXISTS boards (
                    board_id TEXT PRIMARY KEY,
                    log_id TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    text TEXT NOT NULL,
                    start_offset INTEGER NOT NULL,
                    end_offset INTEGER NOT NULL,
                    grade TEXT NOT NULL
                        CHECK (grade IN ('accept', 'review', 'reject')),
                    reasons_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    CHECK (start_offset >= 0),
                    CHECK (end_offset >= start_offset)
                )
            """)

            self._connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_boards_log
                ON boards(log_id)
            """)

            self._connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_boards_grade
                ON boards(grade)
            """)

    def add_board(
        self,
        board: Board,
        inspection: Inspection,
    ) -> None:
        """Store a board and its inspection result.

        Re-adding the same board ID updates its contents and grade.
        """
        if not board.board_id.strip():
            raise ValueError("board_id cannot be empty.")

        if not board.log_id.strip():
            raise ValueError("log_id cannot be empty.")

        if board.start_offset < 0:
            raise ValueError("start_offset cannot be negative.")

        if board.end_offset < board.start_offset:
            raise ValueError("end_offset precedes start_offset.")

        if not isinstance(inspection.grade, Grade):
            raise ValueError("inspection must contain a valid Grade.")

        with self._connection:
            self._connection.execute("""
                INSERT INTO boards (
                    board_id, log_id, source_name, text,
                    start_offset, end_offset, grade, reasons_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(board_id) DO UPDATE SET
                    log_id = excluded.log_id,
                    source_name = excluded.source_name,
                    text = excluded.text,
                    start_offset = excluded.start_offset,
                    end_offset = excluded.end_offset,
                    grade = excluded.grade,
                    reasons_json = excluded.reasons_json
            """, (
                board.board_id,
                board.log_id,
                board.source_name,
                board.text,
                board.start_offset,
                board.end_offset,
                inspection.grade.value,
                json.dumps(inspection.reasons),
            ))

    def get_board(self, board_id: str) -> StoredBoard | None:
        """Retrieve one board by its unique ID."""
        row = self._connection.execute(
            "SELECT * FROM boards WHERE board_id = ?",
            (board_id,),
        ).fetchone()

        return self._to_board(row) if row else None

    def search(
        self,
        query: str,
        limit: int = 5,
        include_review: bool = False,
    ) -> list[SearchResult]:
        """Retrieve boards by keyword relevance.

        This is lexical search, not semantic vector search.
        Scores reward query-term coverage and repeated matches.
        """
        if limit < 1:
            raise ValueError("limit must be positive.")

        query_tokens = _tokens(query)
        if not query_tokens:
            return []

        unique_terms = list(dict.fromkeys(query_tokens))

        if include_review:
            rows = self._connection.execute("""
                SELECT * FROM boards
                WHERE grade IN ('accept', 'review')
            """).fetchall()
        else:
            rows = self._connection.execute("""
                SELECT * FROM boards
                WHERE grade = 'accept'
            """).fetchall()

        results: list[SearchResult] = []

        for row in rows:
            board = self._to_board(row)
            board_tokens = _tokens(board.text)

            if not board_tokens:
                continue

            frequencies: dict[str, int] = {}
            for token in board_tokens:
                frequencies[token] = frequencies.get(token, 0) + 1

            matched = sum(
                1 for term in unique_terms if frequencies.get(term, 0)
            )

            if not matched:
                continue

            # Favor coverage of the query; cap repetition so
            # a word repeated many times cannot dominate ranking.
            coverage = matched / len(unique_terms)
            frequency = sum(
                min(frequencies.get(term, 0), 3)
                for term in unique_terms
            )
            score = coverage + 0.05 * frequency

            if query.casefold().strip() in board.text.casefold():
                score += 0.5

            results.append(SearchResult(board=board, score=score))

        results.sort(
            key=lambda result: (
                result.score,
                result.board.board_id,
            ),
            reverse=True,
        )
        return results[:limit]

    def count(self, grade: Grade | None = None) -> int:
        """Count all boards or boards of one grade."""
        if grade is None:
            row = self._connection.execute(
                "SELECT COUNT(*) AS total FROM boards"
            ).fetchone()
        else:
            row = self._connection.execute(
                "SELECT COUNT(*) AS total FROM boards WHERE grade = ?",
                (grade.value,),
            ).fetchone()

        return int(row["total"])

    def close(self) -> None:
        """Close the database connection."""
        self._connection.close()

    def __enter__(self) -> "LumberYard":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    @staticmethod
    def _to_board(row: sqlite3.Row) -> StoredBoard:
        """Convert a database row into a typed board."""
        return StoredBoard(
            board_id=row["board_id"],
            log_id=row["log_id"],
            source_name=row["source_name"],
            text=row["text"],
            start_offset=row["start_offset"],
            end_offset=row["end_offset"],
            grade=Grade(row["grade"]),
            reasons=tuple(json.loads(row["reasons_json"])),
            created_at=row["created_at"],
        )
