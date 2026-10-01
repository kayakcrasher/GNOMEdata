"""The Lumber Yard: durable storage and evidence retrieval.

Finished boards live in the Lumber Yard.

The yard preserves:
- original board provenance
- inspection grade
- lexical search
- embedding vectors

Embedding vectors form the semantic space of the Lumber Yard.
Different embedding models are kept separate so incompatible
vector spaces are never mixed.
"""

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from app.rag.boards import Board
from app.rag.embeddings import Embedding, build_embedding
from app.rag.grading import Grade, Inspection
from app.rag.vector_search import VectorBoard, search_vectors


@dataclass(frozen=True)
class Collection:
    """A named collection of related lumber."""

    collection_id: str
    name: str
    created_at: str


@dataclass(frozen=True)
class StoredBoard:
    """A board recovered from the Lumber Yard."""

    board_id: str
    collection_id: str
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
    """A retrieved board and its relevance score."""

    board: StoredBoard
    score: float


def _tokens(text: str) -> list[str]:
    """Normalize text into searchable word tokens."""
    return re.findall(
        r"\b[\w'-]+\b",
        text.casefold(),
    )


class LumberYard:
    """SQLite-backed storage for graded and embedded boards."""

    def __init__(
        self,
        database: str | Path = "data/gnomedata.db",
    ) -> None:
        self.database = str(database)

        if self.database != ":memory:":
            Path(self.database).parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        self._connection = sqlite3.connect(
            self.database
        )
        self._connection.row_factory = sqlite3.Row

        self._create_schema()
        self._migrate_collections()

    def _create_schema(self) -> None:
        """Create Lumber Yard tables and indexes."""
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
                        CHECK (
                            grade IN (
                                'accept',
                                'review',
                                'reject'
                            )
                        ),
                    reasons_json TEXT NOT NULL
                        DEFAULT '[]',
                    created_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,
                    CHECK (start_offset >= 0),
                    CHECK (
                        end_offset >= start_offset
                    )
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

            self._connection.execute("""
                CREATE TABLE IF NOT EXISTS board_embeddings (
                    board_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    vector_json TEXT NOT NULL,

                    PRIMARY KEY (
                        board_id,
                        model
                    ),

                    FOREIGN KEY (board_id)
                        REFERENCES boards(board_id)
                        ON DELETE CASCADE,

                    CHECK (dimensions > 0)
                )
            """)

            self._connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_model
                ON board_embeddings(model)
            """)

    def _migrate_collections(self) -> None:
        """Upgrade old Lumber Yard databases to collection-aware storage."""

        with self._connection:
            self._connection.execute("""
                CREATE TABLE IF NOT EXISTS collections (
                    collection_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                )
            """)

            self._connection.execute("""
                INSERT OR IGNORE INTO collections (
                    collection_id,
                    name
                )
                VALUES ('default', 'Default')
            """)

            columns = {
                row["name"]
                for row in self._connection.execute(
                    "PRAGMA table_info(boards)"
                ).fetchall()
            }

            if "collection_id" not in columns:
                self._connection.execute("""
                    ALTER TABLE boards
                    ADD COLUMN collection_id TEXT
                    NOT NULL DEFAULT 'default'
                """)

            self._connection.execute("""
                CREATE INDEX IF NOT EXISTS
                    idx_boards_collection
                ON boards(collection_id)
            """)

    def create_collection(
        self,
        collection_id: str,
        name: str,
    ) -> None:
        """Create or update a named lumber collection."""

        collection_id = collection_id.strip()
        name = name.strip()

        if not collection_id:
            raise ValueError(
                "collection_id cannot be empty."
            )

        if not name:
            raise ValueError(
                "collection name cannot be empty."
            )

        with self._connection:
            self._connection.execute("""
                INSERT INTO collections (
                    collection_id,
                    name
                )
                VALUES (?, ?)

                ON CONFLICT(collection_id)
                DO UPDATE SET
                    name = excluded.name
            """, (
                collection_id,
                name,
            ))

    def get_collection(
        self,
        collection_id: str,
    ) -> Collection | None:
        """Retrieve one lumber collection."""

        row = self._connection.execute("""
            SELECT
                collection_id,
                name,
                created_at
            FROM collections
            WHERE collection_id = ?
        """, (
            collection_id,
        )).fetchone()

        if row is None:
            return None

        return Collection(
            collection_id=row["collection_id"],
            name=row["name"],
            created_at=row["created_at"],
        )

    def add_board(
        self,
        board: Board,
        inspection: Inspection,
        collection_id: str = "default",
    ) -> None:
        """Store a board and its inspection result."""

        if not board.board_id.strip():
            raise ValueError(
                "board_id cannot be empty."
            )

        if not board.log_id.strip():
            raise ValueError(
                "log_id cannot be empty."
            )

        if board.start_offset < 0:
            raise ValueError(
                "start_offset cannot be negative."
            )

        if board.end_offset < board.start_offset:
            raise ValueError(
                "end_offset precedes start_offset."
            )

        if not isinstance(
            inspection.grade,
            Grade,
        ):
            raise ValueError(
                "inspection must contain a valid Grade."
            )

        collection_id = collection_id.strip()

        if not collection_id:
            raise ValueError(
                "collection_id cannot be empty."
            )

        if self.get_collection(collection_id) is None:
            raise KeyError(
                f"unknown collection: {collection_id}"
            )

        with self._connection:
            self._connection.execute("""
                INSERT INTO boards (
                    board_id,
                    log_id,
                    source_name,
                    text,
                    start_offset,
                    end_offset,
                    grade,
                    reasons_json,
                    collection_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)

                ON CONFLICT(board_id)
                DO UPDATE SET
                    log_id = excluded.log_id,
                    source_name = excluded.source_name,
                    text = excluded.text,
                    start_offset = excluded.start_offset,
                    end_offset = excluded.end_offset,
                    grade = excluded.grade,
                    reasons_json = excluded.reasons_json,
                    collection_id = excluded.collection_id
            """, (
                board.board_id,
                board.log_id,
                board.source_name,
                board.text,
                board.start_offset,
                board.end_offset,
                inspection.grade.value,
                json.dumps(
                    inspection.reasons
                ),
                collection_id,
            ))

    def add_embedding(
        self,
        board_id: str,
        embedding: Embedding,
    ) -> None:
        """Place one board into a Lumber Yard vector space."""

        if not board_id.strip():
            raise ValueError(
                "board_id cannot be empty."
            )

        board = self.get_board(board_id)

        if board is None:
            raise KeyError(
                f"unknown board: {board_id}"
            )

        if embedding.dimensions != len(
            embedding.vector
        ):
            raise ValueError(
                "embedding dimensions do not match vector."
            )

        with self._connection:
            self._connection.execute("""
                INSERT INTO board_embeddings (
                    board_id,
                    model,
                    dimensions,
                    vector_json
                )
                VALUES (?, ?, ?, ?)

                ON CONFLICT(board_id, model)
                DO UPDATE SET
                    dimensions = excluded.dimensions,
                    vector_json = excluded.vector_json
            """, (
                board_id,
                embedding.model,
                embedding.dimensions,
                json.dumps(
                    embedding.vector
                ),
            ))

    def get_embedding(
        self,
        board_id: str,
        model: str,
    ) -> Embedding | None:
        """Recover one board embedding from the yard."""

        row = self._connection.execute("""
            SELECT
                model,
                dimensions,
                vector_json
            FROM board_embeddings
            WHERE board_id = ?
              AND model = ?
        """, (
            board_id,
            model,
        )).fetchone()

        if row is None:
            return None

        vector = json.loads(
            row["vector_json"]
        )

        embedding = build_embedding(
            vector,
            row["model"],
        )

        if embedding.dimensions != row["dimensions"]:
            raise ValueError(
                "stored embedding dimensions are corrupt."
            )

        return embedding

    def get_board(
        self,
        board_id: str,
    ) -> StoredBoard | None:
        """Retrieve one board by its unique ID."""

        row = self._connection.execute(
            """
            SELECT *
            FROM boards
            WHERE board_id = ?
            """,
            (board_id,),
        ).fetchone()

        return (
            self._to_board(row)
            if row
            else None
        )

    def search(
        self,
        query: str,
        limit: int = 5,
        include_review: bool = False,
        collection_id: str = "default",
    ) -> list[SearchResult]:
        """Retrieve boards by lexical relevance."""

        if limit < 1:
            raise ValueError(
                "limit must be positive."
            )

        query_tokens = _tokens(query)

        if not query_tokens:
            return []

        unique_terms = list(
            dict.fromkeys(query_tokens)
        )

        rows = self._searchable_rows(
            include_review,
            collection_id,
        )

        results: list[SearchResult] = []

        for row in rows:
            board = self._to_board(row)
            board_tokens = _tokens(board.text)

            if not board_tokens:
                continue

            frequencies: dict[str, int] = {}

            for token in board_tokens:
                frequencies[token] = (
                    frequencies.get(token, 0)
                    + 1
                )

            matched = sum(
                1
                for term in unique_terms
                if frequencies.get(term, 0)
            )

            if not matched:
                continue

            coverage = (
                matched
                / len(unique_terms)
            )

            frequency = sum(
                min(
                    frequencies.get(term, 0),
                    3,
                )
                for term in unique_terms
            )

            score = (
                coverage
                + 0.05 * frequency
            )

            phrase = query.casefold().strip()

            if phrase in board.text.casefold():
                score += 0.5

            results.append(
                SearchResult(
                    board=board,
                    score=score,
                )
            )

        results.sort(
            key=lambda result: (
                -result.score,
                result.board.board_id,
            )
        )

        return results[:limit]

    def vector_search(
        self,
        query: Embedding,
        limit: int = 5,
        minimum_similarity: float = -1.0,
        include_review: bool = False,
        collection_id: str = "default",
    ) -> list[SearchResult]:
        """Retrieve boards from the Lumber Yard by vector similarity."""

        if limit < 1:
            raise ValueError(
                "limit must be positive."
            )

        rows = self._connection.execute("""
            SELECT
                b.*,
                e.model AS embedding_model,
                e.dimensions AS embedding_dimensions,
                e.vector_json AS embedding_vector
            FROM boards AS b
            JOIN board_embeddings AS e
                ON e.board_id = b.board_id
            WHERE e.model = ?
              AND b.collection_id = ?
              AND b.grade IN (
                  'accept',
                  ?
              )
        """, (
            query.model,
            collection_id,
            (
                "review"
                if include_review
                else "accept"
            ),
        )).fetchall()

        vector_boards: list[VectorBoard] = []

        stored_boards: dict[
            str,
            StoredBoard,
        ] = {}

        for row in rows:
            vector = json.loads(
                row["embedding_vector"]
            )

            embedding = build_embedding(
                vector,
                row["embedding_model"],
            )

            if (
                embedding.dimensions
                != row["embedding_dimensions"]
            ):
                raise ValueError(
                    "stored embedding dimensions are corrupt."
                )

            board = self._to_board(row)

            stored_boards[
                board.board_id
            ] = board

            vector_boards.append(
                VectorBoard(
                    board_id=board.board_id,
                    embedding=embedding,
                )
            )

        matches = search_vectors(
            query=query,
            boards=vector_boards,
            limit=limit,
            minimum_similarity=minimum_similarity,
        )

        return [
            SearchResult(
                board=stored_boards[
                    match.board_id
                ],
                score=match.similarity,
            )
            for match in matches
        ]

    def count(
        self,
        grade: Grade | None = None,
    ) -> int:
        """Count all boards or boards of one grade."""

        if grade is None:
            row = self._connection.execute("""
                SELECT COUNT(*) AS total
                FROM boards
            """).fetchone()
        else:
            row = self._connection.execute("""
                SELECT COUNT(*) AS total
                FROM boards
                WHERE grade = ?
            """, (
                grade.value,
            )).fetchone()

        return int(row["total"])

    def _searchable_rows(
        self,
        include_review: bool,
        collection_id: str = "default",
    ) -> list[sqlite3.Row]:
        """Return boards eligible for retrieval."""

        if include_review:
            return self._connection.execute("""
                SELECT *
                FROM boards
                WHERE collection_id = ?
                  AND grade IN (
                      'accept',
                      'review'
                  )
            """, (
                collection_id,
            )).fetchall()

        return self._connection.execute("""
            SELECT *
            FROM boards
            WHERE collection_id = ?
              AND grade = 'accept'
        """, (
            collection_id,
        )).fetchall()

    def close(self) -> None:
        """Close the database connection."""
        self._connection.close()

    def __enter__(self) -> "LumberYard":
        return self

    def __exit__(
        self,
        exc_type,
        exc,
        traceback,
    ) -> None:
        self.close()

    @staticmethod
    def _to_board(
        row: sqlite3.Row,
    ) -> StoredBoard:
        """Convert a database row into typed lumber."""

        return StoredBoard(
            board_id=row["board_id"],
            collection_id=row["collection_id"],
            log_id=row["log_id"],
            source_name=row["source_name"],
            text=row["text"],
            start_offset=row["start_offset"],
            end_offset=row["end_offset"],
            grade=Grade(row["grade"]),
            reasons=tuple(
                json.loads(
                    row["reasons_json"]
                )
            ),
            created_at=row["created_at"],
        )
