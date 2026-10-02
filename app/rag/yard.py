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
from app.rag.hybrid_search import (
    merge_candidates,
    normalize_scores,
    rank_hybrid,
)
from app.rag.stacker import Stack, StackedBoard
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
            self.database,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row

        # Enforce foreign keys and ON DELETE CASCADE.
        self._connection.execute("PRAGMA foreign_keys = ON")

        self._create_schema()
        self._migrate_collections()
        self._create_stack_schema()

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

    def _create_stack_schema(self) -> None:
        """Create durable storage for board stack assignments."""

        with self._connection:
            self._connection.execute("""
                CREATE TABLE IF NOT EXISTS board_stacks (
                    board_id TEXT NOT NULL,
                    stack_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    matched_terms_json TEXT NOT NULL
                        DEFAULT '[]',

                    PRIMARY KEY (
                        board_id,
                        stack_name
                    ),

                    FOREIGN KEY (board_id)
                    REFERENCES boards(board_id)
                    ON DELETE CASCADE,

                    CHECK (
                        confidence >= 0.0
                        AND confidence <= 1.0
                    )
                )
            """)

            self._connection.execute("""
                CREATE INDEX IF NOT EXISTS
                    idx_board_stacks_name
                ON board_stacks(stack_name)
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

    def clear_collection(
        self,
        collection_id: str,
    ) -> int:
        """Remove all lumber from a collection and return the number removed."""

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
            cursor = self._connection.execute(
                "DELETE FROM boards WHERE collection_id = ?",
                (collection_id,),
            )

        return cursor.rowcount


    def list_boards(
        self,
        collection_id: str = "default",
    ) -> list[StoredBoard]:
        """Return all boards belonging to one collection."""

        collection_id = collection_id.strip()

        if not collection_id:
            raise ValueError(
                "collection_id cannot be empty."
            )

        if self.get_collection(collection_id) is None:
            raise KeyError(
                f"unknown collection: {collection_id}"
            )

        rows = self._connection.execute("""
            SELECT *
            FROM boards
            WHERE collection_id = ?
            ORDER BY created_at, board_id
        """, (
            collection_id,
        )).fetchall()

        return [
            self._to_board(row)
            for row in rows
        ]

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

    def set_stacks(
        self,
        stacking: StackedBoard,
    ) -> None:
        """Replace one board's persistent stack assignments."""

        board_id = stacking.board_id.strip()

        if not board_id:
            raise ValueError(
                "board_id cannot be empty."
            )

        if self.get_board(board_id) is None:
            raise KeyError(
                f"unknown board: {board_id}"
            )

        with self._connection:
            self._connection.execute("""
                DELETE FROM board_stacks
                WHERE board_id = ?
            """, (
                board_id,
            ))

            for stack in stacking.stacks:
                if not stack.name.strip():
                    raise ValueError(
                        "stack name cannot be empty."
                    )

                if not 0.0 <= stack.confidence <= 1.0:
                    raise ValueError(
                        "stack confidence must be "
                        "between 0 and 1."
                    )

                self._connection.execute("""
                    INSERT INTO board_stacks (
                        board_id,
                        stack_name,
                        confidence,
                        matched_terms_json
                    )
                    VALUES (?, ?, ?, ?)
                """, (
                    board_id,
                    stack.name,
                    stack.confidence,
                    json.dumps(
                        stack.matched_terms
                    ),
                ))

    def get_stacks(
        self,
        board_id: str,
    ) -> StackedBoard:
        """Recover persistent stack assignments for one board."""

        if not board_id.strip():
            raise ValueError(
                "board_id cannot be empty."
            )

        if self.get_board(board_id) is None:
            raise KeyError(
                f"unknown board: {board_id}"
            )

        rows = self._connection.execute("""
            SELECT
                stack_name,
                confidence,
                matched_terms_json
            FROM board_stacks
            WHERE board_id = ?
            ORDER BY
                confidence DESC,
                stack_name ASC
        """, (
            board_id,
        )).fetchall()

        stacks = tuple(
            Stack(
                name=row["stack_name"],
                confidence=float(
                    row["confidence"]
                ),
                matched_terms=tuple(
                    json.loads(
                        row["matched_terms_json"]
                    )
                ),
            )
            for row in rows
        )

        return StackedBoard(
            board_id=board_id,
            stacks=stacks,
        )

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
        stack_name: str | None = None,
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
            stack_name,
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
        stack_name: str | None = None,
    ) -> list[SearchResult]:
        """Retrieve boards from the Lumber Yard by vector similarity."""

        if limit < 1:
            raise ValueError(
                "limit must be positive."
            )

        parameters: list[object] = [
            query.model,
            collection_id,
            (
                "review"
                if include_review
                else "accept"
            ),
        ]

        stack_clause = ""

        if stack_name is not None:
            stack_name = stack_name.strip()

            if not stack_name:
                raise ValueError(
                    "stack_name cannot be empty."
                )

            stack_clause = """
              AND EXISTS (
                  SELECT 1
                  FROM board_stacks AS s
                  WHERE s.board_id = b.board_id
                    AND s.stack_name = ?
              )
            """

            parameters.append(stack_name)

        rows = self._connection.execute(
            f"""
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
              {stack_clause}
            """,
            parameters,
        ).fetchall()

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

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: Embedding,
        limit: int = 5,
        lexical_weight: float = 0.5,
        vector_weight: float = 0.5,
        minimum_similarity: float = -1.0,
        include_review: bool = False,
        collection_id: str = "default",
        stack_name: str | None = None,
    ) -> list[SearchResult]:
        """Retrieve evidence using lexical and semantic signals."""

        if limit < 1:
            raise ValueError(
                "limit must be positive."
            )

        # Pull a wider candidate pool than the final result set.
        # Ranking needs enough lumber from both retrieval paths
        # to make a meaningful hybrid decision.
        candidate_limit = max(
            limit * 4,
            20,
        )

        lexical_results = self.search(
            query_text,
            limit=candidate_limit,
            include_review=include_review,
            collection_id=collection_id,
            stack_name=stack_name,
        )

        vector_results = self.vector_search(
            query_embedding,
            limit=candidate_limit,
            minimum_similarity=minimum_similarity,
            include_review=include_review,
            collection_id=collection_id,
            stack_name=stack_name,
        )

        lexical_scores = {
            result.board.board_id: result.score
            for result in lexical_results
        }

        vector_scores = {
            result.board.board_id: result.score
            for result in vector_results
        }

        normalized_lexical = normalize_scores(
            lexical_scores
        )

        normalized_vector = normalize_scores(
            vector_scores
        )

        candidates = merge_candidates(
            normalized_lexical,
            normalized_vector,
        )

        ranked = rank_hybrid(
            candidates,
            lexical_weight=lexical_weight,
            vector_weight=vector_weight,
            limit=limit,
        )

        boards: dict[str, StoredBoard] = {}

        for result in lexical_results:
            boards[result.board.board_id] = (
                result.board
            )

        for result in vector_results:
            boards[result.board.board_id] = (
                result.board
            )

        return [
            SearchResult(
                board=boards[result.board_id],
                score=result.score,
            )
            for result in ranked
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
        stack_name: str | None = None,
    ) -> list[sqlite3.Row]:
        """Return boards eligible for retrieval."""

        parameters: list[object] = [
            collection_id,
        ]

        grade_clause = (
            "grade IN ('accept', 'review')"
            if include_review
            else "grade = 'accept'"
        )

        stack_clause = ""

        if stack_name is not None:
            stack_name = stack_name.strip()

            if not stack_name:
                raise ValueError(
                    "stack_name cannot be empty."
                )

            stack_clause = """
                AND EXISTS (
                    SELECT 1
                    FROM board_stacks AS s
                    WHERE s.board_id = boards.board_id
                      AND s.stack_name = ?
                )
            """

            parameters.append(stack_name)

        return self._connection.execute(
            f"""
            SELECT *
            FROM boards
            WHERE collection_id = ?
              AND {grade_clause}
              {stack_clause}
            """,
            parameters,
        ).fetchall()

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
