"""Tests for Lumber Yard collection isolation."""

import sqlite3

from app.rag.boards import Board
from app.rag.embeddings import build_embedding
from app.rag.grading import Grade, Inspection
from app.rag.yard import LumberYard


def make_board(
    board_id: str,
    text: str,
) -> Board:
    """Create simple test lumber."""
    return Board(
        board_id=board_id,
        log_id=f"log-{board_id}",
        source_name="manual.txt",
        text=text,
        start_offset=0,
        end_offset=len(text),
    )


def test_default_collection_exists(tmp_path) -> None:
    with LumberYard(tmp_path / "yard.db") as yard:
        collection = yard.get_collection("default")

        assert collection is not None
        assert collection.collection_id == "default"


def test_collection_can_be_created(tmp_path) -> None:
    with LumberYard(tmp_path / "yard.db") as yard:
        yard.create_collection(
            "oak",
            "Oak Manuals",
        )

        collection = yard.get_collection("oak")

        assert collection is not None
        assert collection.name == "Oak Manuals"


def test_board_can_belong_to_collection(tmp_path) -> None:
    with LumberYard(tmp_path / "yard.db") as yard:
        yard.create_collection(
            "oak",
            "Oak Manuals",
        )

        board = make_board(
            "oak-board",
            "Inspect the hydraulic pump regularly.",
        )

        yard.add_board(
            board,
            Inspection(Grade.ACCEPT, ()),
            collection_id="oak",
        )

        stored = yard.get_board("oak-board")

        assert stored is not None
        assert stored.collection_id == "oak"


def test_vector_search_isolates_collections(tmp_path) -> None:
    with LumberYard(tmp_path / "yard.db") as yard:
        yard.create_collection("oak", "Oak")
        yard.create_collection("hickory", "Hickory")

        oak = make_board(
            "oak-board",
            "Hydraulic pump maintenance.",
        )

        hickory = make_board(
            "hickory-board",
            "Hydraulic pump maintenance.",
        )

        yard.add_board(
            oak,
            Inspection(Grade.ACCEPT, ()),
            collection_id="oak",
        )

        yard.add_board(
            hickory,
            Inspection(Grade.ACCEPT, ()),
            collection_id="hickory",
        )

        yard.add_embedding(
            "oak-board",
            build_embedding(
                [1.0, 0.0],
                "test-model",
            ),
        )

        yard.add_embedding(
            "hickory-board",
            build_embedding(
                [1.0, 0.0],
                "test-model",
            ),
        )

        query = build_embedding(
            [1.0, 0.0],
            "test-model",
        )

        results = yard.vector_search(
            query,
            collection_id="oak",
        )

        assert len(results) == 1
        assert results[0].board.board_id == "oak-board"


def test_old_database_is_migrated_to_default_collection(
    tmp_path,
) -> None:
    database = tmp_path / "legacy.db"

    connection = sqlite3.connect(database)

    connection.execute("""
        CREATE TABLE boards (
            board_id TEXT PRIMARY KEY,
            log_id TEXT NOT NULL,
            source_name TEXT NOT NULL,
            text TEXT NOT NULL,
            start_offset INTEGER NOT NULL,
            end_offset INTEGER NOT NULL,
            grade TEXT NOT NULL,
            reasons_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.execute("""
        INSERT INTO boards (
            board_id,
            log_id,
            source_name,
            text,
            start_offset,
            end_offset,
            grade,
            reasons_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        "legacy-board",
        "legacy-log",
        "legacy.txt",
        "Old lumber survives migration.",
        0,
        30,
        "accept",
        "[]",
    ))

    connection.commit()
    connection.close()

    with LumberYard(database) as yard:
        board = yard.get_board("legacy-board")

        assert board is not None
        assert board.collection_id == "default"
        assert board.text == "Old lumber survives migration."


def test_lexical_search_isolates_collections(
    tmp_path,
) -> None:
    with LumberYard(tmp_path / "yard.db") as yard:
        yard.create_collection("oak", "Oak")
        yard.create_collection("hickory", "Hickory")

        oak = make_board(
            "oak-lexical",
            "Hydraulic pump maintenance instructions.",
        )

        hickory = make_board(
            "hickory-lexical",
            "Hydraulic pump maintenance instructions.",
        )

        yard.add_board(
            oak,
            Inspection(Grade.ACCEPT, ()),
            collection_id="oak",
        )

        yard.add_board(
            hickory,
            Inspection(Grade.ACCEPT, ()),
            collection_id="hickory",
        )

        results = yard.search(
            "hydraulic pump",
            collection_id="oak",
        )

        assert len(results) == 1
        assert results[0].board.board_id == "oak-lexical"
        assert results[0].board.collection_id == "oak"
