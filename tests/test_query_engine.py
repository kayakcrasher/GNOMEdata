"""Tests for GNOMEdata's offline query engine."""

from app.core.query import LocalQueryEngine
from app.rag.boards import Log
from app.rag.local_embedder import LocalHashEmbedder
from app.rag.pipeline import MillPipeline
from app.rag.yard import LumberYard


def build_yard(tmp_path):
    yard = LumberYard(tmp_path / "query.db")
    embedder = LocalHashEmbedder()

    mill = MillPipeline(
        yard=yard,
        embedder=embedder,
    )

    mill.process(
        Log(
            log_id="randy",
            source_name="Piedealer",
            text=(
                "Randy lives on Willowstreet. "
                "He sells pies from his home in Oakville. "
                "He has a dog named Tom and a cat named Blake."
            ),
        )
    )

    mill.process(
        Log(
            log_id="history",
            source_name="OakvilleHistory",
            text=(
                "Oakville is a town in Northern Kentucky. "
                "A severe blizzard created a local tradition "
                "of eating pies."
            ),
        )
    )

    return yard, embedder


def test_query_finds_relevant_local_evidence(tmp_path):
    yard, embedder = build_yard(tmp_path)

    try:
        engine = LocalQueryEngine(yard, embedder)

        result = engine.query(
            "Where does Randy sell pies?"
        )

        assert result.evidence
        assert any(
            item.source_name == "Piedealer"
            for item in result.evidence
        )
        assert result.online_used is False
    finally:
        yard.close()


def test_query_can_filter_selected_boards(tmp_path):
    yard, embedder = build_yard(tmp_path)

    try:
        boards = yard.list_boards("default")

        selected = {
            board.board_id
            for board in boards
            if board.source_name == "OakvilleHistory"
        }

        engine = LocalQueryEngine(yard, embedder)

        result = engine.query(
            "Tell me about Oakville",
            board_ids=selected,
        )

        assert result.evidence

        assert all(
            item.board_id in selected
            for item in result.evidence
        )
    finally:
        yard.close()


def test_online_is_never_used_implicitly(tmp_path):
    yard, embedder = build_yard(tmp_path)

    try:
        engine = LocalQueryEngine(yard, embedder)

        result = engine.query(
            "What animals does Randy own?"
        )

        assert result.online_used is False
    finally:
        yard.close()


def test_query_rejects_empty_question(tmp_path):
    yard, embedder = build_yard(tmp_path)

    try:
        engine = LocalQueryEngine(yard, embedder)

        try:
            engine.query("   ")
        except ValueError as error:
            assert "question" in str(error)
        else:
            raise AssertionError(
                "empty question was accepted"
            )
    finally:
        yard.close()
