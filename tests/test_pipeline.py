"""Tests for the GNOMEdata Mill Controller."""

import pytest

from app.rag.boards import Log
from app.rag.embeddings import StaticEmbedder
from app.rag.grading import Grade
from app.rag.pipeline import MillPipeline
from app.rag.yard import LumberYard


def make_pipeline(
    tmp_path,
    vectors,
    *,
    board_size=800,
    overlap=120,
    embed_review=True,
):
    """Build a deterministic test mill."""

    yard = LumberYard(
        tmp_path / "yard.db"
    )

    embedder = StaticEmbedder(
        vectors,
        model_name="test-mill-model",
    )

    pipeline = MillPipeline(
        yard=yard,
        embedder=embedder,
        board_size=board_size,
        overlap=overlap,
        embed_review=embed_review,
    )

    return yard, pipeline


def test_accepted_board_runs_through_mill(
    tmp_path,
) -> None:
    text = (
        "Inspect the hydraulic pump and replace "
        "the filter during scheduled maintenance."
    )

    yard, pipeline = make_pipeline(
        tmp_path,
        {
            text: [1.0, 0.0, 0.0],
        },
    )

    try:
        report = pipeline.process(
            Log(
                log_id="manual-1",
                source_name="manual.txt",
                text=text,
            )
        )

        assert report.total == 1
        assert report.accepted == 1
        assert report.review == 0
        assert report.rejected == 0
        assert report.embedded == 1

        stored = yard.get_board(
            "manual-1:board:0"
        )

        assert stored is not None
        assert stored.grade == Grade.ACCEPT

        embedding = yard.get_embedding(
            "manual-1:board:0",
            "test-mill-model",
        )

        assert embedding is not None

    finally:
        yard.close()


def test_stacker_result_is_preserved_in_report(
    tmp_path,
) -> None:
    text = (
        "Inspect and service the machine during "
        "scheduled maintenance."
    )

    yard, pipeline = make_pipeline(
        tmp_path,
        {
            text: [1.0, 0.0],
        },
    )

    try:
        report = pipeline.process(
            Log(
                "stack-log",
                "stack.txt",
                text,
            )
        )

        stacks = {
            stack.name
            for stack
            in report.boards[0].stacking.stacks
        }

        assert "maintenance" in stacks

    finally:
        yard.close()


def test_review_board_can_be_embedded(
    tmp_path,
) -> None:
    text = "Inspect pump."

    yard, pipeline = make_pipeline(
        tmp_path,
        {
            text: [1.0, 0.0],
        },
        embed_review=True,
    )

    try:
        report = pipeline.process(
            Log(
                "review-log",
                "review.txt",
                text,
            )
        )

        assert report.review == 1
        assert report.embedded == 1

        assert yard.get_embedding(
            "review-log:board:0",
            "test-mill-model",
        ) is not None

    finally:
        yard.close()


def test_review_embedding_can_be_disabled(
    tmp_path,
) -> None:
    text = "Inspect pump."

    yard, pipeline = make_pipeline(
        tmp_path,
        {},
        embed_review=False,
    )

    try:
        report = pipeline.process(
            Log(
                "review-log",
                "review.txt",
                text,
            )
        )

        assert report.review == 1
        assert report.embedded == 0

        assert yard.get_embedding(
            "review-log:board:0",
            "test-mill-model",
        ) is None

    finally:
        yard.close()


def test_rejected_board_is_never_embedded(
    tmp_path,
) -> None:
    text = "   "

    yard, pipeline = make_pipeline(
        tmp_path,
        {},
    )

    try:
        report = pipeline.process(
            Log(
                "empty-log",
                "empty.txt",
                text,
            )
        )

        assert report.total == 0
        assert report.embedded == 0
        assert yard.count() == 0

    finally:
        yard.close()


def test_multiple_boards_are_processed(
    tmp_path,
) -> None:
    text = (
        "Pump maintenance instructions explain "
        "inspection and filter replacement. "
        "Safety procedures explain machine "
        "shutdown before repair work."
    )

    # Discover exactly what the cutter will produce
    # using a first deterministic pass.
    from app.rag.boards import cut_boards

    log = Log(
        "multi-log",
        "multi.txt",
        text,
    )

    boards = cut_boards(
        log,
        board_size=75,
        overlap=15,
    )

    vectors = {
        board.text: [
            float(index + 1),
            1.0,
        ]
        for index, board in enumerate(boards)
    }

    yard, pipeline = make_pipeline(
        tmp_path,
        vectors,
        board_size=75,
        overlap=15,
    )

    try:
        report = pipeline.process(log)

        assert report.total == len(boards)
        assert yard.count() == len(boards)

        assert all(
            item.embedded
            for item in report.boards
        )

    finally:
        yard.close()


def test_pipeline_vectors_are_searchable(
    tmp_path,
) -> None:
    text = (
        "Hydraulic pump maintenance requires "
        "regular inspection and filter service."
    )

    yard, pipeline = make_pipeline(
        tmp_path,
        {
            text: [1.0, 0.0, 0.0],
            "hydraulic service question": [
                0.95,
                0.05,
                0.0,
            ],
        },
    )

    try:
        pipeline.process(
            Log(
                "search-log",
                "search.txt",
                text,
            )
        )

        query = pipeline.embedder.embed(
            "hydraulic service question"
        )

        results = yard.vector_search(
            query,
        )

        assert results
        assert (
            results[0].board.board_id
            == "search-log:board:0"
        )

        assert results[0].score > 0.9

    finally:
        yard.close()


def test_pipeline_rejects_invalid_board_size(
    tmp_path,
) -> None:
    yard = LumberYard(
        tmp_path / "yard.db"
    )

    try:
        with pytest.raises(
            ValueError,
            match="board_size",
        ):
            MillPipeline(
                yard=yard,
                embedder=StaticEmbedder({}),
                board_size=0,
            )

    finally:
        yard.close()


def test_pipeline_rejects_invalid_overlap(
    tmp_path,
) -> None:
    yard = LumberYard(
        tmp_path / "yard.db"
    )

    try:
        with pytest.raises(
            ValueError,
            match="overlap",
        ):
            MillPipeline(
                yard=yard,
                embedder=StaticEmbedder({}),
                board_size=100,
                overlap=100,
            )

    finally:
        yard.close()


def test_pipeline_preserves_source_provenance(
    tmp_path,
) -> None:
    text = (
        "Inspect hydraulic equipment during "
        "scheduled maintenance operations."
    )

    yard, pipeline = make_pipeline(
        tmp_path,
        {
            text: [1.0, 0.0],
        },
    )

    try:
        pipeline.process(
            Log(
                "provenance-log",
                "hydraulics-manual.txt",
                text,
            )
        )

        stored = yard.get_board(
            "provenance-log:board:0"
        )

        assert stored is not None
        assert stored.log_id == "provenance-log"
        assert stored.source_name == "hydraulics-manual.txt"
        assert stored.text == text
        assert stored.start_offset == 0
        assert stored.end_offset == len(text)

    finally:
        yard.close()


def test_reprocessing_same_log_updates_without_duplicates(
    tmp_path,
) -> None:
    text = (
        "Inspect the hydraulic pump during "
        "scheduled maintenance."
    )

    yard, pipeline = make_pipeline(
        tmp_path,
        {
            text: [1.0, 0.0],
        },
    )

    log = Log(
        "repeat-log",
        "repeat.txt",
        text,
    )

    try:
        pipeline.process(log)
        pipeline.process(log)

        assert yard.count() == 1

        stored = yard.get_embedding(
            "repeat-log:board:0",
            "test-mill-model",
        )

        assert stored is not None

    finally:
        yard.close()


def test_review_lumber_requires_explicit_search_permission(
    tmp_path,
) -> None:
    text = "Inspect pump."

    yard, pipeline = make_pipeline(
        tmp_path,
        {
            text: [1.0, 0.0],
            "pump question": [1.0, 0.0],
        },
        embed_review=True,
    )

    try:
        pipeline.process(
            Log(
                "review-search",
                "review.txt",
                text,
            )
        )

        query = pipeline.embedder.embed(
            "pump question"
        )

        assert yard.vector_search(query) == []

        results = yard.vector_search(
            query,
            include_review=True,
        )

        assert len(results) == 1
        assert (
            results[0].board.board_id
            == "review-search:board:0"
        )

    finally:
        yard.close()


def test_empty_log_produces_empty_report(
    tmp_path,
) -> None:
    yard, pipeline = make_pipeline(
        tmp_path,
        {},
    )

    try:
        report = pipeline.process(
            Log(
                "nothing",
                "empty.txt",
                "",
            )
        )

        assert report.total == 0
        assert report.accepted == 0
        assert report.review == 0
        assert report.rejected == 0
        assert report.embedded == 0
        assert report.boards == ()
        assert yard.count() == 0

    finally:
        yard.close()


def test_milled_vectors_survive_full_yard_reopening(
    tmp_path,
) -> None:
    database = tmp_path / "persistent-yard.db"

    text = (
        "Hydraulic maintenance requires regular "
        "inspection and filter replacement."
    )

    vectors = {
        text: [1.0, 0.0, 0.0],
        "maintenance order": [0.95, 0.05, 0.0],
    }

    with LumberYard(database) as yard:
        pipeline = MillPipeline(
            yard=yard,
            embedder=StaticEmbedder(
                vectors,
                model_name="persistent-model",
            ),
        )

        pipeline.process(
            Log(
                "persistent-log",
                "manual.txt",
                text,
            )
        )

    with LumberYard(database) as yard:
        embedder = StaticEmbedder(
            vectors,
            model_name="persistent-model",
        )

        query = embedder.embed(
            "maintenance order"
        )

        results = yard.vector_search(query)

        assert len(results) == 1
        assert (
            results[0].board.board_id
            == "persistent-log:board:0"
        )
        assert results[0].score > 0.9
