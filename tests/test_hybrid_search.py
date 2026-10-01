"""Tests for GNOMEdata hybrid retrieval."""

from app.rag.hybrid_search import HybridCandidate, rank_hybrid


def test_hybrid_ranking_combines_lexical_and_vector_scores() -> None:
    candidates = [
        HybridCandidate(
            board_id="balanced",
            lexical_score=0.8,
            vector_score=0.8,
        ),
        HybridCandidate(
            board_id="vector-only",
            lexical_score=0.0,
            vector_score=0.9,
        ),
        HybridCandidate(
            board_id="lexical-only",
            lexical_score=0.9,
            vector_score=0.0,
        ),
    ]

    results = rank_hybrid(candidates)

    assert results[0].board_id == "balanced"
    assert results[0].score > results[1].score


def test_hybrid_weights_can_favor_vector_similarity() -> None:
    candidates = [
        HybridCandidate(
            board_id="lexical",
            lexical_score=1.0,
            vector_score=0.2,
        ),
        HybridCandidate(
            board_id="semantic",
            lexical_score=0.2,
            vector_score=1.0,
        ),
    ]

    results = rank_hybrid(
        candidates,
        lexical_weight=0.2,
        vector_weight=0.8,
    )

    assert results[0].board_id == "semantic"


def test_hybrid_weights_must_sum_to_one() -> None:
    candidates = [
        HybridCandidate(
            board_id="board-1",
            lexical_score=1.0,
            vector_score=1.0,
        )
    ]

    try:
        rank_hybrid(
            candidates,
            lexical_weight=0.8,
            vector_weight=0.8,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Expected invalid weights to raise ValueError."
        )


def test_hybrid_ranking_respects_limit() -> None:
    candidates = [
        HybridCandidate(
            board_id=f"board-{index}",
            lexical_score=0.5,
            vector_score=0.5,
        )
        for index in range(10)
    ]

    results = rank_hybrid(
        candidates,
        limit=3,
    )

    assert len(results) == 3


from app.rag.hybrid_search import merge_candidates


def test_merge_candidates_combines_overlapping_results() -> None:
    lexical = {
        "board-a": 0.9,
        "board-b": 0.7,
    }

    vector = {
        "board-b": 0.8,
        "board-c": 0.95,
    }

    candidates = merge_candidates(
        lexical,
        vector,
    )

    by_id = {
        candidate.board_id: candidate
        for candidate in candidates
    }

    assert set(by_id) == {
        "board-a",
        "board-b",
        "board-c",
    }

    assert by_id["board-b"].lexical_score == 0.7
    assert by_id["board-b"].vector_score == 0.8


def test_merge_candidates_fills_missing_signal_with_zero() -> None:
    candidates = merge_candidates(
        {"lexical-only": 0.8},
        {"vector-only": 0.9},
    )

    by_id = {
        candidate.board_id: candidate
        for candidate in candidates
    }

    assert by_id["lexical-only"].vector_score == 0.0
    assert by_id["vector-only"].lexical_score == 0.0


def test_merge_candidates_is_deterministic() -> None:
    candidates = merge_candidates(
        {
            "charlie": 0.5,
            "alpha": 0.5,
        },
        {
            "bravo": 0.5,
            "alpha": 0.5,
        },
    )

    assert [
        candidate.board_id
        for candidate in candidates
    ] == [
        "alpha",
        "bravo",
        "charlie",
    ]


from app.rag.hybrid_search import normalize_scores


def test_normalize_scores_scales_range_zero_to_one() -> None:
    scores = {
        "low": 2.0,
        "middle": 4.0,
        "high": 6.0,
    }

    normalized = normalize_scores(scores)

    assert normalized["low"] == 0.0
    assert normalized["middle"] == 0.5
    assert normalized["high"] == 1.0


def test_normalize_scores_handles_equal_values() -> None:
    normalized = normalize_scores(
        {
            "alpha": 5.0,
            "bravo": 5.0,
        }
    )

    assert normalized == {
        "alpha": 1.0,
        "bravo": 1.0,
    }


def test_normalize_scores_handles_empty_input() -> None:
    assert normalize_scores({}) == {}


def test_lumber_yard_hybrid_search_combines_retrieval_signals(
    tmp_path,
) -> None:
    from app.rag.boards import Board
    from app.rag.embeddings import build_embedding
    from app.rag.grading import Grade, Inspection
    from app.rag.yard import LumberYard

    def board(
        board_id: str,
        text: str,
    ) -> Board:
        return Board(
            board_id=board_id,
            log_id=f"log-{board_id}",
            source_name="manual.txt",
            text=text,
            start_offset=0,
            end_offset=len(text),
        )

    with LumberYard(tmp_path / "yard.db") as yard:
        balanced = board(
            "balanced",
            "Hydraulic pump maintenance procedure.",
        )

        lexical = board(
            "lexical",
            "Hydraulic pump maintenance procedure.",
        )

        semantic = board(
            "semantic",
            "Service the fluid pressure equipment.",
        )

        for item in (
            balanced,
            lexical,
            semantic,
        ):
            yard.add_board(
                item,
                Inspection(Grade.ACCEPT, ()),
            )

        yard.add_embedding(
            "balanced",
            build_embedding(
                [0.8, 0.6],
                "hybrid-model",
            ),
        )

        yard.add_embedding(
            "lexical",
            build_embedding(
                [0.6, 0.8],
                "hybrid-model",
            ),
        )

        yard.add_embedding(
            "semantic",
            build_embedding(
                [1.0, 0.0],
                "hybrid-model",
            ),
        )

        query = build_embedding(
            [1.0, 0.0],
            "hybrid-model",
        )

        results = yard.hybrid_search(
            "hydraulic pump maintenance",
            query,
            limit=3,
        )

        assert len(results) == 3

        assert {
            result.board.board_id
            for result in results
        } == {
            "balanced",
            "lexical",
            "semantic",
        }

        assert (
            results[0].board.board_id
            == "balanced"
        )
