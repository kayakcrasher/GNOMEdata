"""Hybrid retrieval ranking for GNOMEdata.

Hybrid ranking combines two independent retrieval signals:

- lexical score: how strongly the words match
- vector score: how strongly the meanings match

This module ranks already-discovered candidates. It does not
decide which boards are eligible for retrieval.
"""

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class HybridCandidate:
    """One board carrying lexical and semantic evidence."""

    board_id: str
    lexical_score: float
    vector_score: float


@dataclass(frozen=True)
class HybridResult:
    """One board after hybrid scoring."""

    board_id: str
    lexical_score: float
    vector_score: float
    score: float


def rank_hybrid(
    candidates: Sequence[HybridCandidate],
    lexical_weight: float = 0.5,
    vector_weight: float = 0.5,
    limit: int = 5,
) -> list[HybridResult]:
    """Combine lexical and vector signals into one ranking.

    Weights must be finite, non-negative, and sum to 1.
    Higher final scores rank first.
    """

    if limit < 1:
        raise ValueError(
            "limit must be positive."
        )

    if not math.isfinite(lexical_weight):
        raise ValueError(
            "lexical_weight must be finite."
        )

    if not math.isfinite(vector_weight):
        raise ValueError(
            "vector_weight must be finite."
        )

    if lexical_weight < 0.0:
        raise ValueError(
            "lexical_weight cannot be negative."
        )

    if vector_weight < 0.0:
        raise ValueError(
            "vector_weight cannot be negative."
        )

    if not math.isclose(
        lexical_weight + vector_weight,
        1.0,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "hybrid weights must sum to 1."
        )

    results: list[HybridResult] = []

    for candidate in candidates:
        if not candidate.board_id.strip():
            raise ValueError(
                "board_id cannot be empty."
            )

        if not math.isfinite(
            candidate.lexical_score
        ):
            raise ValueError(
                "lexical_score must be finite."
            )

        if not math.isfinite(
            candidate.vector_score
        ):
            raise ValueError(
                "vector_score must be finite."
            )

        score = (
            lexical_weight
            * candidate.lexical_score
            + vector_weight
            * candidate.vector_score
        )

        results.append(
            HybridResult(
                board_id=candidate.board_id,
                lexical_score=candidate.lexical_score,
                vector_score=candidate.vector_score,
                score=score,
            )
        )

    results.sort(
        key=lambda result: (
            -result.score,
            result.board_id,
        )
    )

    return results[:limit]


def merge_candidates(
    lexical_scores: dict[str, float],
    vector_scores: dict[str, float],
) -> list[HybridCandidate]:
    """Merge lexical and vector candidates by board ID.

    The union of both candidate sets is returned.
    A board missing from one retrieval path receives
    a score of zero for that signal.

    Results are ordered by board ID so candidate generation
    remains deterministic before ranking.
    """

    board_ids = (
        set(lexical_scores)
        | set(vector_scores)
    )

    candidates: list[HybridCandidate] = []

    for board_id in sorted(board_ids):
        if not board_id.strip():
            raise ValueError(
                "board_id cannot be empty."
            )

        lexical_score = lexical_scores.get(
            board_id,
            0.0,
        )

        vector_score = vector_scores.get(
            board_id,
            0.0,
        )

        if not math.isfinite(lexical_score):
            raise ValueError(
                "lexical score must be finite."
            )

        if not math.isfinite(vector_score):
            raise ValueError(
                "vector score must be finite."
            )

        candidates.append(
            HybridCandidate(
                board_id=board_id,
                lexical_score=lexical_score,
                vector_score=vector_score,
            )
        )

    return candidates


def normalize_scores(
    scores: dict[str, float],
) -> dict[str, float]:
    """Normalize a score mapping into the range 0.0 through 1.0.

    Min-max normalization preserves relative ordering while
    placing different retrieval signals onto a common scale.

    When every score is equal, all entries receive 1.0 because
    that signal provides no basis for distinguishing candidates.
    """

    if not scores:
        return {}

    for board_id, score in scores.items():
        if not board_id.strip():
            raise ValueError(
                "board_id cannot be empty."
            )

        if not math.isfinite(score):
            raise ValueError(
                "score must be finite."
            )

    minimum = min(scores.values())
    maximum = max(scores.values())

    if math.isclose(
        minimum,
        maximum,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        return {
            board_id: 1.0
            for board_id in scores
        }

    score_range = maximum - minimum

    return {
        board_id: (
            (score - minimum)
            / score_range
        )
        for board_id, score in scores.items()
    }
