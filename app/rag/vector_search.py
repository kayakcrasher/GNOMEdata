"""Vector search for the GNOMEdata Lumber Yard.

The Lumber Yard stores finished boards in vector space.

When an order arrives, its query embedding is compared with candidate
boards using cosine similarity. The closest compatible lumber is
returned first.

This module performs ranking only. It does not generate embeddings,
store boards, or generate answers.
"""

from dataclasses import dataclass
from collections.abc import Iterable

from app.rag.embeddings import (
    Embedding,
    validate_compatible,
)
from app.rag.similarity import cosine_similarity


@dataclass(frozen=True)
class VectorBoard:
    """A board prepared for placement in the vector Lumber Yard."""

    board_id: str
    embedding: Embedding


@dataclass(frozen=True)
class VectorMatch:
    """One board retrieved from the Lumber Yard."""

    board_id: str
    similarity: float


def search_vectors(
    query: Embedding,
    boards: Iterable[VectorBoard],
    limit: int = 5,
    minimum_similarity: float = -1.0,
) -> list[VectorMatch]:
    """Rank compatible boards by cosine similarity.

    Boards using a different embedding model or vector dimension
    are rejected rather than silently compared.

    minimum_similarity allows the caller to keep weak matches
    out of the shipment.
    """

    if limit < 1:
        raise ValueError(
            "limit must be positive."
        )

    if not -1.0 <= minimum_similarity <= 1.0:
        raise ValueError(
            "minimum_similarity must be between -1 and 1."
        )

    matches: list[VectorMatch] = []

    for board in boards:
        if not board.board_id.strip():
            raise ValueError(
                "board_id cannot be empty."
            )

        validate_compatible(
            query,
            board.embedding,
        )

        score = cosine_similarity(
            query.vector,
            board.embedding.vector,
        )

        if score < minimum_similarity:
            continue

        matches.append(
            VectorMatch(
                board_id=board.board_id,
                similarity=score,
            )
        )

    matches.sort(
        key=lambda match: (
            -match.similarity,
            match.board_id,
        )
    )

    return matches[:limit]
