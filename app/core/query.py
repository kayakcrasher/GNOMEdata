"""Offline-first query engine for GNOMEdata."""

from dataclasses import dataclass

from app.rag.embeddings import Embedder
from app.rag.yard import LumberYard


@dataclass(frozen=True)
class QueryEvidence:
    board_id: str
    source_name: str
    text: str
    score: float


@dataclass(frozen=True)
class QueryResult:
    question: str
    collection_id: str
    evidence: tuple[QueryEvidence, ...]
    online_used: bool = False


class LocalQueryEngine:
    """Search GNOMEdata without requiring network access."""

    def __init__(
        self,
        yard: LumberYard,
        embedder: Embedder,
    ) -> None:
        self.yard = yard
        self.embedder = embedder

    def query(
        self,
        question: str,
        collection_id: str = "default",
        limit: int = 5,
        board_ids: set[str] | None = None,
        allow_online: bool = False,
    ) -> QueryResult:
        question = question.strip()

        if not question:
            raise ValueError("question cannot be empty.")

        if limit < 1:
            raise ValueError("limit must be positive.")

        if self.yard.get_collection(collection_id) is None:
            raise KeyError(
                f"unknown collection: {collection_id}"
            )

        # Core querying is always local.
        # allow_online is deliberately retained in the API contract
        # for a future explicitly invoked online provider.
        query_embedding = self.embedder.embed(question)

        results = self.yard.hybrid_search(
            question,
            query_embedding,
            limit=max(limit, 20) if board_ids else limit,
            collection_id=collection_id,
        )

        if board_ids is not None:
            results = [
                result
                for result in results
                if result.board.board_id in board_ids
            ]

        results = results[:limit]

        evidence = tuple(
            QueryEvidence(
                board_id=result.board.board_id,
                source_name=result.board.source_name,
                text=result.board.text,
                score=result.score,
            )
            for result in results
        )

        return QueryResult(
            question=question,
            collection_id=collection_id,
            evidence=evidence,
            online_used=False,
        )
