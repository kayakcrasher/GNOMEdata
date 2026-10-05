"""End-to-end document question answering for GNOMEdata."""

from dataclasses import dataclass

from app.inference.router import InferenceRouter
from app.rag.answer_engine import (
    AnswerEngine,
    Evidence,
)
from app.rag.embeddings import Embedder
from app.rag.yard import LumberYard


@dataclass(frozen=True)
class DocumentQAResult:
    """Result of retrieval followed by grounded inference."""

    question: str
    answer: str
    provider: str
    model: str
    evidence: tuple[Evidence, ...]


class DocumentQA:
    """Retrieve relevant lumber and answer from it."""

    def __init__(
        self,
        yard: LumberYard,
        embedder: Embedder,
        router: InferenceRouter,
        provider: str,
        *,
        collection_id: str = "default",
        retrieval_limit: int = 4,
        lexical_weight: float = 0.5,
        vector_weight: float = 0.5,
    ) -> None:
        if retrieval_limit < 1:
            raise ValueError(
                "retrieval_limit must be positive."
            )

        self.yard = yard
        self.embedder = embedder
        self.collection_id = collection_id
        self.retrieval_limit = retrieval_limit
        self.lexical_weight = lexical_weight
        self.vector_weight = vector_weight

        self.answer_engine = AnswerEngine(
            router=router,
            provider=provider,
            max_evidence=retrieval_limit,
        )

    def retrieve(
        self,
        question: str,
    ) -> list[Evidence]:
        """Retrieve evidence relevant to a question."""

        question = question.strip()

        if not question:
            raise ValueError(
                "question cannot be empty."
            )

        query_embedding = self.embedder.embed(
            question
        )

        results = self.yard.hybrid_search(
            query_text=question,
            query_embedding=query_embedding,
            limit=self.retrieval_limit,
            lexical_weight=self.lexical_weight,
            vector_weight=self.vector_weight,
            collection_id=self.collection_id,
        )

        return [
            Evidence(
                text=result.board.text,
                source=result.board.source_name,
                score=result.score,
            )
            for result in results
        ]

    def ask(
        self,
        question: str,
        *,
        max_tokens: int = 128,
        temperature: float = 0.1,
    ) -> DocumentQAResult:
        """Retrieve evidence and generate a grounded answer."""

        evidence = self.retrieve(question)

        answer = self.answer_engine.answer(
            question,
            evidence,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return DocumentQAResult(
            question=question.strip(),
            answer=answer.text,
            provider=answer.provider,
            model=answer.model,
            evidence=answer.evidence,
        )
