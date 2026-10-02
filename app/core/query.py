"""Offline-first query engine for GNOMEdata."""

from dataclasses import dataclass

from app.brain.entities import extract_entities
from app.brain.facts import Fact, extract_facts
from app.rag.embeddings import Embedder
from app.rag.yard import LumberYard


@dataclass(frozen=True)
class QueryEvidence:
    """One board supporting a query result."""

    board_id: str
    source_name: str
    text: str
    score: float


@dataclass(frozen=True)
class QueryFact:
    """A structured fact discovered in retrieved evidence."""

    subject: str
    relation: str
    value: str
    source_name: str
    board_id: str


@dataclass(frozen=True)
class QueryResult:
    """Complete local GNOMEdata query result."""

    question: str
    collection_id: str
    evidence: tuple[QueryEvidence, ...]
    answer: str | None = None
    facts: tuple[QueryFact, ...] = ()
    online_used: bool = False


class LocalQueryEngine:
    """Search and reason over GNOMEdata without network access."""

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
        """Retrieve evidence and reason over it locally."""

        question = question.strip()

        if not question:
            raise ValueError("question cannot be empty.")

        if limit < 1:
            raise ValueError("limit must be positive.")

        if self.yard.get_collection(collection_id) is None:
            raise KeyError(
                f"unknown collection: {collection_id}"
            )

        # Retrieval stays entirely local.
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

        facts = self._collect_facts(evidence)

        answer = self._answer_from_facts(
            question,
            facts,
        )

        return QueryResult(
            question=question,
            collection_id=collection_id,
            evidence=evidence,
            answer=answer,
            facts=facts,
            online_used=False,
        )

    @staticmethod
    def _collect_facts(
        evidence: tuple[QueryEvidence, ...],
    ) -> tuple[QueryFact, ...]:
        """Extract structured facts from retrieved boards."""

        collected: list[QueryFact] = []
        seen: set[tuple[str, str, str]] = set()

        for item in evidence:
            for fact in extract_facts(item.text):
                key = (
                    fact.subject.casefold(),
                    fact.relation.casefold(),
                    fact.value.casefold(),
                )

                if key in seen:
                    continue

                seen.add(key)

                collected.append(
                    QueryFact(
                        subject=fact.subject,
                        relation=fact.relation,
                        value=fact.value,
                        source_name=item.source_name,
                        board_id=item.board_id,
                    )
                )

        return tuple(collected)

    @staticmethod
    def _answer_from_facts(
        question: str,
        facts: tuple[QueryFact, ...],
    ) -> str | None:
        """Produce a conservative answer from explicit local facts."""

        if not facts:
            return None

        question_lower = question.casefold()

        # Identify subjects explicitly mentioned by the user.
        matching = [
            fact
            for fact in facts
            if fact.subject.casefold() in question_lower
        ]

        if not matching:
            matching = list(facts)

        # Location questions.
        if (
            question_lower.startswith("where")
            or "where " in question_lower
        ):
            location_facts = [
                fact
                for fact in matching
                if fact.relation in {
                    "lives_at",
                    "sells",
                }
            ]

            if location_facts:
                fact = location_facts[0]

                if fact.relation == "lives_at":
                    return (
                        f"{fact.subject} lives at "
                        f"{fact.value}."
                    )

                if fact.relation == "sells":
                    return (
                        f"{fact.subject} sells "
                        f"{fact.value}."
                    )

        # Pet/name questions.
        if any(
            word in question_lower
            for word in ("dog", "cat", "pet", "named", "name")
        ):
            named_facts = [
                fact
                for fact in matching
                if fact.relation == "has_named"
            ]

            if named_facts:
                fact = named_facts[0]
                return (
                    f"{fact.subject} has one named "
                    f"{fact.value}."
                )

        # General subject summary.
        if matching:
            statements: list[str] = []

            for fact in matching[:4]:
                if fact.relation == "lives_at":
                    statements.append(
                        f"{fact.subject} lives at {fact.value}."
                    )
                elif fact.relation == "sells":
                    statements.append(
                        f"{fact.subject} sells {fact.value}."
                    )
                elif fact.relation == "has_named":
                    statements.append(
                        f"{fact.subject} has one named {fact.value}."
                    )

            if statements:
                return " ".join(statements)

        return None
