"""Grounded document answering for GNOMEdata."""

from dataclasses import dataclass

from app.inference.router import InferenceRouter
from app.inference.types import (
    InferenceRequest,
    InferenceResult,
)


@dataclass(frozen=True)
class Evidence:
    """One retrieved piece of supporting evidence."""

    text: str
    source: str
    score: float | None = None
    page: int | None = None


@dataclass(frozen=True)
class GroundedAnswer:
    """Answer plus the evidence used to produce it."""

    text: str
    provider: str
    model: str
    evidence: tuple[Evidence, ...]


class AnswerEngine:
    """Turn retrieved evidence into grounded model requests."""

    def __init__(
        self,
        router: InferenceRouter,
        provider: str,
        max_evidence: int = 4,
        minimum_score: float | None = None,
    ) -> None:
        if max_evidence < 1:
            raise ValueError(
                "max_evidence must be positive."
            )

        self.router = router
        self.provider = provider
        self.max_evidence = max_evidence
        self.minimum_score = minimum_score

    def select_evidence(
        self,
        evidence: list[Evidence],
    ) -> list[Evidence]:
        """Filter, rank, deduplicate, and limit evidence."""

        filtered = []

        for item in evidence:
            text = item.text.strip()

            if not text:
                continue

            if (
                self.minimum_score is not None
                and item.score is not None
                and item.score < self.minimum_score
            ):
                continue

            filtered.append(item)

        filtered.sort(
            key=lambda item: (
                -item.score
                if item.score is not None
                else 0.0
            )
        )

        selected = []
        seen = set()

        for item in filtered:
            fingerprint = " ".join(
                item.text.casefold().split()
            )

            if fingerprint in seen:
                continue

            seen.add(fingerprint)
            selected.append(item)

            if len(selected) >= self.max_evidence:
                break

        return selected

    def build_prompt(
        self,
        question: str,
        evidence: list[Evidence],
    ) -> str:
        """Build a compact grounded QA prompt."""

        selected = self.select_evidence(
            evidence
        )

        if not selected:
            return (
                f"QUESTION: {question.strip()}\n"
                "EVIDENCE: No relevant evidence "
                "was retrieved.\n"
                "ANSWER:"
            )

        blocks = []

        for index, item in enumerate(
            selected,
            start=1,
        ):
            location = item.source.strip()

            if item.page is not None:
                location += (
                    f", page {item.page}"
                )

            blocks.append(
                f"[{index}] SOURCE: {location}\n"
                f"{item.text.strip()}"
            )

        evidence_text = "\n\n".join(
            blocks
        )

        return (
            f"QUESTION: {question.strip()}\n"
            f"EVIDENCE:\n{evidence_text}\n"
            "ANSWER:"
        )

    def answer(
        self,
        question: str,
        evidence: list[Evidence],
        *,
        max_tokens: int = 128,
        temperature: float = 0.1,
    ) -> GroundedAnswer:
        """Answer using only retrieved evidence."""

        if not question.strip():
            raise ValueError(
                "question cannot be empty."
            )

        selected = self.select_evidence(
            evidence
        )

        if not selected:
            return GroundedAnswer(
                text="NOT ENOUGH EVIDENCE",
                provider=self.provider,
                model="retrieval-gate",
                evidence=(),
            )

        request = InferenceRequest(
            prompt=self.build_prompt(
                question,
                selected,
            ),
            system_prompt=(
                "You answer questions about documents. "
                "Use only the supplied evidence. "
                "Extract the shortest answer that directly "
                "answers the question. "
                "Prefer exact names, objects, numbers, dates, "
                "and facts appearing in the evidence. "
                "Do not invent facts. "
                "If the evidence does not contain the answer, "
                "output exactly: NOT ENOUGH EVIDENCE"
            ),
            max_tokens=max_tokens,
            temperature=temperature,
        )

        result: InferenceResult = (
            self.router.generate(
                self.provider,
                request,
            )
        )

        text = result.text.strip()

        if not text:
            text = "NOT ENOUGH EVIDENCE"

        return GroundedAnswer(
            text=text,
            provider=result.provider,
            model=result.model,
            evidence=tuple(selected),
        )
