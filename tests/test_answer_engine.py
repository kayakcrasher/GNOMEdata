"""Tests for GNOMEdata grounded answering."""

from app.inference.base import InferenceProvider
from app.inference.router import InferenceRouter
from app.inference.types import (
    InferenceRequest,
    InferenceResult,
)
from app.rag.answer_engine import (
    AnswerEngine,
    Evidence,
)


class TestProvider(InferenceProvider):
    name = "test"
    model_name = "test-model"

    def generate(
        self,
        request: InferenceRequest,
    ) -> InferenceResult:
        return InferenceResult(
            text="Randy sells pies.",
            provider=self.name,
            model=self.model_name,
            metadata={
                "prompt": request.prompt,
            },
        )

    def health(self) -> bool:
        return True


def build_engine() -> AnswerEngine:
    router = InferenceRouter()
    router.register(TestProvider())

    return AnswerEngine(
        router=router,
        provider="test",
    )


def test_grounded_answer() -> None:
    engine = build_engine()

    evidence = [
        Evidence(
            text=(
                "Randy lives in Oakville. "
                "Randy sells pies from his home."
            ),
            source="customers.txt",
            page=3,
            score=0.94,
        )
    ]

    result = engine.answer(
        "What does Randy sell?",
        evidence,
    )

    assert result.text == "Randy sells pies."
    assert result.provider == "test"
    assert result.model == "test-model"
    assert len(result.evidence) == 1
    assert result.evidence[0].source == "customers.txt"


def test_prompt_contains_question_and_source() -> None:
    engine = build_engine()

    prompt = engine.build_prompt(
        "What does Randy sell?",
        [
            Evidence(
                text="Randy sells pies.",
                source="customers.txt",
                page=3,
            )
        ],
    )

    assert "What does Randy sell?" in prompt
    assert "Randy sells pies." in prompt
    assert "customers.txt" in prompt
    assert "page 3" in prompt


def test_evidence_limit() -> None:
    engine = AnswerEngine(
        router=build_engine().router,
        provider="test",
        max_evidence=2,
    )

    evidence = [
        Evidence(
            text=f"Evidence {index}",
            source=f"source-{index}.txt",
        )
        for index in range(5)
    ]

    result = engine.answer(
        "Test?",
        evidence,
    )

    assert len(result.evidence) == 2


def test_empty_evidence() -> None:
    engine = build_engine()

    prompt = engine.build_prompt(
        "What happened?",
        [],
    )

    assert "No relevant evidence" in prompt
