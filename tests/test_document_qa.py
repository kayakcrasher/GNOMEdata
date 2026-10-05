"""Tests for GNOMEdata end-to-end document QA."""

from app.inference.base import InferenceProvider
from app.inference.router import InferenceRouter
from app.inference.types import (
    InferenceRequest,
    InferenceResult,
    ProviderCapabilities,
)
from app.rag.boards import Log
from app.rag.document_qa import DocumentQA
from app.rag.local_embedder import LocalHashEmbedder
from app.rag.pipeline import MillPipeline
from app.rag.yard import LumberYard


class FakeProvider(InferenceProvider):
    """Deterministic inference worker for pipeline tests."""

    name = "fake"

    def generate(
        self,
        request: InferenceRequest,
    ) -> InferenceResult:
        assert "Randy sells pies" in request.prompt

        return InferenceResult(
            text="Randy sells pies.",
            provider=self.name,
            model="fake-grounded-model",
            metadata={},
        )

    def health(self) -> bool:
        return True

    def capabilities(
        self,
    ) -> ProviderCapabilities:
        return ProviderCapabilities(
            generation=True,
            chat=False,
            embeddings=False,
            tools=False,
            structured_output=False,
        )


def test_document_qa_end_to_end(
    tmp_path,
) -> None:
    database = tmp_path / "yard.db"

    yard = LumberYard(database)
    embedder = LocalHashEmbedder()

    pipeline = MillPipeline(
        yard=yard,
        embedder=embedder,
        board_size=800,
        overlap=120,
    )

    log = Log(
        log_id="randy-document",
        source_name="randy.txt",
        text=(
            "Randy lives in Oakville. "
            "Randy owns a dog named Tom. "
            "Randy sells pies from his home. "
            "His shop is open every weekday."
        ),
    )

    report = pipeline.process(log)

    assert report.total >= 1
    assert report.embedded >= 1

    router = InferenceRouter()
    router.register(FakeProvider())

    qa = DocumentQA(
        yard=yard,
        embedder=embedder,
        router=router,
        provider="fake",
    )

    result = qa.ask(
        "What does Randy sell?"
    )

    assert result.answer == "Randy sells pies."
    assert result.provider == "fake"
    assert result.model == "fake-grounded-model"

    assert result.evidence
    assert any(
        "Randy sells pies" in item.text
        for item in result.evidence
    )

    assert any(
        item.source == "randy.txt"
        for item in result.evidence
    )

    yard.close()
