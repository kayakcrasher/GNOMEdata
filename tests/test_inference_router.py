"""Tests for the model-agnostic inference router."""

import pytest

from app.inference.base import InferenceProvider
from app.inference.router import InferenceRouter
from app.inference.types import (
    InferenceRequest,
    InferenceResult,
    ProviderCapabilities,
)


class FakeProvider(InferenceProvider):
    """Deterministic provider used for router tests."""

    name = "fake"
    model_name = "fake-v1"

    def __init__(
        self,
        healthy: bool = True,
    ) -> None:
        self._healthy = healthy

    def generate(
        self,
        request: InferenceRequest,
    ) -> InferenceResult:
        return InferenceResult(
            text=f"ECHO: {request.prompt}",
            provider=self.name,
            model=self.model_name,
        )

    def health(self) -> bool:
        return self._healthy

    def capabilities(
        self,
    ) -> ProviderCapabilities:
        return ProviderCapabilities(
            generation=True,
        )


def test_register_and_list_provider() -> None:
    router = InferenceRouter()
    router.register(FakeProvider())

    assert router.names() == ("fake",)


def test_route_generation() -> None:
    router = InferenceRouter()
    router.register(FakeProvider())

    result = router.generate(
        "fake",
        InferenceRequest(
            prompt="WAWA SULI",
        ),
    )

    assert result.text == "ECHO: WAWA SULI"
    assert result.provider == "fake"
    assert result.model == "fake-v1"


def test_health() -> None:
    router = InferenceRouter()
    router.register(FakeProvider())

    assert router.health() == {
        "fake": True,
    }


def test_duplicate_provider_rejected() -> None:
    router = InferenceRouter()

    router.register(FakeProvider())

    with pytest.raises(ValueError):
        router.register(FakeProvider())


def test_unknown_provider_rejected() -> None:
    router = InferenceRouter()

    with pytest.raises(KeyError):
        router.get("missing")


def test_unhealthy_provider_rejected() -> None:
    router = InferenceRouter()

    router.register(
        FakeProvider(
            healthy=False,
        )
    )

    with pytest.raises(RuntimeError):
        router.generate(
            "fake",
            InferenceRequest(
                prompt="hello",
            ),
        )
