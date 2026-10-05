"""Base interface for GNOMEdata inference providers."""

from abc import ABC, abstractmethod

from app.inference.types import (
    InferenceRequest,
    InferenceResult,
    ProviderCapabilities,
)


class InferenceProvider(ABC):
    """Common contract implemented by every inference engine."""

    name: str = "unknown"
    model_name: str = "unknown"

    @abstractmethod
    def generate(
        self,
        request: InferenceRequest,
    ) -> InferenceResult:
        """Generate a response."""

    @abstractmethod
    def health(self) -> bool:
        """Return whether the provider is ready."""

    def capabilities(
        self,
    ) -> ProviderCapabilities:
        """Describe optional provider features."""

        return ProviderCapabilities()
