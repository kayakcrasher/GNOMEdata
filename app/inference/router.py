"""Inference provider routing for GNOMEdata."""

from app.inference.base import InferenceProvider
from app.inference.types import (
    InferenceRequest,
    InferenceResult,
)


class InferenceRouter:
    """Register and route requests to inference providers."""

    def __init__(self) -> None:
        self._providers: dict[str, InferenceProvider] = {}

    def register(
        self,
        provider: InferenceProvider,
        *,
        name: str | None = None,
    ) -> None:
        """Register an inference provider."""

        provider_name = name or provider.name

        if not provider_name:
            raise ValueError(
                "Provider must have a name."
            )

        if provider_name in self._providers:
            raise ValueError(
                f"Provider already registered: "
                f"{provider_name}"
            )

        self._providers[provider_name] = provider

    def unregister(
        self,
        name: str,
    ) -> None:
        """Remove a provider."""

        if name not in self._providers:
            raise KeyError(
                f"Unknown provider: {name}"
            )

        del self._providers[name]

    def get(
        self,
        name: str,
    ) -> InferenceProvider:
        """Return a registered provider."""

        try:
            return self._providers[name]
        except KeyError as error:
            available = ", ".join(
                sorted(self._providers)
            ) or "(none)"

            raise KeyError(
                f"Unknown provider: {name}. "
                f"Available: {available}"
            ) from error

    def names(self) -> tuple[str, ...]:
        """Return registered provider names."""

        return tuple(
            sorted(self._providers)
        )

    def health(
        self,
    ) -> dict[str, bool]:
        """Return provider health states."""

        return {
            name: provider.health()
            for name, provider
            in self._providers.items()
        }

    def generate(
        self,
        provider: str,
        request: InferenceRequest,
    ) -> InferenceResult:
        """Route a generation request."""

        engine = self.get(provider)

        if not engine.health():
            raise RuntimeError(
                f"Provider is unhealthy: {provider}"
            )

        return engine.generate(request)
