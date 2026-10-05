"""Shared types for GNOMEdata inference engines."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class InferenceRequest:
    """A model-agnostic inference request."""

    prompt: str
    max_tokens: int = 128
    temperature: float = 0.2
    system_prompt: str | None = None


@dataclass(frozen=True)
class InferenceResult:
    """Normalized output returned by every inference engine."""

    text: str
    provider: str
    model: str

    input_tokens: int | None = None
    output_tokens: int | None = None

    metadata: dict = field(
        default_factory=dict,
    )


@dataclass(frozen=True)
class ProviderCapabilities:
    """Features supported by an inference provider."""

    generation: bool = True
    chat: bool = False
    embeddings: bool = False
    tools: bool = False
    structured_output: bool = False
