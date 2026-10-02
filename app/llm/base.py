"""Interfaces shared by GNOMEdata language models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMResponse:
    """Result produced by a local language model."""

    text: str
    model: str
    context_used: int


class LocalLLM(ABC):
    """Contract implemented by GNOMEdata local models."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        context: tuple[str, ...] = (),
    ) -> LLMResponse:
        """Generate a response using optional retrieved context."""
