"""Embedding machinery for the GNOMEdata Lumber Yard.

An embedding converts text into a numerical vector.

GNOMEdata owns the interface. External models are workers that can
be swapped without changing the rest of the mill.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


Vector = tuple[float, ...]


@dataclass(frozen=True)
class Embedding:
    """A vector produced for one piece of text."""

    vector: Vector
    model: str
    dimensions: int


class Embedder(Protocol):
    """Contract every GNOMEdata embedding worker must follow."""

    @property
    def model_name(self) -> str:
        """Return the embedding model identifier."""
        ...

    def embed(self, text: str) -> Embedding:
        """Convert one text passage into a vector."""
        ...


def build_embedding(
    values: Sequence[float],
    model: str,
) -> Embedding:
    """Validate raw model output and build an Embedding."""

    if not model.strip():
        raise ValueError(
            "embedding model name cannot be empty."
        )

    if not values:
        raise ValueError(
            "embedding vector cannot be empty."
        )

    vector = tuple(float(value) for value in values)

    return Embedding(
        vector=vector,
        model=model,
        dimensions=len(vector),
    )


def validate_compatible(
    first: Embedding,
    second: Embedding,
) -> None:
    """Ensure two embeddings belong to the same vector space.

    Cosine similarity only makes sense when embeddings were produced
    in compatible vector spaces.
    """

    if first.model != second.model:
        raise ValueError(
            "embeddings use different models."
        )

    if first.dimensions != second.dimensions:
        raise ValueError(
            "embeddings have different dimensions."
        )


class StaticEmbedder:
    """Small deterministic embedder used for tests and development.

    This is not semantic AI. It lets the rest of GNOMEdata use the
    Embedder contract before a remote or local model is connected.
    """

    def __init__(
        self,
        vectors: dict[str, Sequence[float]],
        model_name: str = "gnomedata-static-test",
    ) -> None:

        if not model_name.strip():
            raise ValueError(
                "model_name cannot be empty."
            )

        self._vectors = dict(vectors)
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed(self, text: str) -> Embedding:
        if not text.strip():
            raise ValueError(
                "cannot embed empty text."
            )

        try:
            values = self._vectors[text]
        except KeyError as exc:
            raise KeyError(
                "no static vector configured for this text."
            ) from exc

        return build_embedding(
            values,
            self.model_name,
        )
