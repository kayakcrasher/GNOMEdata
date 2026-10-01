"""Hugging Face embedding worker for GNOMEdata.

Hugging Face supplies compute.
GNOMEdata owns the mill.

This adapter converts text into vectors while obeying GNOMEdata's
existing embedding contract. It can later be replaced by a local
model without changing the Lumber Yard.
"""

import os
from dataclasses import dataclass

import httpx

from app.rag.embeddings import (
    Embedding,
    build_embedding,
)


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_BASE_URL = (
    "https://router.huggingface.co/hf-inference/models"
)


@dataclass(frozen=True)
class HuggingFaceConfig:
    """Configuration for the Hugging Face embedding worker."""

    token: str
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    timeout: float = 30.0

    @classmethod
    def from_env(cls) -> "HuggingFaceConfig":
        """Load worker configuration from environment variables."""

        token = os.getenv(
            "HUGGINGFACE_TOKEN",
            "",
        ).strip()

        if not token:
            raise ValueError(
                "HUGGINGFACE_TOKEN is not configured."
            )

        model = os.getenv(
            "GNOMEDATA_EMBEDDING_MODEL",
            DEFAULT_MODEL,
        ).strip()

        if not model:
            raise ValueError(
                "embedding model cannot be empty."
            )

        return cls(
            token=token,
            model=model,
        )


class HuggingFaceEmbedder:
    """Remote embedding worker using Hugging Face inference."""

    def __init__(
        self,
        config: HuggingFaceConfig,
    ) -> None:
        if not config.token.strip():
            raise ValueError(
                "Hugging Face token cannot be empty."
            )

        if not config.model.strip():
            raise ValueError(
                "Hugging Face model cannot be empty."
            )

        self.config = config

    @property
    def model_name(self) -> str:
        """Return the vector-space identity."""
        return self.config.model

    def embed(
        self,
        text: str,
    ) -> Embedding:
        """Send text to the worker and return GNOMEdata lumber coordinates."""

        if not text.strip():
            raise ValueError(
                "cannot embed empty text."
            )

        url = (
            f"{self.config.base_url.rstrip('/')}/"
            f"{self.config.model}/"
            "pipeline/feature-extraction"
        )

        response = httpx.post(
            url,
            headers={
                "Authorization": (
                    f"Bearer {self.config.token}"
                ),
                "Content-Type": "application/json",
            },
            json={
                "inputs": text,
            },
            timeout=self.config.timeout,
        )

        response.raise_for_status()

        payload = response.json()

        vector = self._extract_vector(
            payload
        )

        return build_embedding(
            vector,
            self.model_name,
        )

    @staticmethod
    def _extract_vector(
        payload,
    ) -> list[float]:
        """Normalize common Hugging Face embedding response shapes."""

        if not isinstance(payload, list):
            raise ValueError(
                "unexpected Hugging Face embedding response."
            )

        if not payload:
            raise ValueError(
                "Hugging Face returned an empty embedding."
            )

        # Already a single embedding vector.
        if all(
            isinstance(value, (int, float))
            for value in payload
        ):
            return [
                float(value)
                for value in payload
            ]

        # Some feature-extraction models return token vectors.
        # Mean-pool them into one passage vector.
        if all(
            isinstance(row, list)
            for row in payload
        ):
            rows = payload

            if not rows:
                raise ValueError(
                    "Hugging Face returned no token vectors."
                )

            dimensions = len(rows[0])

            if dimensions == 0:
                raise ValueError(
                    "Hugging Face returned empty token vectors."
                )

            if any(
                len(row) != dimensions
                for row in rows
            ):
                raise ValueError(
                    "Hugging Face returned inconsistent vector dimensions."
                )

            return [
                sum(
                    float(row[index])
                    for row in rows
                ) / len(rows)
                for index in range(dimensions)
            ]

        raise ValueError(
            "unsupported Hugging Face embedding response."
        )
