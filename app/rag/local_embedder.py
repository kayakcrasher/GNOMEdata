"""Small deterministic offline embedding worker for GNOMEdata.

This worker intentionally requires no network service and no API key.

It uses feature hashing to place tokens into a fixed vector space.
This is not intended to replace modern neural embeddings. It provides
GNOMEdata with a portable local retrieval baseline.
"""

import hashlib
import math
import re

from app.rag.embeddings import Embedding, build_embedding


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


class LocalHashEmbedder:
    """Generate deterministic local text vectors."""

    def __init__(
        self,
        dimensions: int = 512,
        model_name: str = "gnomedata-local-hash-v1",
    ) -> None:
        if dimensions < 8:
            raise ValueError(
                "dimensions must be at least 8."
            )

        self.dimensions = dimensions
        self.model_name = model_name

    def embed(self, text: str) -> Embedding:
        if not text.strip():
            raise ValueError(
                "cannot embed empty text."
            )

        tokens = TOKEN_PATTERN.findall(
            text.casefold()
        )

        vector = [0.0] * self.dimensions

        for token in tokens:
            digest = hashlib.blake2b(
                token.encode("utf-8"),
                digest_size=8,
            ).digest()

            value = int.from_bytes(
                digest,
                "big",
            )

            index = value % self.dimensions

            # Signed feature hashing reduces collision bias.
            sign = (
                1.0
                if (value >> 1) & 1
                else -1.0
            )

            vector[index] += sign

        magnitude = math.sqrt(
            sum(value * value for value in vector)
        )

        if magnitude:
            vector = [
                value / magnitude
                for value in vector
            ]

        return build_embedding(
            vector,
            self.model_name,
        )
