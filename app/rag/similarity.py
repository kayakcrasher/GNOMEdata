"""Vector similarity tools for the GNOMEdata Lumber Yard.

Boards stored in the Lumber Yard can be represented by embedding
vectors. Cosine similarity measures how closely two vector directions
align.

This module deliberately knows nothing about Hugging Face, Groq,
SQLite, or any other provider. It is pure vector math.
"""

import math
from collections.abc import Sequence


Vector = Sequence[float]


def dot_product(a: Vector, b: Vector) -> float:
    """Return the dot product of two equal-length vectors."""
    _validate_pair(a, b)

    return sum(
        x * y
        for x, y in zip(a, b)
    )


def magnitude(vector: Vector) -> float:
    """Return the Euclidean magnitude of a vector."""
    if not vector:
        raise ValueError("vector cannot be empty.")

    return math.sqrt(
        sum(value * value for value in vector)
    )


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Measure directional similarity between two vectors.

    Returns a value from -1.0 to 1.0.

     1.0 -> same direction
     0.0 -> perpendicular
    -1.0 -> opposite direction

    A high score means vector similarity. It does not prove
    factual correctness or answer relevance.
    """
    _validate_pair(a, b)

    magnitude_a = magnitude(a)
    magnitude_b = magnitude(b)

    if magnitude_a == 0.0 or magnitude_b == 0.0:
        raise ValueError(
            "cosine similarity is undefined for a zero vector."
        )

    similarity = (
        dot_product(a, b)
        / (magnitude_a * magnitude_b)
    )

    # Protect against tiny floating-point overshoots such as
    # 1.0000000000000002.
    return max(-1.0, min(1.0, similarity))


def _validate_pair(a: Vector, b: Vector) -> None:
    """Validate vectors before comparing them."""
    if not a or not b:
        raise ValueError("vectors cannot be empty.")

    if len(a) != len(b):
        raise ValueError(
            "vectors must have the same dimensions."
        )
