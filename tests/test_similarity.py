"""Tests for GNOMEdata Lumber Yard vector similarity."""

import math

import pytest

from app.rag.similarity import (
    cosine_similarity,
    dot_product,
    magnitude,
)


def test_dot_product() -> None:
    """Dot product should multiply and sum matching dimensions."""
    result = dot_product(
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    )

    assert result == pytest.approx(32.0)


def test_vector_magnitude() -> None:
    """Magnitude should measure vector length."""
    result = magnitude([3.0, 4.0])

    assert result == pytest.approx(5.0)


def test_identical_vectors_have_similarity_one() -> None:
    """Identical lumber vectors should point the same direction."""
    vector = [0.2, 0.5, 0.8]

    result = cosine_similarity(vector, vector)

    assert result == pytest.approx(1.0)


def test_parallel_vectors_have_similarity_one() -> None:
    """Magnitude changes should not alter vector direction."""
    result = cosine_similarity(
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
    )

    assert result == pytest.approx(1.0)


def test_perpendicular_vectors_have_similarity_zero() -> None:
    """Perpendicular vectors should have no directional similarity."""
    result = cosine_similarity(
        [1.0, 0.0],
        [0.0, 1.0],
    )

    assert result == pytest.approx(0.0)


def test_opposite_vectors_have_similarity_negative_one() -> None:
    """Opposite vector directions should score negative one."""
    result = cosine_similarity(
        [1.0, 0.0],
        [-1.0, 0.0],
    )

    assert result == pytest.approx(-1.0)


def test_similar_vectors_score_higher_than_unrelated_vectors() -> None:
    """Nearby lumber should outrank unrelated lumber."""
    query = [1.0, 1.0, 0.0]

    similar = [0.9, 1.0, 0.0]
    unrelated = [0.0, 0.0, 1.0]

    assert (
        cosine_similarity(query, similar)
        > cosine_similarity(query, unrelated)
    )


def test_mismatched_dimensions_are_rejected() -> None:
    """Vectors from incompatible spaces cannot be compared."""
    with pytest.raises(
        ValueError,
        match="same dimensions",
    ):
        cosine_similarity(
            [1.0, 2.0],
            [1.0, 2.0, 3.0],
        )


def test_empty_vectors_are_rejected() -> None:
    """Empty vectors contain no direction."""
    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        cosine_similarity([], [])


def test_zero_vector_is_rejected() -> None:
    """Cosine similarity is undefined for a zero vector."""
    with pytest.raises(
        ValueError,
        match="zero vector",
    ):
        cosine_similarity(
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
        )


def test_similarity_is_finite() -> None:
    """Normal vector comparisons should never produce NaN or infinity."""
    result = cosine_similarity(
        [0.123, 0.456, 0.789],
        [0.321, 0.654, 0.987],
    )

    assert math.isfinite(result)
    assert -1.0 <= result <= 1.0
