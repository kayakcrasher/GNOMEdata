"""Cross-entropy loss for GNOME Micro."""

import numpy as np


def _normalizer(
    targets: np.ndarray,
    weights: np.ndarray | None,
) -> float:
    """Return the denominator used by weighted mean loss."""

    if weights is None:
        return float(targets.size)

    if weights.shape != targets.shape:
        raise ValueError(
            "weights must have the same shape as targets."
        )

    total = float(np.sum(weights))

    if total <= 0.0:
        raise ValueError(
            "weights must contain positive mass."
        )

    return total


def cross_entropy_forward(
    logits: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Return weighted mean cross-entropy and probabilities."""

    shifted = logits - np.max(
        logits,
        axis=-1,
        keepdims=True,
    )

    exp = np.exp(shifted)

    probabilities = (
        exp
        / np.sum(
            exp,
            axis=-1,
            keepdims=True,
        )
    )

    batch, length = targets.shape

    rows = np.arange(batch)[:, None]
    columns = np.arange(length)[None, :]

    chosen = probabilities[
        rows,
        columns,
        targets,
    ]

    token_loss = -np.log(chosen + 1e-12)

    if weights is None:
        loss = np.mean(token_loss)
    else:
        denominator = _normalizer(
            targets,
            weights,
        )

        loss = np.sum(
            token_loss * weights
        ) / denominator

    return float(loss), probabilities


def cross_entropy_backward(
    probabilities: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Gradient of weighted mean cross-entropy."""

    gradient = probabilities.copy()

    batch, length = targets.shape

    rows = np.arange(batch)[:, None]
    columns = np.arange(length)[None, :]

    gradient[
        rows,
        columns,
        targets,
    ] -= 1.0

    if weights is None:
        gradient /= targets.size
    else:
        denominator = _normalizer(
            targets,
            weights,
        )

        gradient *= weights[..., None]
        gradient /= denominator

    return gradient
