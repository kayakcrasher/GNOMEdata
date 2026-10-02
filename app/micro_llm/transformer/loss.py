"""Cross-entropy loss for GNOME Micro."""

import numpy as np


def cross_entropy_forward(
    logits: np.ndarray,
    targets: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return mean cross-entropy and softmax probabilities."""

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

    loss = -np.mean(
        np.log(chosen + 1e-12)
    )

    return float(loss), probabilities


def cross_entropy_backward(
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Gradient of mean cross-entropy with respect to logits."""

    gradient = probabilities.copy()

    batch, length = targets.shape

    rows = np.arange(batch)[:, None]
    columns = np.arange(length)[None, :]

    gradient[
        rows,
        columns,
        targets,
    ] -= 1.0

    gradient /= (
        batch * length
    )

    return gradient
