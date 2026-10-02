"""Trainable token and position embeddings for GNOME Micro."""

from dataclasses import dataclass

import numpy as np

from app.micro_llm.transformer.config import TransformerConfig


@dataclass
class EmbeddingCache:
    """Values required for embedding backprop."""

    tokens: np.ndarray
    length: int


class TransformerEmbeddings:
    """Learnable byte-token and positional representations."""

    def __init__(
        self,
        config: TransformerConfig,
        rng: np.random.Generator,
    ) -> None:
        self.config = config

        self.token = (
            rng.standard_normal(
                (
                    config.vocab_size,
                    config.d_model,
                )
            ).astype(np.float32)
            * 0.02
        )

        self.position = (
            rng.standard_normal(
                (
                    config.context_size,
                    config.d_model,
                )
            ).astype(np.float32)
            * 0.02
        )

    def forward(
        self,
        tokens: np.ndarray,
        return_cache: bool = False,
    ):
        if tokens.ndim != 2:
            raise ValueError(
                "tokens must have shape (batch, sequence)."
            )

        length = tokens.shape[1]

        if length > self.config.context_size:
            raise ValueError(
                "sequence exceeds context window."
            )

        output = (
            self.token[tokens]
            + self.position[:length][None, :, :]
        )

        if return_cache:
            return output, EmbeddingCache(
                tokens=tokens,
                length=length,
            )

        return output

    def backward(
        self,
        grad_output: np.ndarray,
        cache: EmbeddingCache,
    ) -> dict[str, np.ndarray]:
        """Accumulate gradients into token and position tables."""

        grad_token = np.zeros_like(
            self.token,
            dtype=grad_output.dtype,
        )

        grad_position = np.zeros_like(
            self.position,
            dtype=grad_output.dtype,
        )

        # A token may appear multiple times, so += indexing alone
        # is unsafe with repeated advanced indices. np.add.at
        # performs the required accumulation.
        np.add.at(
            grad_token,
            cache.tokens,
            grad_output,
        )

        grad_position[:cache.length] = (
            grad_output.sum(axis=0)
        )

        return {
            "token": grad_token,
            "position": grad_position,
        }

    @property
    def parameter_count(self) -> int:
        return (
            self.token.size
            + self.position.size
        )
