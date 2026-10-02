"""Token and position embeddings for GNOME Micro."""

import numpy as np

from app.micro_llm.transformer.config import TransformerConfig


class TransformerEmbeddings:
    """Learnable token and position representations."""

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
    ) -> np.ndarray:
        if tokens.ndim != 2:
            raise ValueError(
                "tokens must have shape "
                "(batch, sequence)."
            )

        length = tokens.shape[1]

        if length > self.config.context_size:
            raise ValueError(
                "sequence exceeds context window."
            )

        positions = self.position[:length]

        return (
            self.token[tokens]
            + positions[None, :, :]
        )

    @property
    def parameter_count(self) -> int:
        return (
            self.token.size
            + self.position.size
        )
