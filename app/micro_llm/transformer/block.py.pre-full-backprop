"""Transformer block for GNOME Micro v0.4."""

import numpy as np

from app.micro_llm.transformer.attention import (
    scaled_dot_product_attention,
)
from app.micro_llm.transformer.config import TransformerConfig


def layer_norm(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """Normalize each token across its feature dimension."""

    mean = x.mean(
        axis=-1,
        keepdims=True,
    )

    variance = x.var(
        axis=-1,
        keepdims=True,
    )

    normalized = (
        (x - mean)
        / np.sqrt(variance + epsilon)
    )

    return normalized * gamma + beta


class TransformerBlock:
    """One causal Transformer block."""

    def __init__(
        self,
        config: TransformerConfig,
        rng: np.random.Generator,
    ) -> None:
        self.config = config

        d = config.d_model
        ff = config.d_ff

        scale = 1.0 / np.sqrt(d)

        def matrix(
            rows: int,
            columns: int,
        ) -> np.ndarray:
            return (
                rng.standard_normal(
                    (rows, columns)
                ).astype(np.float32)
                * scale
            )

        self.wq = matrix(d, d)
        self.wk = matrix(d, d)
        self.wv = matrix(d, d)
        self.wo = matrix(d, d)

        self.w1 = matrix(d, ff)
        self.b1 = np.zeros(
            ff,
            dtype=np.float32,
        )

        self.w2 = matrix(ff, d)
        self.b2 = np.zeros(
            d,
            dtype=np.float32,
        )

        self.ln1_gamma = np.ones(
            d,
            dtype=np.float32,
        )

        self.ln1_beta = np.zeros(
            d,
            dtype=np.float32,
        )

        self.ln2_gamma = np.ones(
            d,
            dtype=np.float32,
        )

        self.ln2_beta = np.zeros(
            d,
            dtype=np.float32,
        )

    def _split_heads(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        batch, length, _ = x.shape

        x = x.reshape(
            batch,
            length,
            self.config.num_heads,
            self.config.head_size,
        )

        return x.transpose(
            0,
            2,
            1,
            3,
        )

    def _merge_heads(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        batch, heads, length, depth = x.shape

        return (
            x.transpose(0, 2, 1, 3)
            .reshape(
                batch,
                length,
                heads * depth,
            )
        )

    def forward(
        self,
        x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run one Transformer block."""

        q = self._split_heads(
            x @ self.wq
        )

        k = self._split_heads(
            x @ self.wk
        )

        v = self._split_heads(
            x @ self.wv
        )

        attention_output, weights = (
            scaled_dot_product_attention(
                q,
                k,
                v,
            )
        )

        attention_output = (
            self._merge_heads(
                attention_output
            )
            @ self.wo
        )

        x = layer_norm(
            x + attention_output,
            self.ln1_gamma,
            self.ln1_beta,
        )

        hidden = np.maximum(
            0.0,
            x @ self.w1 + self.b1,
        )

        feed_forward = (
            hidden @ self.w2
            + self.b2
        )

        x = layer_norm(
            x + feed_forward,
            self.ln2_gamma,
            self.ln2_beta,
        )

        return x, weights

    @property
    def parameter_count(self) -> int:
        arrays = (
            self.wq,
            self.wk,
            self.wv,
            self.wo,
            self.w1,
            self.b1,
            self.w2,
            self.b2,
            self.ln1_gamma,
            self.ln1_beta,
            self.ln2_gamma,
            self.ln2_beta,
        )

        return sum(
            array.size
            for array in arrays
        )
