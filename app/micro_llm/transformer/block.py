"""Trainable Transformer block for GNOME Micro v0.4."""

from dataclasses import dataclass

import numpy as np

from app.micro_llm.transformer.config import TransformerConfig
from app.micro_llm.transformer.multihead import (
    MultiHeadAttention,
    MultiHeadCache,
)
from app.micro_llm.transformer.norm import (
    LayerNormCache,
    layer_norm_backward,
    layer_norm_forward,
)


def layer_norm(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """Compatibility wrapper for existing tests."""

    output, _ = layer_norm_forward(
        x,
        gamma,
        beta,
        epsilon,
    )

    return output


@dataclass
class BlockCache:
    """Values required for Transformer block backward."""

    x: np.ndarray

    attention_output: np.ndarray
    attention_cache: MultiHeadCache

    norm1: np.ndarray
    norm1_cache: LayerNormCache

    hidden_pre: np.ndarray
    hidden: np.ndarray

    norm2_cache: LayerNormCache


class TransformerBlock:
    """One trainable causal Transformer block."""

    def __init__(
        self,
        config: TransformerConfig,
        rng: np.random.Generator,
    ) -> None:
        self.config = config

        d = config.d_model
        ff = config.d_ff

        self.attention = MultiHeadAttention(
            config,
            rng,
        )

        scale = 1.0 / np.sqrt(d)

        self.w1 = (
            rng.standard_normal((d, ff))
            .astype(np.float32)
            * scale
        )

        self.b1 = np.zeros(
            ff,
            dtype=np.float32,
        )

        self.w2 = (
            rng.standard_normal((ff, d))
            .astype(np.float32)
            * scale
        )

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

    # Compatibility with the original block API.
    @property
    def wq(self) -> np.ndarray:
        return self.attention.wq

    @property
    def wk(self) -> np.ndarray:
        return self.attention.wk

    @property
    def wv(self) -> np.ndarray:
        return self.attention.wv

    @property
    def wo(self) -> np.ndarray:
        return self.attention.wo

    def forward(
        self,
        x: np.ndarray,
        return_cache: bool = False,
    ):
        """Run the Transformer block."""

        attention_output, attention_cache = (
            self.attention.forward(x)
        )

        residual1 = (
            x + attention_output
        )

        norm1, norm1_cache = (
            layer_norm_forward(
                residual1,
                self.ln1_gamma,
                self.ln1_beta,
            )
        )

        hidden_pre = (
            norm1 @ self.w1
            + self.b1
        )

        hidden = np.maximum(
            hidden_pre,
            0.0,
        )

        feed_forward = (
            hidden @ self.w2
            + self.b2
        )

        residual2 = (
            norm1 + feed_forward
        )

        output, norm2_cache = (
            layer_norm_forward(
                residual2,
                self.ln2_gamma,
                self.ln2_beta,
            )
        )

        if return_cache:
            cache = BlockCache(
                x=x,
                attention_output=attention_output,
                attention_cache=attention_cache,
                norm1=norm1,
                norm1_cache=norm1_cache,
                hidden_pre=hidden_pre,
                hidden=hidden,
                norm2_cache=norm2_cache,
            )

            return output, cache

        # Preserve old API:
        # second result used to be attention weights.
        return output, attention_cache.attention.weights

    def backward(
        self,
        grad_output: np.ndarray,
        cache: BlockCache,
    ) -> tuple[
        np.ndarray,
        dict[str, np.ndarray],
    ]:
        """Backpropagate through the entire block."""

        # ---- LayerNorm #2 ----

        (
            grad_residual2,
            grad_ln2_gamma,
            grad_ln2_beta,
        ) = layer_norm_backward(
            grad_output,
            cache.norm2_cache,
        )

        # residual2 = norm1 + feed_forward
        grad_norm1 = grad_residual2.copy()
        grad_feed_forward = grad_residual2

        # ---- W2 ----

        flat_hidden = cache.hidden.reshape(
            -1,
            self.config.d_ff,
        )

        flat_grad_ff = grad_feed_forward.reshape(
            -1,
            self.config.d_model,
        )

        grad_w2 = (
            flat_hidden.T
            @ flat_grad_ff
        )

        grad_b2 = np.sum(
            flat_grad_ff,
            axis=0,
        )

        grad_hidden = (
            grad_feed_forward
            @ self.w2.T
        )

        # ---- ReLU ----

        grad_hidden_pre = (
            grad_hidden
            * (
                cache.hidden_pre > 0.0
            )
        )

        # ---- W1 ----

        flat_norm1 = cache.norm1.reshape(
            -1,
            self.config.d_model,
        )

        flat_grad_hidden_pre = (
            grad_hidden_pre.reshape(
                -1,
                self.config.d_ff,
            )
        )

        grad_w1 = (
            flat_norm1.T
            @ flat_grad_hidden_pre
        )

        grad_b1 = np.sum(
            flat_grad_hidden_pre,
            axis=0,
        )

        grad_norm1 += (
            grad_hidden_pre
            @ self.w1.T
        )

        # ---- LayerNorm #1 ----

        (
            grad_residual1,
            grad_ln1_gamma,
            grad_ln1_beta,
        ) = layer_norm_backward(
            grad_norm1,
            cache.norm1_cache,
        )

        # residual1 = x + attention(x)
        grad_x = grad_residual1.copy()
        grad_attention_output = (
            grad_residual1
        )

        # ---- Multi-head attention ----

        (
            grad_attention_input,
            attention_grads,
        ) = self.attention.backward(
            grad_attention_output,
            cache.attention_cache,
        )

        grad_x += grad_attention_input

        gradients = {
            "wq": attention_grads["wq"],
            "wk": attention_grads["wk"],
            "wv": attention_grads["wv"],
            "wo": attention_grads["wo"],

            "w1": grad_w1,
            "b1": grad_b1,
            "w2": grad_w2,
            "b2": grad_b2,

            "ln1_gamma": grad_ln1_gamma,
            "ln1_beta": grad_ln1_beta,
            "ln2_gamma": grad_ln2_gamma,
            "ln2_beta": grad_ln2_beta,
        }

        return grad_x, gradients

    @property
    def parameter_count(self) -> int:
        return (
            self.attention.parameter_count
            + self.w1.size
            + self.b1.size
            + self.w2.size
            + self.b2.size
            + self.ln1_gamma.size
            + self.ln1_beta.size
            + self.ln2_gamma.size
            + self.ln2_beta.size
        )
