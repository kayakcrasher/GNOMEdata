"""Trainable multi-head causal attention for GNOME Micro."""

from dataclasses import dataclass

import numpy as np

from app.micro_llm.transformer.attention import (
    AttentionCache,
    scaled_dot_product_attention,
    scaled_dot_product_attention_backward,
)
from app.micro_llm.transformer.config import TransformerConfig


@dataclass
class MultiHeadCache:
    x: np.ndarray
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    merged: np.ndarray
    attention: AttentionCache


class MultiHeadAttention:
    """Multi-head causal self-attention with explicit backprop."""

    def __init__(
        self,
        config: TransformerConfig,
        rng: np.random.Generator,
    ) -> None:
        self.config = config

        d = config.d_model
        scale = 1.0 / np.sqrt(d)

        def matrix() -> np.ndarray:
            return (
                rng.standard_normal((d, d))
                .astype(np.float32)
                * scale
            )

        self.wq = matrix()
        self.wk = matrix()
        self.wv = matrix()
        self.wo = matrix()

    def _split_heads(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        batch, length, _ = x.shape

        return (
            x.reshape(
                batch,
                length,
                self.config.num_heads,
                self.config.head_size,
            )
            .transpose(0, 2, 1, 3)
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
    ) -> tuple[np.ndarray, MultiHeadCache]:
        q = self._split_heads(x @ self.wq)
        k = self._split_heads(x @ self.wk)
        v = self._split_heads(x @ self.wv)

        attended, _, attention_cache = (
            scaled_dot_product_attention(
                q,
                k,
                v,
                return_cache=True,
            )
        )

        merged = self._merge_heads(attended)

        output = merged @ self.wo

        cache = MultiHeadCache(
            x=x,
            q=q,
            k=k,
            v=v,
            merged=merged,
            attention=attention_cache,
        )

        return output, cache

    def backward(
        self,
        grad_output: np.ndarray,
        cache: MultiHeadCache,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        x = cache.x

        # output = merged @ wo
        grad_wo = (
            cache.merged.reshape(
                -1,
                self.config.d_model,
            ).T
            @ grad_output.reshape(
                -1,
                self.config.d_model,
            )
        )

        grad_merged = (
            grad_output @ self.wo.T
        )

        # Reverse merge_heads.
        grad_attended = self._split_heads(
            grad_merged
        )

        grad_q, grad_k, grad_v = (
            scaled_dot_product_attention_backward(
                grad_attended,
                cache.attention,
            )
        )

        # Reverse split_heads.
        grad_q_flat = self._merge_heads(grad_q)
        grad_k_flat = self._merge_heads(grad_k)
        grad_v_flat = self._merge_heads(grad_v)

        flat_x = x.reshape(
            -1,
            self.config.d_model,
        )

        grad_wq = (
            flat_x.T
            @ grad_q_flat.reshape(
                -1,
                self.config.d_model,
            )
        )

        grad_wk = (
            flat_x.T
            @ grad_k_flat.reshape(
                -1,
                self.config.d_model,
            )
        )

        grad_wv = (
            flat_x.T
            @ grad_v_flat.reshape(
                -1,
                self.config.d_model,
            )
        )

        grad_x = (
            grad_q_flat @ self.wq.T
            + grad_k_flat @ self.wk.T
            + grad_v_flat @ self.wv.T
        )

        gradients = {
            "wq": grad_wq,
            "wk": grad_wk,
            "wv": grad_wv,
            "wo": grad_wo,
        }

        return grad_x, gradients

    @property
    def parameter_count(self) -> int:
        return (
            self.wq.size
            + self.wk.size
            + self.wv.size
            + self.wo.size
        )
