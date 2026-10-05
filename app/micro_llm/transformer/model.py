"""Trainable GNOME Micro v0.4 Transformer."""

from dataclasses import dataclass

import numpy as np

from app.micro_llm.transformer.block import (
    BlockCache,
    TransformerBlock,
)
from app.micro_llm.transformer.config import TransformerConfig
from app.micro_llm.transformer.embeddings import (
    EmbeddingCache,
    TransformerEmbeddings,
)
from app.micro_llm.transformer.loss import (
    cross_entropy_backward,
    cross_entropy_forward,
)


@dataclass
class ModelCache:
    embedding: EmbeddingCache
    blocks: list[BlockCache]
    hidden: np.ndarray
    probabilities: np.ndarray
    targets: np.ndarray
    weights: np.ndarray | None = None
    copy_mask: np.ndarray | None = None


class TransformerLanguageModel:
    """Tiny causal Transformer with explicit NumPy backprop."""

    def __init__(
        self,
        config: TransformerConfig | None = None,
    ) -> None:
        self.config = config or TransformerConfig()

        self.rng = np.random.default_rng(
            self.config.seed
        )

        self.embeddings = TransformerEmbeddings(
            self.config,
            self.rng,
        )

        self.blocks = [
            TransformerBlock(
                self.config,
                self.rng,
            )
            for _ in range(
                self.config.num_layers
            )
        ]

        d = self.config.d_model

        self.lm_head = (
            self.rng.standard_normal(
                (
                    d,
                    self.config.vocab_size,
                )
            ).astype(np.float32)
            * (1.0 / np.sqrt(d))
        )

        self.lm_bias = np.zeros(
            self.config.vocab_size,
            dtype=np.float32,
        )

        # v0.6 trainable copy-head strength.
        self.copy_strength = np.array(
            0.0,
            dtype=np.float32,
        )

    def forward(
        self,
        tokens: np.ndarray,
    ) -> np.ndarray:
        x = self.embeddings.forward(tokens)

        for block in self.blocks:
            x, _ = block.forward(x)

        return (
            x @ self.lm_head
            + self.lm_bias
        )

    def loss(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
    ) -> float:
        logits = self.forward(inputs)

        loss, _ = cross_entropy_forward(
            logits,
            targets,
        )

        return loss

    def forward_loss(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> tuple[float, ModelCache]:
        hidden, embedding_cache = (
            self.embeddings.forward(
                inputs,
                return_cache=True,
            )
        )

        block_caches = []

        for block in self.blocks:
            hidden, cache = block.forward(
                hidden,
                return_cache=True,
            )

            block_caches.append(cache)

        logits = (
            hidden @ self.lm_head
            + self.lm_bias
        )

        # v0.6 copy bias:
        # favor bytes that already occur in the visible context.
        copy_mask = np.zeros_like(logits)

        batch, length = inputs.shape

        rows = np.arange(batch)[:, None]
        positions = np.arange(length)[None, :]

        for source_position in range(length):
            source_tokens = inputs[:, source_position]

            copy_mask[
                rows,
                positions,
                source_tokens[:, None],
            ] += 1.0

        copy_mask /= max(1, length)

        logits = (
            logits
            + self.copy_strength * copy_mask
        )

        loss, probabilities = (
            cross_entropy_forward(
                logits,
                targets,
                weights=weights,
            )
        )

        return loss, ModelCache(
            embedding=embedding_cache,
            blocks=block_caches,
            hidden=hidden,
            probabilities=probabilities,
            targets=targets,
            weights=weights,
            copy_mask=copy_mask,
        )

    def backward(
        self,
        cache: ModelCache,
    ) -> dict[str, np.ndarray]:
        grad_logits = cross_entropy_backward(
            cache.probabilities,
            cache.targets,
            weights=cache.weights,
        )

        flat_hidden = cache.hidden.reshape(
            -1,
            self.config.d_model,
        )

        flat_logits = grad_logits.reshape(
            -1,
            self.config.vocab_size,
        )

        grad_lm_head = (
            flat_hidden.T
            @ flat_logits
        )

        grad_lm_bias = np.sum(
            flat_logits,
            axis=0,
        )

        grad_hidden = (
            grad_logits
            @ self.lm_head.T
        )

        grad_copy_strength = np.array(
            np.sum(
                grad_logits * cache.copy_mask
            ),
            dtype=np.float32,
        )

        gradients = {
            "lm_head": grad_lm_head,
            "lm_bias": grad_lm_bias,
            "copy_strength": grad_copy_strength,
        }

        for index in reversed(
            range(len(self.blocks))
        ):
            grad_hidden, block_grads = (
                self.blocks[index].backward(
                    grad_hidden,
                    cache.blocks[index],
                )
            )

            for name, gradient in block_grads.items():
                gradients[
                    f"blocks.{index}.{name}"
                ] = gradient

        embedding_grads = (
            self.embeddings.backward(
                grad_hidden,
                cache.embedding,
            )
        )

        gradients["embeddings.token"] = (
            embedding_grads["token"]
        )

        gradients["embeddings.position"] = (
            embedding_grads["position"]
        )

        return gradients

    @property
    def parameter_count(self) -> int:
        return (
            self.embeddings.parameter_count
            + sum(
                block.parameter_count
                for block in self.blocks
            )
            + self.lm_head.size
            + self.lm_bias.size
            + self.copy_strength.size
        )
