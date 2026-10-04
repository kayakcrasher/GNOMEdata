"""Configuration for GNOME Micro Transformer v0.4."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int = 256
    context_size: int = 32

    # Keep v0.4 small enough for phone experiments.
    d_model: int = 64
    num_heads: int = 4
    d_ff: int = 128
    num_layers: int = 1

    dropout: float = 0.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads."
            )

        if self.context_size < 1:
            raise ValueError(
                "context_size must be positive."
            )

        if self.num_layers < 1:
            raise ValueError(
                "num_layers must be positive."
            )

    @property
    def head_size(self) -> int:
        return self.d_model // self.num_heads
