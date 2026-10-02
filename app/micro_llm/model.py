"""Pure-Python bigram language model."""

import math
import random


class MicroLanguageModel:
    """Trainable next-token model with no external ML dependency."""

    def __init__(
        self,
        vocab_size: int,
        seed: int = 42,
    ) -> None:
        if vocab_size < 1:
            raise ValueError("vocab_size must be positive.")

        self.vocab_size = vocab_size

        rng = random.Random(seed)

        self.weights = [
            [
                rng.uniform(-0.01, 0.01)
                for _ in range(vocab_size)
            ]
            for _ in range(vocab_size)
        ]

    @staticmethod
    def _softmax(values: list[float]) -> list[float]:
        maximum = max(values)

        exponentials = [
            math.exp(value - maximum)
            for value in values
        ]

        total = sum(exponentials)

        return [
            value / total
            for value in exponentials
        ]

    def probabilities(
        self,
        token_id: int,
    ) -> list[float]:
        return self._softmax(
            self.weights[token_id]
        )

    def train(
        self,
        tokens: list[int],
        epochs: int = 100,
        learning_rate: float = 0.1,
    ) -> list[float]:
        """Train with gradient descent on next-token prediction."""

        if len(tokens) < 2:
            raise ValueError(
                "training data requires at least two tokens."
            )

        losses: list[float] = []

        for _ in range(epochs):
            total_loss = 0.0
            examples = 0

            for current, target in zip(
                tokens,
                tokens[1:],
            ):
                probabilities = self.probabilities(current)

                probability = max(
                    probabilities[target],
                    1e-12,
                )

                total_loss -= math.log(probability)
                examples += 1

                # Gradient of softmax cross entropy.
                for index in range(self.vocab_size):
                    gradient = probabilities[index]

                    if index == target:
                        gradient -= 1.0

                    self.weights[current][index] -= (
                        learning_rate * gradient
                    )

            losses.append(total_loss / examples)

        return losses

    def sample_next(
        self,
        token_id: int,
        rng: random.Random,
    ) -> int:
        probabilities = self.probabilities(token_id)

        return rng.choices(
            range(self.vocab_size),
            weights=probabilities,
            k=1,
        )[0]

    def generate(
        self,
        start_token: int,
        length: int = 20,
        seed: int = 42,
    ) -> list[int]:
        """Generate tokens autoregressively."""

        if length < 1:
            return []

        rng = random.Random(seed)

        output = [start_token]
        current = start_token

        for _ in range(length - 1):
            current = self.sample_next(
                current,
                rng,
            )
            output.append(current)

        return output
