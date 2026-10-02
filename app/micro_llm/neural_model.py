"""GNOME Micro v0.3 neural byte language model."""

from pathlib import Path
import numpy as np


class NeuralLanguageModel:
    """Small neural next-byte predictor trained entirely locally."""

    def __init__(
        self,
        vocab_size: int = 256,
        context_size: int = 32,
        embedding_size: int = 64,
        hidden_size: int = 128,
        seed: int = 42,
    ) -> None:
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        rng = np.random.default_rng(seed)

        self.embeddings = (
            rng.standard_normal(
                (vocab_size, embedding_size)
            ).astype(np.float32)
            * 0.02
        )

        input_size = context_size * embedding_size

        self.w1 = (
            rng.standard_normal(
                (input_size, hidden_size)
            ).astype(np.float32)
            * np.sqrt(2.0 / input_size)
        )

        self.b1 = np.zeros(
            hidden_size,
            dtype=np.float32,
        )

        self.w2 = (
            rng.standard_normal(
                (hidden_size, vocab_size)
            ).astype(np.float32)
            * np.sqrt(2.0 / hidden_size)
        )

        self.b2 = np.zeros(
            vocab_size,
            dtype=np.float32,
        )

    @property
    def parameter_count(self) -> int:
        return (
            self.embeddings.size
            + self.w1.size
            + self.b1.size
            + self.w2.size
            + self.b2.size
        )

    @staticmethod
    def _softmax(
        logits: np.ndarray,
    ) -> np.ndarray:
        shifted = logits - np.max(
            logits,
            axis=1,
            keepdims=True,
        )

        exp = np.exp(shifted)

        return exp / np.sum(
            exp,
            axis=1,
            keepdims=True,
        )

    def _prepare(
        self,
        tokens: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(tokens) <= self.context_size:
            raise ValueError(
                "corpus is smaller than context window."
            )

        count = len(tokens) - self.context_size

        x = np.empty(
            (count, self.context_size),
            dtype=np.int64,
        )

        y = np.empty(
            count,
            dtype=np.int64,
        )

        source = np.asarray(
            tokens,
            dtype=np.int64,
        )

        for index in range(count):
            x[index] = source[
                index:index + self.context_size
            ]

            y[index] = source[
                index + self.context_size
            ]

        return x, y

    def _forward(
        self,
        x: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        embedded = self.embeddings[x]

        flat = embedded.reshape(
            len(x),
            -1,
        )

        hidden_pre = (
            flat @ self.w1
            + self.b1
        )

        hidden = np.tanh(hidden_pre)

        logits = (
            hidden @ self.w2
            + self.b2
        )

        return flat, hidden, logits

    def loss(
        self,
        tokens: list[int],
        batch_size: int = 128,
    ) -> float:
        x, y = self._prepare(tokens)

        total = 0.0
        seen = 0

        for start in range(
            0,
            len(x),
            batch_size,
        ):
            xb = x[start:start + batch_size]
            yb = y[start:start + batch_size]

            _, _, logits = self._forward(xb)

            probabilities = self._softmax(logits)

            chosen = probabilities[
                np.arange(len(yb)),
                yb,
            ]

            total += float(
                -np.log(
                    np.maximum(chosen, 1e-12)
                ).sum()
            )

            seen += len(yb)

        return total / seen

    def train(
        self,
        tokens: list[int],
        epochs: int = 20,
        learning_rate: float = 0.03,
        batch_size: int = 64,
        seed: int = 42,
        validation_tokens: list[int] | None = None,
    ) -> list[tuple[float, float | None]]:
        x, y = self._prepare(tokens)

        rng = np.random.default_rng(seed)

        history: list[
            tuple[float, float | None]
        ] = []

        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(x))

            for start in range(
                0,
                len(order),
                batch_size,
            ):
                indices = order[
                    start:start + batch_size
                ]

                xb = x[indices]
                yb = y[indices]

                flat, hidden, logits = self._forward(xb)

                probabilities = self._softmax(logits)

                batch_count = len(yb)

                d_logits = probabilities.copy()

                d_logits[
                    np.arange(batch_count),
                    yb,
                ] -= 1.0

                d_logits /= batch_count

                grad_w2 = hidden.T @ d_logits
                grad_b2 = d_logits.sum(axis=0)

                d_hidden = d_logits @ self.w2.T

                d_hidden_pre = (
                    d_hidden
                    * (1.0 - hidden * hidden)
                )

                grad_w1 = (
                    flat.T @ d_hidden_pre
                )

                grad_b1 = (
                    d_hidden_pre.sum(axis=0)
                )

                d_flat = (
                    d_hidden_pre @ self.w1.T
                )

                d_embeddings = d_flat.reshape(
                    batch_count,
                    self.context_size,
                    self.embedding_size,
                )

                self.w2 -= (
                    learning_rate * grad_w2
                )

                self.b2 -= (
                    learning_rate * grad_b2
                )

                self.w1 -= (
                    learning_rate * grad_w1
                )

                self.b1 -= (
                    learning_rate * grad_b1
                )

                for row in range(batch_count):
                    np.add.at(
                        self.embeddings,
                        xb[row],
                        -learning_rate
                        * d_embeddings[row],
                    )

            training_loss = self.loss(
                tokens,
                batch_size=batch_size,
            )

            validation_loss = None

            if validation_tokens:
                validation_loss = self.loss(
                    validation_tokens,
                    batch_size=batch_size,
                )

            history.append(
                (
                    training_loss,
                    validation_loss,
                )
            )

            validation_text = (
                f"{validation_loss:.4f}"
                if validation_loss is not None
                else "n/a"
            )

            print(
                f"Epoch {epoch:02d} | "
                f"train {training_loss:.4f} | "
                f"valid {validation_text}"
            )

        return history

    def probabilities(
        self,
        context: list[int],
    ) -> np.ndarray:
        context = context[-self.context_size:]

        if len(context) < self.context_size:
            context = (
                [32]
                * (
                    self.context_size
                    - len(context)
                )
                + context
            )

        x = np.asarray(
            [context],
            dtype=np.int64,
        )

        _, _, logits = self._forward(x)

        return self._softmax(logits)[0]

    def generate(
        self,
        prompt: list[int],
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        seed: int = 42,
    ) -> list[int]:
        if not prompt:
            raise ValueError(
                "prompt cannot be empty."
            )

        if temperature <= 0:
            raise ValueError(
                "temperature must be positive."
            )

        rng = np.random.default_rng(seed)

        output = list(prompt)

        for _ in range(max_new_tokens):
            probabilities = self.probabilities(
                output
            )

            logits = np.log(
                np.maximum(
                    probabilities,
                    1e-12,
                )
            )

            logits /= temperature

            probabilities = self._softmax(
                logits.reshape(1, -1)
            )[0]

            token = int(
                rng.choice(
                    self.vocab_size,
                    p=probabilities,
                )
            )

            output.append(token)

        return output

    def save(
        self,
        path: str | Path,
    ) -> None:
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez_compressed(
            path,
            vocab_size=self.vocab_size,
            context_size=self.context_size,
            embedding_size=self.embedding_size,
            hidden_size=self.hidden_size,
            embeddings=self.embeddings,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "NeuralLanguageModel":
        data = np.load(
            path,
            allow_pickle=False,
        )

        model = cls(
            vocab_size=int(data["vocab_size"]),
            context_size=int(data["context_size"]),
            embedding_size=int(data["embedding_size"]),
            hidden_size=int(data["hidden_size"]),
        )

        model.embeddings = data[
            "embeddings"
        ].astype(np.float32)

        model.w1 = data["w1"].astype(
            np.float32
        )

        model.b1 = data["b1"].astype(
            np.float32
        )

        model.w2 = data["w2"].astype(
            np.float32
        )

        model.b2 = data["b2"].astype(
            np.float32
        )

        return model
