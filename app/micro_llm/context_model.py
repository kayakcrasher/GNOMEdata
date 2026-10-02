"""Context-window language model for GNOME Micro."""

from collections import Counter, defaultdict
import json
import random
from pathlib import Path


class ContextLanguageModel:
    """Pure-Python variable-order context language model."""

    def __init__(
        self,
        context_size: int = 6,
    ) -> None:
        if context_size < 1:
            raise ValueError(
                "context_size must be positive."
            )

        self.context_size = context_size

        # context tuple -> counts of possible next tokens
        self.counts: dict[
            tuple[int, ...],
            Counter[int],
        ] = defaultdict(Counter)

    def train(
        self,
        tokens: list[int],
    ) -> int:
        """Learn next-token counts from a token sequence."""

        if len(tokens) < 2:
            raise ValueError(
                "training data requires at least two tokens."
            )

        examples = 0

        for index in range(1, len(tokens)):
            target = tokens[index]

            maximum = min(
                self.context_size,
                index,
            )

            # Learn every context length so generation
            # can gracefully back off when needed.
            for size in range(1, maximum + 1):
                context = tuple(
                    tokens[index - size:index]
                )

                self.counts[context][target] += 1

            examples += 1

        return examples

    def distribution(
        self,
        context: list[int],
    ) -> Counter[int] | None:
        """Find the longest known context."""

        maximum = min(
            self.context_size,
            len(context),
        )

        for size in range(maximum, 0, -1):
            key = tuple(context[-size:])

            choices = self.counts.get(key)

            if choices:
                return choices

        return None

    def predict_next(
        self,
        context: list[int],
        rng: random.Random,
    ) -> int | None:
        """Sample one next token from learned counts."""

        choices = self.distribution(context)

        if not choices:
            return None

        tokens = list(choices.keys())
        weights = list(choices.values())

        return rng.choices(
            tokens,
            weights=weights,
            k=1,
        )[0]

    def generate(
        self,
        prompt: list[int],
        max_new_tokens: int = 100,
        seed: int = 42,
    ) -> list[int]:
        """Continue a prompt using learned context."""

        if not prompt:
            raise ValueError(
                "prompt cannot be empty."
            )

        output = list(prompt)
        rng = random.Random(seed)

        for _ in range(max_new_tokens):
            token = self.predict_next(
                output,
                rng,
            )

            if token is None:
                break

            output.append(token)

        return output

    def save(
        self,
        path: str | Path,
    ) -> None:
        """Save model state as portable JSON."""

        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        records = []

        for context, counter in self.counts.items():
            records.append(
                {
                    "context": list(context),
                    "next": {
                        str(token): count
                        for token, count
                        in counter.items()
                    },
                }
            )

        payload = {
            "version": 1,
            "context_size": self.context_size,
            "records": records,
        }

        path.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "ContextLanguageModel":
        """Load a previously trained model."""

        payload = json.loads(
            Path(path).read_text(
                encoding="utf-8"
            )
        )

        model = cls(
            context_size=payload["context_size"]
        )

        for record in payload["records"]:
            context = tuple(record["context"])

            for token, count in record["next"].items():
                model.counts[context][int(token)] = count

        return model
