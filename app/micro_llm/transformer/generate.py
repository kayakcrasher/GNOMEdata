"""Generate text with a trained GNOME Micro Transformer."""

import argparse
from pathlib import Path

import numpy as np

from app.micro_llm.transformer.model import TransformerLanguageModel
from app.micro_llm.transformer.train import parameters


def load_model(path: Path) -> TransformerLanguageModel:
    model = TransformerLanguageModel()
    params = parameters(model)

    with np.load(path) as state:
        missing = set(params) - set(state.files)

        if missing:
            raise RuntimeError(
                f"Checkpoint missing parameters: {sorted(missing)}"
            )

        for name, parameter in params.items():
            saved = state[name]

            if saved.shape != parameter.shape:
                raise RuntimeError(
                    f"Shape mismatch for {name}: "
                    f"{saved.shape} != {parameter.shape}"
                )

            parameter[...] = saved

    return model


def generate(
    model: TransformerLanguageModel,
    prompt: str,
    max_tokens: int,
    temperature: float,
    seed: int,
) -> bytes:
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero.")

    rng = np.random.default_rng(seed)

    generated = bytearray(
        prompt.encode("utf-8", errors="replace")
    )

    if not generated:
        generated.append(10)

    for _ in range(max_tokens):
        context = np.frombuffer(
            bytes(generated[-model.config.context_size:]),
            dtype=np.uint8,
        ).astype(np.int64)

        logits = model.forward(
            context[None, :]
        )[0, -1]

        logits = logits / temperature
        logits = logits - np.max(logits)

        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()

        token = int(
            rng.choice(
                model.config.vocab_size,
                p=probabilities,
            )
        )

        generated.append(token)

    return bytes(generated)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("prompt")

    parser.add_argument(
        "--model",
        default="models/gnome-transformer-v04.npz",
    )

    parser.add_argument(
        "--tokens",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    model = load_model(
        Path(args.model)
    )

    output = generate(
        model=model,
        prompt=args.prompt,
        max_tokens=args.tokens,
        temperature=args.temperature,
        seed=args.seed,
    )

    print(
        output.decode(
            "utf-8",
            errors="replace",
        )
    )


if __name__ == "__main__":
    main()
