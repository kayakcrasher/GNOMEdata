"""Generate text with GNOME Micro v0.3."""

import argparse

from app.micro_llm.byte_tokenizer import ByteTokenizer
from app.micro_llm.neural_model import NeuralLanguageModel


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("prompt")

    parser.add_argument(
        "--model",
        default="models/gnome-micro-v03.npz",
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

    tokenizer = ByteTokenizer()

    model = NeuralLanguageModel.load(
        args.model
    )

    prompt = tokenizer.encode(
        args.prompt
    )

    generated = model.generate(
        prompt,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        seed=args.seed,
    )

    print(tokenizer.decode(generated))


if __name__ == "__main__":
    main()
