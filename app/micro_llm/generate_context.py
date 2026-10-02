"""Generate text with GNOME Micro v0.2."""

import argparse

from app.micro_llm.byte_tokenizer import ByteTokenizer
from app.micro_llm.context_model import ContextLanguageModel


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "prompt",
    )

    parser.add_argument(
        "--model",
        default="models/gnome-micro-v02.json",
    )

    parser.add_argument(
        "--tokens",
        type=int,
        default=200,
    )

    args = parser.parse_args()

    tokenizer = ByteTokenizer()

    model = ContextLanguageModel.load(
        args.model
    )

    prompt = tokenizer.encode(
        args.prompt
    )

    generated = model.generate(
        prompt,
        max_new_tokens=args.tokens,
    )

    print(
        tokenizer.decode(generated)
    )


if __name__ == "__main__":
    main()
