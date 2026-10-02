"""Train GNOME Micro v0.2 from local text files."""

import argparse
from pathlib import Path

from app.micro_llm.byte_tokenizer import ByteTokenizer
from app.micro_llm.context_model import ContextLanguageModel


def load_corpus(directory: Path) -> str:
    """Load all local .txt training files."""

    files = sorted(
        directory.rglob("*.txt")
    )

    if not files:
        raise RuntimeError(
            f"No .txt training files found in {directory}"
        )

    chunks: list[str] = []

    for path in files:
        text = path.read_text(
            encoding="utf-8"
        ).strip()

        if text:
            chunks.append(text)

    if not chunks:
        raise RuntimeError(
            "Training files contained no text."
        )

    return "\n\n".join(chunks)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/training",
    )

    parser.add_argument(
        "--output",
        default="models/gnome-micro-v02.json",
    )

    parser.add_argument(
        "--context",
        type=int,
        default=12,
    )

    args = parser.parse_args()

    corpus = load_corpus(
        Path(args.data)
    )

    tokenizer = ByteTokenizer()

    tokens = tokenizer.encode(corpus)

    model = ContextLanguageModel(
        context_size=args.context
    )

    examples = model.train(tokens)

    model.save(args.output)

    print("GNOME Micro v0.2")
    print("-----------------")
    print("Characters:", len(corpus))
    print("Byte tokens:", len(tokens))
    print("Training examples:", examples)
    print("Contexts learned:", len(model.counts))
    print("Context window:", model.context_size)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
