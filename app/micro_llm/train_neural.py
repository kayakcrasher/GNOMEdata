"""Train the GNOME Micro neural language model."""

import argparse
from pathlib import Path

from app.micro_llm.byte_tokenizer import ByteTokenizer
from app.micro_llm.neural_model import NeuralLanguageModel
from app.micro_llm.train_context import load_corpus


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/training",
    )

    parser.add_argument(
        "--validation",
        default="data/validation.txt",
    )

    parser.add_argument(
        "--output",
        default="models/gnome-micro-v03.npz",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    tokenizer = ByteTokenizer()

    corpus = load_corpus(
        Path(args.data)
    )

    training_tokens = tokenizer.encode(
        corpus
    )

    validation_path = Path(
        args.validation
    )

    validation_tokens = None

    if validation_path.exists():
        validation_tokens = tokenizer.encode(
            validation_path.read_text(
                encoding="utf-8"
            )
        )

    model = NeuralLanguageModel(
        context_size=32,
        embedding_size=64,
        hidden_size=128,
    )

    print("GNOME Micro v0.3 — NEURAL GNOME")
    print("--------------------------------")
    print("Training bytes:", len(training_tokens))
    print("Parameters:", f"{model.parameter_count:,}")
    print("Context:", model.context_size)
    print("Embedding:", model.embedding_size)
    print("Hidden:", model.hidden_size)
    print()

    model.train(
        training_tokens,
        epochs=args.epochs,
        learning_rate=0.03,
        batch_size=64,
        validation_tokens=validation_tokens,
    )

    model.save(args.output)

    print()
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
