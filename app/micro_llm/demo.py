"""Train and run the GNOMEdata MicroLLM."""

from app.micro_llm.model import MicroLanguageModel
from app.micro_llm.tokenizer import Tokenizer


CORPUS = """
randy sells pies in oakville.
randy makes pies at home.
tom is randy's dog.
oakville loves pies.
randy sells pies from home.
tom lives with randy.
pies are sold in oakville.
"""


def main() -> None:
    tokenizer = Tokenizer.train(CORPUS)

    tokens = tokenizer.encode(CORPUS)

    model = MicroLanguageModel(
        tokenizer.vocab_size
    )

    losses = model.train(
        tokens,
        epochs=300,
        learning_rate=0.08,
    )

    print("GNOMEdata MicroLLM")
    print("------------------")
    print("Vocabulary:", tokenizer.vocab_size)
    print("Tokens:", len(tokens))
    print("Initial loss:", round(losses[0], 4))
    print("Final loss:", round(losses[-1], 4))

    start = tokenizer.encode("randy")[0]

    generated = model.generate(
        start,
        length=20,
        seed=7,
    )

    print()
    print("Generated:")
    print(tokenizer.decode(generated))


if __name__ == "__main__":
    main()
