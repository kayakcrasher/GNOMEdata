"""Build a varied synthetic training corpus for GNOME Micro."""

from pathlib import Path
import random


OUTPUT = Path("data/training/generated.txt")
TARGET = 30_000

random.seed(42)

people = [
    "Alice", "Bob", "Carlos", "Diana",
    "Evelyn", "Frank", "Grace", "Henry",
]

places = [
    "Denver", "Oakville", "Memphis", "Boston",
    "Seattle", "Austin", "Chicago", "Atlanta",
]

products = [
    "books", "tools", "software", "coffee",
    "furniture", "parts", "bread", "equipment",
]

subjects = [
    "the company",
    "the business",
    "the organization",
    "the research team",
]

templates = [
    "{person} lives in {place}.",
    "{person}'s recorded location is {place}.",
    "The evidence states that {person} lives in {place}.",
    "Where does {person} live? {person} lives in {place}.",
    "What city is associated with {person}? The answer is {place}.",

    "{person} sells {product} in {place}.",
    "The records show that {person} sells {product}.",
    "What does {person} sell? {person} sells {product}.",

    "{subject} reported revenue of ${revenue}.",
    "{subject} reported expenses of ${expenses}.",
    "Revenue was ${revenue} and expenses were ${expenses}.",
    "Profit equals revenue minus expenses.",
    "Financial analysis should use values supported by the records.",

    "Evidence should remain connected to its source.",
    "An answer should use relevant evidence.",
    "Unsupported information should not be presented as fact.",
    "Several documents may contain related information.",
    "A question can require evidence from multiple documents.",
    "Conflicting records should be identified rather than ignored.",
    "A summary should preserve the meaning of the source.",
    "Research can contain people, places, dates, events, and measurements.",
    "Stored information can be searched to answer questions.",
    "GNOMEdata processes information locally.",
]


def make_line() -> str:
    revenue = random.randrange(
        10_000,
        500_001,
        100,
    )

    expenses = random.randrange(
        5_000,
        revenue + 1,
        100,
    )

    template = random.choice(templates)

    return template.format(
        person=random.choice(people),
        place=random.choice(places),
        product=random.choice(products),
        subject=random.choice(subjects),
        revenue=f"{revenue:,}",
        expenses=f"{expenses:,}",
    )


def main() -> None:
    lines = []
    size = 0

    while size < TARGET:
        line = make_line()

        lines.append(line)
        size += len(line.encode("utf-8")) + 1

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    text = "\n".join(lines) + "\n"

    OUTPUT.write_text(
        text,
        encoding="utf-8",
    )

    print("GNOME FEED MILL")
    print("----------------")
    print("Lines:", len(lines))
    print(
        "Bytes:",
        len(text.encode("utf-8")),
    )
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
