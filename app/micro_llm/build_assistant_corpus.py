"""Build compact training lessons for GNOME Micro document assistance."""

from pathlib import Path
import random


OUTPUT = Path(
    "data/training_assistant/curriculum.txt"
)

rng = random.Random(2026)

names = [
    "Randy", "Maya", "Luis", "Sarah",
    "Daniel", "Nina", "Marcus", "Helen",
]

cities = [
    "Oakville", "Denver", "Boston", "Memphis",
    "Austin", "Phoenix", "Seattle", "Atlanta",
]

products = [
    "pies", "chairs", "tools", "books",
    "cables", "lamps", "baskets", "coffee",
]

databases = [
    "SQLite", "PostgreSQL", "MariaDB",
]

languages = [
    "Python", "Rust", "Java", "Go",
]

days = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday",
]

lessons = []


def add(question, evidence, answer):
    lesson = (
        f"QUESTION: {question}\n"
        f"EVIDENCE: {evidence}\n"
        f"ANSWER: {answer}\n"
    )

    size = len(
        lesson.encode("utf-8")
    )

    if size > 480:
        raise RuntimeError(
            f"Lesson exceeds budget: {size}"
        )

    lessons.append(lesson)


# -------------------------------------------------
# Direct factual extraction
# -------------------------------------------------

for _ in range(120):
    name = rng.choice(names)
    city = rng.choice(cities)

    add(
        f"Where does {name} live?",
        f"{name} lives in {city}.",
        f"{name} lives in {city}.",
    )


for _ in range(120):
    name = rng.choice(names)
    product = rng.choice(products)

    add(
        f"What does {name} sell?",
        f"{name} sells {product}.",
        f"{name} sells {product}.",
    )


# -------------------------------------------------
# Distractor resistance
# -------------------------------------------------

for _ in range(120):
    name = rng.choice(names)
    city = rng.choice(cities)
    product = rng.choice(products)

    add(
        f"Where does {name} live?",
        (
            f"{name} sells {product}. "
            f"The warehouse opens at 8 AM. "
            f"{name} lives in {city}."
        ),
        f"{name} lives in {city}.",
    )


# -------------------------------------------------
# Technical documentation
# -------------------------------------------------

for _ in range(120):
    database = rng.choice(databases)
    language = rng.choice(languages)

    add(
        "What database does the application use?",
        (
            f"The application uses {database}. "
            f"Document processing is written in {language}."
        ),
        f"The application uses {database}.",
    )


for _ in range(120):
    database = rng.choice(databases)
    language = rng.choice(languages)

    add(
        "What language processes documents?",
        (
            f"Records are stored in {database}. "
            f"{language} processes incoming documents."
        ),
        f"{language} processes incoming documents.",
    )


# -------------------------------------------------
# Simple arithmetic from evidence
# -------------------------------------------------

for _ in range(120):
    revenue = rng.randrange(
        500,
        5001,
        50,
    )

    expenses = rng.randrange(
        100,
        revenue,
        50,
    )

    profit = revenue - expenses

    add(
        "What was the profit?",
        (
            f"Revenue was ${revenue}. "
            f"Expenses were ${expenses}."
        ),
        f"Profit was ${profit}.",
    )


# -------------------------------------------------
# Missing evidence / abstention
# -------------------------------------------------

for _ in range(120):
    city = rng.choice(cities)
    day = rng.choice(days)

    add(
        "Who owns the building?",
        (
            f"The building is in {city}. "
            f"It is inspected every {day}."
        ),
        "The evidence does not identify the owner.",
    )


for _ in range(120):
    product = rng.choice(products)

    add(
        "What color is the truck?",
        (
            f"The shipment contains {product}. "
            "The truck arrived before noon."
        ),
        "The evidence does not say what color the truck is.",
    )


# -------------------------------------------------
# Conflicting evidence
# -------------------------------------------------

for _ in range(120):
    first, second = rng.sample(
        days,
        2,
    )

    add(
        "When did the shipment arrive?",
        (
            f"Report A says {first}. "
            f"Report B says {second}."
        ),
        (
            "The evidence conflicts: "
            f"one report says {first} "
            f"and another says {second}."
        ),
    )


# -------------------------------------------------
# Incident summaries
# -------------------------------------------------

for _ in range(120):
    start = rng.randint(8, 18)
    duration = rng.randint(2, 30)

    add(
        "Summarize the outage.",
        (
            f"Service failed at {start}:00. "
            f"Service returned {duration} minutes later "
            "after the network interface was restarted."
        ),
        (
            f"Service was unavailable for {duration} minutes "
            "and recovered after the network interface "
            "was restarted."
        ),
    )


# Shuffle so training does not encounter
# one task type in giant contiguous blocks.

rng.shuffle(lessons)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT.write_text(
    "\n".join(lessons),
    encoding="utf-8",
)

print("GNOME ASSISTANT CURRICULUM")
print("--------------------------")
print("Lessons:", f"{len(lessons):,}")
print(
    "Bytes:",
    f"{OUTPUT.stat().st_size:,}",
)
print("Output:", OUTPUT)
