"""Build concise factual-answer training for GNOME Micro."""

from pathlib import Path
import random


OUTPUT = Path(
    "data/training_compression/compression.txt"
)

rng = random.Random(187)


names = [
    "Randy", "Maya", "Luis", "Sarah",
    "Daniel", "Nina", "Marcus", "Helen",
    "Zelda", "Theo", "Iris", "Caleb",
]

cities = [
    "Oakville", "Chicago", "Memphis",
    "Denver", "Austin", "Boston",
    "Seattle", "Phoenix",
]

products = [
    "pies", "bicycles", "books", "lamps",
    "coffee", "tools", "chairs", "baskets",
]

animals = [
    "dog", "cat", "parrot", "rabbit",
]

pet_names = [
    "Tom", "Rex", "Blue", "Milo",
    "Pepper", "Luna", "Scout", "Ruby",
]

jobs = [
    "mechanic", "teacher", "baker",
    "carpenter", "engineer", "nurse",
]

locations = [
    "home",
    "garage",
    "workshop",
    "store",
    "market stall",
]

lessons = []


def add(question, evidence, answer):
    lesson = (
        f"QUESTION: {question}\n"
        f"EVIDENCE: {evidence}\n"
        f"ANSWER: {answer}\n"
    )

    if len(lesson.encode("utf-8")) <= 500:
        lessons.append(lesson)


# --------------------------------------------------
# Product extraction
# --------------------------------------------------

for _ in range(500):
    name = rng.choice(names)
    city = rng.choice(cities)
    product = rng.choice(products)
    animal = rng.choice(animals)
    pet = rng.choice(pet_names)
    location = rng.choice(locations)

    evidence = (
        f"{name} lives in {city}. "
        f"{name} has a {animal} named {pet}. "
        f"{name} sells {product} from the {location}. "
        f"The weather is usually mild."
    )

    add(
        f"What does {name} sell?",
        evidence,
        f"{name} sells {product}.",
    )


# --------------------------------------------------
# Location extraction
# --------------------------------------------------

for _ in range(350):
    name = rng.choice(names)
    city = rng.choice(cities)
    product = rng.choice(products)
    job = rng.choice(jobs)

    evidence = (
        f"{name} works as a {job}. "
        f"{name} sells {product}. "
        f"{name} lives in {city}."
    )

    add(
        f"Where does {name} live?",
        evidence,
        f"{name} lives in {city}.",
    )


# --------------------------------------------------
# Pet extraction
# --------------------------------------------------

for _ in range(300):
    name = rng.choice(names)
    city = rng.choice(cities)
    animal = rng.choice(animals)
    pet = rng.choice(pet_names)
    product = rng.choice(products)

    evidence = (
        f"{name} lives in {city}. "
        f"{name} sells {product}. "
        f"{name} has a {animal} named {pet}."
    )

    add(
        f"What is the name of {name}'s {animal}?",
        evidence,
        f"{name}'s {animal} is named {pet}.",
    )


# --------------------------------------------------
# Occupation extraction
# --------------------------------------------------

for _ in range(300):
    name = rng.choice(names)
    city = rng.choice(cities)
    job = rng.choice(jobs)
    product = rng.choice(products)

    evidence = (
        f"{name} lives in {city}. "
        f"{name} works as a {job}. "
        f"{name} sometimes sells {product}."
    )

    add(
        f"What is {name}'s job?",
        evidence,
        f"{name} is a {job}.",
    )


# --------------------------------------------------
# Missing information
# --------------------------------------------------

for _ in range(250):
    name = rng.choice(names)
    city = rng.choice(cities)
    product = rng.choice(products)

    evidence = (
        f"{name} lives in {city}. "
        f"{name} sells {product}."
    )

    add(
        f"What is {name}'s favorite color?",
        evidence,
        "The evidence does not say.",
    )


# --------------------------------------------------
# Two-fact compression
# --------------------------------------------------

for _ in range(300):
    name = rng.choice(names)
    city = rng.choice(cities)
    product = rng.choice(products)
    animal = rng.choice(animals)
    pet = rng.choice(pet_names)

    evidence = (
        f"{name} owns a {animal} named {pet}. "
        f"{name} lives in {city}. "
        f"{name} sells {product}."
    )

    add(
        f"Where does {name} live and what does "
        f"{name} sell?",
        evidence,
        f"{name} lives in {city} and sells {product}.",
    )


rng.shuffle(lessons)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT.write_text(
    "\n".join(lessons),
    encoding="utf-8",
)

print("GNOME COMPRESSION SCHOOL")
print("------------------------")
print("Lessons:", f"{len(lessons):,}")
print("Bytes:", f"{OUTPUT.stat().st_size:,}")
print("Output:", OUTPUT)
