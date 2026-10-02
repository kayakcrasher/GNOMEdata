"""Local entity extraction for the GNOMEdata Brain.

This module deliberately uses standard-library Python.
It identifies useful candidate entities without requiring
a remote model or external NLP service.
"""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Entity:
    """One entity discovered in source text."""

    text: str
    kind: str
    start: int
    end: int


CAPITALIZED_PHRASE = re.compile(
    r"\b(?:[A-Z][A-Za-z'-]*)(?:\s+[A-Z][A-Za-z'-]*)*\b"
)

MONEY = re.compile(
    r"(?<!\w)\$\s?\d[\d,]*(?:\.\d{1,2})?"
)

PERCENT = re.compile(
    r"(?<!\w)\d+(?:\.\d+)?\s?%"
)

YEAR = re.compile(
    r"\b(?:18|19|20|21)\d{2}\b"
)


def _add_matches(
    entities: list[Entity],
    pattern: re.Pattern[str],
    text: str,
    kind: str,
) -> None:
    """Append regex matches as entities."""

    for match in pattern.finditer(text):
        entities.append(
            Entity(
                text=match.group(0),
                kind=kind,
                start=match.start(),
                end=match.end(),
            )
        )


def extract_entities(text: str) -> tuple[Entity, ...]:
    """Extract useful candidate entities from local text."""

    if not text.strip():
        return ()

    entities: list[Entity] = []

    _add_matches(entities, MONEY, text, "money")
    _add_matches(entities, PERCENT, text, "percent")
    _add_matches(entities, YEAR, text, "year")
    _add_matches(
        entities,
        CAPITALIZED_PHRASE,
        text,
        "named",
    )

    # Remove exact duplicate spans.
    unique: dict[tuple[int, int, str], Entity] = {}

    for entity in entities:
        key = (
            entity.start,
            entity.end,
            entity.kind,
        )
        unique[key] = entity

    return tuple(
        sorted(
            unique.values(),
            key=lambda item: (
                item.start,
                item.end,
                item.kind,
            ),
        )
    )
