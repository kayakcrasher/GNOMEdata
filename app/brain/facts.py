"""Fact extraction for the local GNOMEdata Brain."""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Fact:
    """A structured fact extracted from source text."""

    subject: str
    relation: str
    value: str
    sentence: str


PATTERNS = (
    (
        re.compile(
            r"\b(?P<subject>[A-Z][A-Za-z'-]*) "
            r"lives (?:on|at|in) "
            r"(?P<value>[^.!?]+)",
            re.IGNORECASE,
        ),
        "lives_at",
    ),
    (
        re.compile(
            r"\b(?P<subject>[A-Z][A-Za-z'-]*) "
            r"has (?:a|an) "
            r"(?P<object>[A-Za-z'-]+) "
            r"(?:named|called) "
            r"(?P<value>[A-Za-z'-]+)",
            re.IGNORECASE,
        ),
        "has_named",
    ),
    (
        re.compile(
            r"\b(?P<subject>[A-Z][A-Za-z'-]*) "
            r"sells "
            r"(?P<value>[^.!?]+)",
            re.IGNORECASE,
        ),
        "sells",
    ),
)


def split_sentences(text: str) -> tuple[str, ...]:
    """Perform lightweight local sentence splitting."""

    pieces = re.split(
        r"(?<=[.!?])\s+",
        text.strip(),
    )

    return tuple(
        piece.strip()
        for piece in pieces
        if piece.strip()
    )


def extract_facts(text: str) -> tuple[Fact, ...]:
    """Extract deterministic relationships from text."""

    facts: list[Fact] = []

    for sentence in split_sentences(text):
        for pattern, relation in PATTERNS:
            match = pattern.search(sentence)

            if match is None:
                continue

            facts.append(
                Fact(
                    subject=match.group("subject"),
                    relation=relation,
                    value=match.group("value").strip(),
                    sentence=sentence,
                )
            )

    return tuple(facts)
