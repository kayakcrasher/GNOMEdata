"""Build supervised GNOME Micro lessons from approved webpages."""

import argparse
import re
from pathlib import Path

from app.web.fetcher import WebFetcher


def normalize(text: str) -> str:
    """Normalize webpage text without destroying sentence structure."""

    text = text.replace("\r", "\n")

    lines = []

    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()

        if len(line) < 20:
            continue

        lines.append(line)

    return "\n".join(lines)


def sentences(text: str) -> list[str]:
    """Split text into simple sentence-like units."""

    pieces = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    return [
        piece.strip()
        for piece in pieces
        if len(piece.strip()) >= 30
    ]


def useful_sentence(text: str) -> bool:
    """Reject navigation, UI noise, and malformed fragments."""

    lowered = text.lower()

    noise = (
        "navigation index",
        "next | previous",
        "theme auto",
        "documentation »",
        "skip to content",
        "table of contents",
    )

    if any(item in lowered for item in noise):
        return False

    if text.count("|") >= 2:
        return False

    letters = sum(
        char.isalpha()
        for char in text
    )

    if letters < 25:
        return False

    return True


def build_lessons(
    text: str,
    source: str,
) -> list[str]:
    """Create compact evidence-summary lessons."""

    units = [
        unit
        for unit in sentences(text)
        if useful_sentence(unit)
    ]

    lessons = []

    # Three sentences usually fit comfortably inside
    # GNOME's 512-byte context after formatting.
    for index in range(0, len(units) - 2, 3):
        group = units[index:index + 3]

        evidence = " ".join(group)

        # Keep enough room for QUESTION/ANSWER formatting.
        if len(evidence.encode("utf-8")) > 330:
            continue

        # Extractive summary is intentional for v0.5:
        # the first sentence provides reliable supervision.
        answer = group[0]

        lesson = (
            "QUESTION: Summarize this information briefly.\n"
            f"EVIDENCE: {evidence}\n"
            f"ANSWER: {answer}\n"
        )

        if len(lesson.encode("utf-8")) > 500:
            continue

        lessons.append(lesson)

    return lessons


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "urls",
        nargs="+",
        help="Approved HTTP/HTTPS webpages.",
    )

    parser.add_argument(
        "--output",
        default="data/training_web/web_school.txt",
    )

    parser.add_argument(
        "--max-lessons-per-page",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    fetcher = WebFetcher()

    all_lessons = []

    for url in args.urls:
        print()
        print("FETCHING")
        print(url)

        page = fetcher.fetch(url)

        cleaned = normalize(page.text)

        lessons = build_lessons(
            cleaned,
            page.url,
        )

        lessons = lessons[
            :args.max_lessons_per_page
        ]

        all_lessons.extend(lessons)

        print("Title:", page.title)
        print(
            "Clean characters:",
            f"{len(cleaned):,}",
        )
        print(
            "Lessons:",
            f"{len(lessons):,}",
        )

    output = Path(args.output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        "\n".join(all_lessons),
        encoding="utf-8",
    )

    print()
    print("GNOME WEB SCHOOL")
    print("----------------")
    print(
        "Lessons:",
        f"{len(all_lessons):,}",
    )
    print(
        "Bytes:",
        f"{output.stat().st_size:,}",
    )
    print(
        "Output:",
        output,
    )


if __name__ == "__main__":
    main()
