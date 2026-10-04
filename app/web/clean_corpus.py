"""Clean Web Eye captures into MicroLLM training text."""

import argparse
import re
from pathlib import Path


NOISE_PATTERNS = [
    re.compile(r"^GNOMEdata WEB EYE$"),
    re.compile(r"^-{3,}$"),
    re.compile(r"^Fetching:"),
    re.compile(r"^Status:"),
    re.compile(r"^Title:"),
    re.compile(r"^Characters:"),
    re.compile(r"^Links:"),
    re.compile(r"^CONTENT$"),
    re.compile(r"^FIRST LINKS$"),
    re.compile(r"^\.\.\. \[truncated\]$"),
    re.compile(r"^https?://\S+$"),
]


def is_noise(line: str) -> bool:
    stripped = line.strip()

    if not stripped:
        return False

    return any(
        pattern.search(stripped)
        for pattern in NOISE_PATTERNS
    )


def clean_text(text: str) -> str:
    """Remove Web Eye/UI noise while preserving discussion text."""

    lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if is_noise(line):
            continue

        # Normalize excessive whitespace.
        line = re.sub(
            r"\s+",
            " ",
            line,
        )

        # Remove common post timestamp-only lines.
        if re.fullmatch(
            r"\d{2}/\d{2}/\d{2}\([A-Za-z]{3}\)"
            r"\d{2}:\d{2}:\d{2}",
            line,
        ):
            continue

        lines.append(line)

    # Collapse excessive blank lines.
    output = "\n".join(lines)

    output = re.sub(
        r"\n{3,}",
        "\n\n",
        output,
    )

    return output.strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "inputs",
        nargs="+",
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    documents = []

    for filename in args.inputs:
        path = Path(filename)

        cleaned = clean_text(
            path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        )

        documents.append(cleaned)

        print(
            f"{path}: "
            f"{len(cleaned):,} cleaned characters"
        )

    combined = "\n\n".join(
        documents
    )

    output = Path(args.output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        combined,
        encoding="utf-8",
    )

    print()
    print(
        f"Saved {len(combined):,} characters "
        f"to {output}"
    )


if __name__ == "__main__":
    main()
