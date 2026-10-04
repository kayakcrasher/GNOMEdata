"""Command-line webpage reader for GNOMEdata."""

import argparse

from app.web.fetcher import WebFetcher


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read a webpage with GNOMEdata."
        )
    )

    parser.add_argument(
        "url",
        help="HTTP or HTTPS webpage to read.",
    )

    parser.add_argument(
        "--preview",
        type=int,
        default=1500,
        help="Maximum text characters to display.",
    )

    args = parser.parse_args()

    fetcher = WebFetcher()

    print("GNOMEdata WEB EYE")
    print("-----------------")
    print(f"Fetching: {args.url}")
    print()

    page = fetcher.fetch(
        args.url
    )

    print(f"Status: {page.status}")
    print(
        f"Title: "
        f"{page.title or '(no title)'}"
    )
    print(
        f"Characters: {len(page.text):,}"
    )
    print(
        f"Links: {len(page.links):,}"
    )

    print()
    print("CONTENT")
    print("-------")

    print(
        page.text[:args.preview]
    )

    if (
        len(page.text)
        > args.preview
    ):
        print()
        print("... [truncated]")
        print()

    print()
    print("FIRST LINKS")
    print("-----------")

    for link in page.links[:10]:
        print(link)


if __name__ == "__main__":
    main()
