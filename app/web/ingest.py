"""Ingest structured web threads into GNOMEdata."""

import argparse
from pathlib import Path

from app.rag.embeddings import Embedder
from app.rag.local_embedder import LocalHashEmbedder
from app.rag.pipeline import MillPipeline
from app.rag.boards import Log
from app.rag.yard import LumberYard

from app.web.fetcher import WebFetcher
from app.web.fourchan import extract_posts


def build_post_text(
    post,
    url: str,
) -> str:
    """Create provenance-preserving text for one post."""

    lines = [
        f"Source URL: {url}",
        f"Post ID: {post.post_id}",
    ]

    if post.reply_ids:
        lines.append(
            "Replies to: "
            + ", ".join(post.reply_ids)
        )

    lines.extend([
        "",
        post.text,
    ])

    return "\n".join(lines)


def ingest_thread(
    url: str,
    collection_id: str = "web",
) -> None:
    """Fetch a 4chan thread and mill each post."""

    print("GNOMEdata WEB MILL")
    print("------------------")
    print("Fetching:", url)

    page = WebFetcher().fetch(url)
    posts = extract_posts(page.html)

    print("Title:", page.title)
    print("Posts discovered:", len(posts))

    if not posts:
        raise RuntimeError(
            "No structured posts were discovered."
        )

    yard = LumberYard()

    # Web collections are created automatically on first use.
    if yard.get_collection(collection_id) is None:
        yard.create_collection(
            collection_id=collection_id,
            name=f"Web: {collection_id}",
        )

        print(
            "Created collection:",
            collection_id,
        )
    else:
        print(
            "Using collection:",
            collection_id,
        )

    embedder: Embedder = LocalHashEmbedder()

    mill = MillPipeline(
        yard=yard,
        embedder=embedder,
        collection_id=collection_id,
    )

    total_boards = 0
    total_embedded = 0

    for index, post in enumerate(
        posts,
        start=1,
    ):
        source_name = (
            f"4chan /g/ post {post.post_id}"
        )

        log = Log(
            log_id=f"web-4chan-g-{post.post_id}",
            source_name=source_name,
            text=build_post_text(
                post,
                page.url,
            ),
        )

        report = mill.process(log)

        total_boards += report.total
        total_embedded += report.embedded

        if (
            index == 1
            or index % 25 == 0
            or index == len(posts)
        ):
            print(
                f"[{index:03d}/{len(posts):03d}] "
                f"post {post.post_id} milled"
            )

    print()
    print("WEB LOAD COMPLETE")
    print("-----------------")
    print("Collection:", collection_id)
    print("Posts:", len(posts))
    print("Boards:", total_boards)
    print("Embedded:", total_embedded)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch and ingest a web thread "
            "into GNOMEdata."
        )
    )

    parser.add_argument(
        "url",
    )

    parser.add_argument(
        "--collection",
        default="web",
    )

    args = parser.parse_args()

    ingest_thread(
        args.url,
        args.collection,
    )


if __name__ == "__main__":
    main()
