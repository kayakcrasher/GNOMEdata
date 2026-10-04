"""Structured 4chan thread extraction for GNOMEdata."""

from dataclasses import dataclass
from html.parser import HTMLParser
import re


@dataclass(frozen=True)
class FourChanPost:
    """A single extracted 4chan post."""

    post_id: str
    text: str
    reply_ids: tuple[str, ...]


class FourChanThreadParser(HTMLParser):
    """Extract blockquote.postMessage elements."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)

        self.posts: list[FourChanPost] = []

        self._inside_message = False
        self._post_id = ""
        self._parts: list[str] = []
        self._reply_ids: set[str] = set()

    def handle_starttag(
        self,
        tag: str,
        attrs,
    ) -> None:
        tag = tag.lower()
        attributes = dict(attrs)

        classes = set(
            attributes.get(
                "class",
                "",
            ).split()
        )

        if (
            not self._inside_message
            and tag == "blockquote"
            and "postMessage" in classes
        ):
            raw_id = attributes.get(
                "id",
                "",
            )

            match = re.fullmatch(
                r"m(\d+)",
                raw_id,
            )

            if not match:
                return

            self._inside_message = True
            self._post_id = match.group(1)
            self._parts = []
            self._reply_ids = set()
            return

        if not self._inside_message:
            return

        if tag == "br":
            self._parts.append("\n")
            return

        if tag == "a":
            href = attributes.get(
                "href",
                "",
            )

            match = re.search(
                r"#p(\d+)",
                href,
            )

            if match:
                self._reply_ids.add(
                    match.group(1)
                )

    def handle_endtag(
        self,
        tag: str,
    ) -> None:
        if (
            not self._inside_message
            or tag.lower() != "blockquote"
        ):
            return

        text = "".join(
            self._parts
        )

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        text = re.sub(
            r" *\n *",
            "\n",
            text,
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        text = text.strip()

        if text:
            self.posts.append(
                FourChanPost(
                    post_id=self._post_id,
                    text=text,
                    reply_ids=tuple(
                        sorted(
                            self._reply_ids,
                            key=int,
                        )
                    ),
                )
            )

        self._inside_message = False
        self._post_id = ""
        self._parts = []
        self._reply_ids = set()

    def handle_data(
        self,
        data: str,
    ) -> None:
        if self._inside_message:
            self._parts.append(data)


def extract_posts(
    html: str,
) -> tuple[FourChanPost, ...]:
    """Extract structured posts from a 4chan thread."""

    parser = FourChanThreadParser()
    parser.feed(html)

    return tuple(parser.posts)
