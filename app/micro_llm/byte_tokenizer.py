"""UTF-8 byte tokenizer for GNOME Micro.

Every possible byte has its own token ID, so arbitrary UTF-8
input can be represented without an unknown-token vocabulary.
"""


class ByteTokenizer:
    """Encode text as UTF-8 byte token IDs."""

    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        data = bytes(
            token
            for token in tokens
            if 0 <= token <= 255
        )

        return data.decode(
            "utf-8",
            errors="replace",
        )
