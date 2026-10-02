"""Tiny word tokenizer for the GNOMEdata MicroLLM."""

import re
from dataclasses import dataclass


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass
class Tokenizer:
    """Minimal trainable vocabulary."""

    token_to_id: dict[str, int]
    id_to_token: dict[int, str]

    @classmethod
    def train(cls, text: str) -> "Tokenizer":
        tokens = cls.split(text)

        vocabulary = ["<UNK>"] + sorted(set(tokens))

        token_to_id = {
            token: index
            for index, token in enumerate(vocabulary)
        }

        id_to_token = {
            index: token
            for token, index in token_to_id.items()
        }

        return cls(token_to_id, id_to_token)

    @staticmethod
    def split(text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text.lower())

    def encode(self, text: str) -> list[int]:
        unknown = self.token_to_id["<UNK>"]

        return [
            self.token_to_id.get(token, unknown)
            for token in self.split(text)
        ]

    def decode(self, ids: list[int]) -> str:
        tokens = [
            self.id_to_token.get(index, "<UNK>")
            for index in ids
        ]

        text = " ".join(tokens)

        # Basic punctuation cleanup.
        for mark in ".,!?;:":
            text = text.replace(" " + mark, mark)

        return text

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)
