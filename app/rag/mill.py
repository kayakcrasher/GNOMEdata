"""Shared products produced by the GNOMEdata mill.

The physical stages of the mill live in focused modules:

boards.py
    Cuts contextual boards from source logs.

slabs.py
    Catches unusable or suspicious material.

grading.py
    Inspects board quality.

stacker.py
    Sorts usable lumber.

yard.py
    Stores and retrieves finished lumber.

This module contains mill products that do not yet belong to
their own processing stage.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RailroadTie:
    """A precise structured fact extracted from source material.

    Boards preserve contextual passages.

    Railroad ties preserve individual facts that may eventually
    support structured retrieval alongside semantic board search.
    """

    tie_id: str
    log_id: str
    source_name: str
    subject: str
    relation: str
    value: str
