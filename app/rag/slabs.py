
"""The Slab Catcher: identify extraction waste without losing source data."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Slab:
    """A passage flagged as possible extraction waste."""

    log_id: str
    text: str
    reason: str
    start_offset: int
    end_offset: int


@dataclass(frozen=True)
class SlabReport:
    """Results of inspecting a source log."""

    clean_text: str
    slabs: tuple[Slab, ...]


def catch_slabs(log_id: str, text: str) -> SlabReport:
    """Remove obvious extraction clutter and record every removed line.

    Offsets refer to character positions in the original text.
    The original text is never modified.
    """
    if not log_id.strip():
        raise ValueError("log_id cannot be empty.")

    kept: list[str] = []
    slabs: list[Slab] = []
    offset = 0

    lines = text.splitlines(keepends=True)

    for line in lines:
        content = line.rstrip("\r\n")
        stripped = content.strip()
        start = offset
        end = offset + len(line)
        reason = _slab_reason(stripped)

        if reason:
            slabs.append(
                Slab(
                    log_id=log_id,
                    text=content,
                    reason=reason,
                    start_offset=start,
                    end_offset=end,
                )
            )
        else:
            kept.append(line)

        offset = end

    return SlabReport(
        clean_text="".join(kept),
        slabs=tuple(slabs),
    )


def _slab_reason(line: str) -> str | None:
    """Return a reason when a line looks like extraction waste."""
    if not line:
        return "empty_line"

    if line.startswith("\ufffd"):
        return "possible_text_corruption"

    if line.isdigit():
        return "standalone_page_number"

    normalized = " ".join(line.lower().split())

    if normalized in {
        "table of contents",
        "all rights reserved",
        "confidential",
    }:
        return "possible_repeated_boilerplate"

    return None

