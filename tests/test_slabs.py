
"""Tests for the Slab Catcher."""

import pytest

from app.rag.slabs import catch_slabs


def test_removes_page_numbers_and_records_them() -> None:
    source = "Pump maintenance\n42\nReplace the filter."
    report = catch_slabs("manual-1", source)

    assert report.clean_text == (
        "Pump maintenance\nReplace the filter."
    )
    assert len(report.slabs) == 1
    assert report.slabs[0].text == "42"
    assert report.slabs[0].reason == "standalone_page_number"
    assert report.slabs[0].log_id == "manual-1"


def test_flags_empty_lines_without_losing_source_offsets() -> None:
    source = "First line\n\nSecond line"
    report = catch_slabs("log-2", source)

    assert report.clean_text == "First line\nSecond line"
    assert report.slabs[0].reason == "empty_line"
    assert report.slabs[0].start_offset == 11
    assert report.slabs[0].end_offset == 12


def test_keeps_unusual_but_meaningful_text() -> None:
    source = "Warning: pressure must stay below 40 PSI."
    report = catch_slabs("manual-3", source)

    assert report.clean_text == source
    assert report.slabs == ()


def test_rejects_blank_log_id() -> None:
    with pytest.raises(ValueError):
        catch_slabs(" ", "Some source text")


def test_preserves_original_source() -> None:
    source = "Manual\n7\nKeep this instruction."
    original = source

    catch_slabs("manual-4", source)

    assert source == original

