"""Tests for GNOMEdata's local intelligence layer."""

from app.brain.entities import extract_entities
from app.brain.facts import extract_facts


def test_extracts_named_entities() -> None:
    entities = extract_entities(
        "Randy lives in Oakville."
    )

    values = {
        entity.text
        for entity in entities
    }

    assert "Randy" in values
    assert "Oakville" in values


def test_extracts_financial_entities() -> None:
    entities = extract_entities(
        "Revenue was $42,000 in 2026 with growth of 12.5%."
    )

    pairs = {
        (entity.text, entity.kind)
        for entity in entities
    }

    assert ("$42,000", "money") in pairs
    assert ("2026", "year") in pairs
    assert ("12.5%", "percent") in pairs


def test_extracts_randy_location_fact() -> None:
    facts = extract_facts(
        "Randy lives on Willowstreet."
    )

    assert any(
        fact.subject.lower() == "randy"
        and fact.relation == "lives_at"
        and "Willowstreet" in fact.value
        for fact in facts
    )


def test_extracts_named_pet_fact() -> None:
    facts = extract_facts(
        "Randy has a dog named Tom."
    )

    assert any(
        fact.subject.lower() == "randy"
        and fact.relation == "has_named"
        and fact.value.lower() == "tom"
        for fact in facts
    )


def test_extracts_sales_fact() -> None:
    facts = extract_facts(
        "Randy sells pies from his home in Oakville."
    )

    assert any(
        fact.subject.lower() == "randy"
        and fact.relation == "sells"
        and "pies" in fact.value.lower()
        for fact in facts
    )


def test_query_result_can_hold_local_answer() -> None:
    from app.core.query import QueryResult

    result = QueryResult(
        question="Where does Randy sell pies?",
        collection_id="default",
        evidence=(),
        answer="Randy sells pies from his home in Oakville.",
    )

    assert result.answer == (
        "Randy sells pies from his home in Oakville."
    )


def test_query_engine_synthesizes_sales_answer() -> None:
    from app.core.query import (
        LocalQueryEngine,
        QueryFact,
    )

    facts = (
        QueryFact(
            subject="Randy",
            relation="sells",
            value="pies from his home in Oakville",
            source_name="RandyRecords",
            board_id="board-1",
        ),
    )

    answer = LocalQueryEngine._answer_from_facts(
        "Where does Randy sell pies?",
        facts,
    )

    assert answer == (
        "Randy sells pies from his home in Oakville."
    )


def test_query_result_can_hold_local_answer() -> None:
    from app.core.query import QueryResult

    result = QueryResult(
        question="Where does Randy sell pies?",
        collection_id="default",
        evidence=(),
        answer="Randy sells pies from his home in Oakville.",
    )

    assert result.answer == (
        "Randy sells pies from his home in Oakville."
    )


def test_query_engine_synthesizes_sales_answer() -> None:
    from app.core.query import (
        LocalQueryEngine,
        QueryFact,
    )

    facts = (
        QueryFact(
            subject="Randy",
            relation="sells",
            value="pies from his home in Oakville",
            source_name="RandyRecords",
            board_id="board-1",
        ),
    )

    answer = LocalQueryEngine._answer_from_facts(
        "Where does Randy sell pies?",
        facts,
    )

    assert answer == (
        "Randy sells pies from his home in Oakville."
    )
