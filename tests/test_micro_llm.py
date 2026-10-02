"""Tests for GNOMEdata's local micro-LLM boundary."""

from app.llm import MicroLLM


def test_micro_llm_identifies_itself() -> None:
    model = MicroLLM()

    assert model.model_name == "gnome-micro-99k"


def test_micro_llm_uses_context() -> None:
    model = MicroLLM()

    response = model.generate(
        "What does Randy sell?",
        context=(
            "Randy sells pies from his home.",
        ),
    )

    assert "Randy sells pies" in response.text
    assert response.context_used == 1
    assert response.model == "gnome-micro-99k"


def test_micro_llm_handles_missing_context() -> None:
    model = MicroLLM()

    response = model.generate(
        "What powers the machine?",
    )

    assert response.context_used == 0
    assert response.text


def test_micro_llm_handles_empty_prompt() -> None:
    model = MicroLLM()

    response = model.generate("   ")

    assert response.text
    assert response.context_used == 0


def test_micro_llm_can_receive_multiple_boards() -> None:
    model = MicroLLM()

    response = model.generate(
        "Tell me about Randy.",
        context=(
            "Randy sells pies.",
            "Randy owns a dog named Tom.",
        ),
    )

    assert response.context_used == 2
    assert "Randy sells pies" in response.text
    assert "Tom" in response.text


def test_micro_llm_can_receive_multiple_boards() -> None:
    model = MicroLLM()

    response = model.generate(
        "Tell me about Randy.",
        context=(
            "Randy sells pies.",
            "Randy owns a dog named Tom.",
        ),
    )

    assert response.context_used == 2
    assert "Randy sells pies" in response.text
    assert "Tom" in response.text
