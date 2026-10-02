"""Tiny offline GNOME runtime.

This first implementation establishes the runtime boundary.
The trained micro-model will plug into this class next.
"""

from .base import LLMResponse, LocalLLM


class MicroLLM(LocalLLM):
    """GNOMEdata's tiny offline language-model runtime."""

    def __init__(
        self,
        model_name: str = "gnome-micro-99k",
    ) -> None:
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        prompt: str,
        context: tuple[str, ...] = (),
    ) -> LLMResponse:
        question = prompt.strip()

        if not question:
            return LLMResponse(
                text="Give me something to work with.",
                model=self.model_name,
                context_used=0,
            )

        if context:
            evidence = "\n\n".join(context)

            text = (
                "I searched the Lumber Yard and found "
                "relevant local material.\n\n"
                f"{evidence}"
            )
        else:
            text = (
                "I don't have enough local evidence "
                "to answer that yet."
            )

        return LLMResponse(
            text=text,
            model=self.model_name,
            context_used=len(context),
        )
