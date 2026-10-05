"""GNOME Micro inference provider."""

from pathlib import Path

from app.inference.base import InferenceProvider
from app.inference.types import (
    InferenceRequest,
    InferenceResult,
    ProviderCapabilities,
)
from app.micro_llm.transformer.generate import (
    generate,
    load_model,
)


class GnomeProvider(InferenceProvider):
    """Run GNOME Micro through the standard inference interface."""

    name = "gnome"

    def __init__(
        self,
        model_path: str = "models/gnome-v06-best.npz",
    ) -> None:
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"GNOME checkpoint not found: {self.model_path}"
            )

        self.model = load_model(self.model_path)
        self.model_name = self.model_path.stem

    def generate(
        self,
        request: InferenceRequest,
    ) -> InferenceResult:
        """Generate text using GNOME Micro."""

        prompt = request.prompt

        if request.system_prompt:
            prompt = (
                f"SYSTEM: {request.system_prompt}\n"
                f"{request.prompt}"
            )

        output = generate(
            model=self.model,
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            seed=42,
        )

        decoded = output.decode(
            "utf-8",
            errors="replace",
        )

        # generate() currently returns prompt + completion.
        if decoded.startswith(prompt):
            text = decoded[len(prompt):]
        else:
            text = decoded

        return InferenceResult(
            text=text.strip(),
            provider=self.name,
            model=self.model_name,
            metadata={
                "checkpoint": str(self.model_path),
                "parameter_count": self.model.parameter_count,
            },
        )

    def health(self) -> bool:
        """GNOME is healthy when its checkpoint is loaded."""

        return self.model is not None

    def capabilities(
        self,
    ) -> ProviderCapabilities:
        return ProviderCapabilities(
            generation=True,
            chat=False,
            embeddings=False,
            tools=False,
            structured_output=False,
        )
