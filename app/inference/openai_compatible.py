"""OpenAI-compatible inference provider for GNOMEdata."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.inference.base import InferenceProvider
from app.inference.types import (
    InferenceRequest,
    InferenceResult,
    ProviderCapabilities,
)


class OpenAICompatibleProvider(InferenceProvider):
    """Inference through an OpenAI-compatible HTTP server."""

    name = "openai-compatible"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = (
                f"Bearer {self.api_key}"
            )

        return headers

    def generate(
        self,
        request: InferenceRequest,
    ) -> InferenceResult:
        """Generate through /v1/chat/completions."""

        messages = []

        if request.system_prompt:
            messages.append({
                "role": "system",
                "content": request.system_prompt,
            })

        messages.append({
            "role": "user",
            "content": request.prompt,
        })

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }

        body = json.dumps(payload).encode("utf-8")

        http_request = Request(
            f"{self.base_url}/v1/chat/completions",
            data=body,
            headers=self._headers(),
            method="POST",
        )

        try:
            with urlopen(
                http_request,
                timeout=self.timeout,
            ) as response:
                result = json.loads(
                    response.read().decode("utf-8")
                )

        except HTTPError as error:
            detail = error.read().decode(
                "utf-8",
                errors="replace",
            )

            raise RuntimeError(
                f"Inference server returned HTTP "
                f"{error.code}: {detail}"
            ) from error

        except URLError as error:
            raise RuntimeError(
                "Unable to reach inference server: "
                f"{error.reason}"
            ) from error

        try:
            text = result["choices"][0]["message"]["content"]
        except (
            KeyError,
            IndexError,
            TypeError,
        ) as error:
            raise RuntimeError(
                "Inference server returned an "
                "unexpected response."
            ) from error

        usage = result.get("usage", {})

        return InferenceResult(
            text=text,
            provider=self.name,
            model=self.model_name,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get(
                "completion_tokens"
            ),
            metadata={
                "base_url": self.base_url,
            },
        )

    def health(self) -> bool:
        """Check whether the inference server responds."""

        request = Request(
            f"{self.base_url}/v1/models",
            headers=self._headers(),
            method="GET",
        )

        try:
            with urlopen(
                request,
                timeout=min(self.timeout, 5.0),
            ) as response:
                return 200 <= response.status < 300

        except (
            HTTPError,
            URLError,
            TimeoutError,
        ):
            return False

    def capabilities(
        self,
    ) -> ProviderCapabilities:
        return ProviderCapabilities(
            generation=True,
            chat=True,
            embeddings=False,
            tools=False,
            structured_output=False,
        )
