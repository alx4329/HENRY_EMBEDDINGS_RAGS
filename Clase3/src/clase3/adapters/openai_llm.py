"""Adapter: cliente LLM sobre la API de chat de OpenAI."""

from __future__ import annotations

from openai import OpenAI

from clase3.domain.messages import AIMessage, BaseMessage


class OpenAIChatClient:
    """Implementación de ``LLMClient`` para los modelos chat de OpenAI."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        temperature: float = 0.2,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()

    @property
    def model_name(self) -> str:
        return self._model

    def chat(self, messages: list[BaseMessage]) -> AIMessage:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[m.to_openai() for m in messages],
            temperature=self._temperature,
        )
        content = response.choices[0].message.content or ""
        return AIMessage(content=content)
