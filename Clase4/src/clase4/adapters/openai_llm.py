"""Adapter: cliente OpenAI Chat Completions."""

from __future__ import annotations

from openai import OpenAI

from clase4.domain.messages import AIMessage, BaseMessage


class OpenAIChatClient:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        temperature: float = 0.4,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()

    @property
    def model_name(self) -> str:
        return self._model

    def chat(
        self,
        messages: list[BaseMessage],
        *,
        temperature: float | None = None,
    ) -> AIMessage:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[m.to_openai() for m in messages],
            temperature=temperature if temperature is not None else self._temperature,
        )
        content = response.choices[0].message.content or ""
        return AIMessage(content=content)
