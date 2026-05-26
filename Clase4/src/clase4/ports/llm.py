"""Port: cliente LLM."""

from __future__ import annotations

from typing import Protocol

from clase4.domain.messages import AIMessage, BaseMessage


class LLMClient(Protocol):
    @property
    def model_name(self) -> str: ...

    def chat(
        self,
        messages: list[BaseMessage],
        *,
        temperature: float | None = None,
    ) -> AIMessage:
        """Envía la conversación y devuelve la respuesta del modelo."""
        ...
