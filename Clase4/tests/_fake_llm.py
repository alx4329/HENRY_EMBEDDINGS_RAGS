"""Fake LLM utilizable en tests: scripted o configurable.

No depende de red. Implementa el ``LLMClient`` Protocol del paquete ``clase4``.
"""

from __future__ import annotations

from collections.abc import Callable

from clase4.domain.messages import AIMessage, BaseMessage


class ScriptedLLM:
    """LLM cuyo comportamiento se define con una función ``responder``.

    La función recibe la lista de mensajes y devuelve el texto a producir.
    Si no se provee, responde con la concatenación de los contenidos.
    """

    def __init__(
        self,
        responder: Callable[[list[BaseMessage]], str] | None = None,
        *,
        model_name: str = "fake-llm",
    ) -> None:
        self._responder = responder or (lambda msgs: "\n".join(m.content for m in msgs))
        self._model_name = model_name
        self.calls: list[list[BaseMessage]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    def chat(
        self,
        messages: list[BaseMessage],
        *,
        temperature: float | None = None,
    ) -> AIMessage:
        self.calls.append(list(messages))
        return AIMessage(content=self._responder(messages))
