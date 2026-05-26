"""Port: cliente LLM.

Una interfaz mínima para generar texto a partir de una lista de mensajes. El
resto del pipeline no necesita conocer OpenAI ni ningún proveedor.
"""

from __future__ import annotations

from typing import Protocol

from clase3.domain.messages import AIMessage, BaseMessage


class LLMClient(Protocol):
    """Contrato para clientes de LLM."""

    @property
    def model_name(self) -> str: ...

    def chat(self, messages: list[BaseMessage]) -> AIMessage:
        """Envía la conversación y recibe la respuesta del modelo."""
        ...
