"""Tipos de mensaje para el LLM. Análogos a los del SDK de OpenAI pero
desacoplados para que el dominio no dependa del proveedor concreto."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Role = Literal["system", "user", "assistant"]


class BaseMessage(BaseModel):
    """Mensaje base con role y content.

    El campo ``role`` se restringe a los valores válidos para la API de chat;
    las subclases lo fijan a un Literal concreto mediante el default.
    """

    role: Role
    content: str

    def to_openai(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class SystemMessage(BaseMessage):
    role: Role = "system"


class UserMessage(BaseMessage):
    role: Role = "user"


class AIMessage(BaseMessage):
    role: Role = "assistant"
