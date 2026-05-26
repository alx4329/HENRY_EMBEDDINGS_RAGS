"""Mensajes para LLM — copia desacoplada para no depender de OpenAI."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Role = Literal["system", "user", "assistant"]


class BaseMessage(BaseModel):
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
