"""Servicio: generación de la respuesta final con el LLM."""

from __future__ import annotations

from clase3.domain.messages import AIMessage, BaseMessage
from clase3.ports.llm import LLMClient


class AnswerGenerator:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def generate(self, messages: list[BaseMessage]) -> AIMessage:
        return self._llm.chat(messages)
