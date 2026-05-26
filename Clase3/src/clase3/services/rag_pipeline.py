"""Servicio: orquestador del pipeline RAG.

Conecta Retriever → PromptAugmenter → AnswerGenerator. Implementa el
patrón orquestación clásico de RAG: ``retrieve → augment → generate``.
"""

from __future__ import annotations

from dataclasses import dataclass

from clase3.domain.document import RetrievedChunk
from clase3.domain.messages import BaseMessage
from clase3.services.augmenter import PromptAugmenter
from clase3.services.generator import AnswerGenerator
from clase3.services.retriever import Retriever


@dataclass(frozen=True)
class RAGAnswer:
    question: str
    answer: str
    retrieved: list[RetrievedChunk]
    messages: list[BaseMessage]


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        augmenter: PromptAugmenter,
        generator: AnswerGenerator,
    ) -> None:
        self._retriever = retriever
        self._augmenter = augmenter
        self._generator = generator

    @property
    def retriever(self) -> Retriever:
        """Acceso público al retriever para evaluaciones y experimentos."""
        return self._retriever

    def retrieve_only(
        self, question: str, *, where: dict | None = None
    ) -> list[RetrievedChunk]:
        """Devuelve sólo los chunks recuperados, sin invocar al LLM."""
        return self._retriever.retrieve(question, where=where)

    def ask(self, question: str, *, where: dict | None = None) -> RAGAnswer:
        retrieved = self._retriever.retrieve(question, where=where)
        messages = self._augmenter.build_messages(question, retrieved)
        ai_message = self._generator.generate(messages)
        return RAGAnswer(
            question=question,
            answer=ai_message.content,
            retrieved=retrieved,
            messages=[*messages, ai_message],
        )
