"""Test de integración del pipeline RAG usando fakes (sin red).

Demuestra que la arquitectura SOLID realmente permite sustituir cualquier
adapter por un fake en tiempo de test, sin tocar el resto del código.
"""

from __future__ import annotations

from dataclasses import dataclass

from clase3.domain.document import Chunk, RetrievedChunk
from clase3.domain.messages import AIMessage, BaseMessage
from clase3.services.augmenter import PromptAugmenter
from clase3.services.generator import AnswerGenerator
from clase3.services.rag_pipeline import RAGPipeline
from clase3.services.retriever import Retriever


@dataclass
class _FakeStore:
    name: str = "fake"
    _chunks: list[Chunk] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._chunks = self._chunks or [
            Chunk(
                content="Rorschach es un vigilante absolutista de Watchmen.",
                document_id="watchmen",
                chunk_index=0,
                metadata={"title": "Watchmen", "source": "watchmen.md"},
            ),
            Chunk(
                content="Persépolis narra la vida de Marjane Satrapi en Teherán.",
                document_id="persepolis",
                chunk_index=0,
                metadata={"title": "Persépolis", "source": "persepolis.md"},
            ),
        ]

    def add(self, chunks: list[Chunk]) -> None:  # pragma: no cover
        self._chunks.extend(chunks)

    def query(
        self, query_text: str, top_k: int = 3, where=None
    ) -> list[RetrievedChunk]:
        scored = [
            (
                len(set(query_text.lower().split()) & set(chunk.content.lower().split())),
                chunk,
            )
            for chunk in self._chunks
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievedChunk(chunk=chunk, distance=1.0 - (score / 5))
            for score, chunk in scored[:top_k]
        ]

    def count(self) -> int:
        return len(self._chunks)


class _FakeLLM:
    model_name = "fake-llm"

    def chat(self, messages: list[BaseMessage]) -> AIMessage:
        user_content = messages[-1].content
        # Extraer la pregunta de la sección "## Pregunta"
        question = ""
        for block in user_content.split("##"):
            if block.lstrip().lower().startswith("pregunta"):
                question = block.split("\n", 1)[1] if "\n" in block else ""
                break
        question = question.strip()
        if "Rorschach" in question:
            return AIMessage(content="Rorschach es un vigilante de Watchmen (Watchmen).")
        if "Marjane" in question:
            return AIMessage(content="Marjane es la narradora de Persépolis (Persépolis).")
        return AIMessage(content="No lo sé.")


def test_pipeline_end_to_end_with_fakes() -> None:
    store = _FakeStore()
    retriever = Retriever(store=store, top_k=2)  # type: ignore[arg-type]
    augmenter = PromptAugmenter()
    generator = AnswerGenerator(llm=_FakeLLM())  # type: ignore[arg-type]
    pipeline = RAGPipeline(retriever=retriever, augmenter=augmenter, generator=generator)

    result = pipeline.ask("¿Quién es Rorschach?")
    assert "Watchmen" in result.answer
    assert any(item.chunk.document_id == "watchmen" for item in result.retrieved)


def test_pipeline_retriever_is_exposed_publicly() -> None:
    store = _FakeStore()
    retriever = Retriever(store=store, top_k=2)  # type: ignore[arg-type]
    augmenter = PromptAugmenter()
    generator = AnswerGenerator(llm=_FakeLLM())  # type: ignore[arg-type]
    pipeline = RAGPipeline(retriever=retriever, augmenter=augmenter, generator=generator)

    only_retrieved = pipeline.retrieve_only("Marjane")
    assert only_retrieved
    assert only_retrieved[0].chunk.document_id == "persepolis"
    assert pipeline.retriever is retriever


def test_pipeline_falls_back_when_no_match() -> None:
    store = _FakeStore()
    retriever = Retriever(store=store, top_k=2)  # type: ignore[arg-type]
    augmenter = PromptAugmenter()
    generator = AnswerGenerator(llm=_FakeLLM())  # type: ignore[arg-type]
    pipeline = RAGPipeline(retriever=retriever, augmenter=augmenter, generator=generator)
    result = pipeline.ask("¿Pregunta totalmente sin relación?")
    # El FakeLLM responde "No lo sé" cuando no detecta keywords conocidos.
    assert "no lo sé" in result.answer.lower()
