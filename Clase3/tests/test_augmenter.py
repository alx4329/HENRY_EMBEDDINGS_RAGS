"""Tests del PromptAugmenter (sin red, sin API key)."""

from __future__ import annotations

from clase3.domain.document import Chunk, RetrievedChunk
from clase3.services.augmenter import PromptAugmenter


def _retrieved(text: str, *, title: str, distance: float = 0.2) -> RetrievedChunk:
    chunk = Chunk(
        content=text,
        document_id="doc",
        chunk_index=0,
        metadata={"title": title, "source": f"{title}.md"},
    )
    return RetrievedChunk(chunk=chunk, distance=distance)


def test_builds_system_and_user_messages() -> None:
    aug = PromptAugmenter()
    msgs = aug.build_messages(
        "¿Qué es X?",
        [_retrieved("X es un concepto teórico.", title="ConceptoX")],
    )
    assert msgs[0].role == "system"
    assert msgs[1].role == "user"
    assert "ConceptoX" in msgs[1].content
    assert "X es un concepto teórico" in msgs[1].content


def test_handles_empty_context() -> None:
    aug = PromptAugmenter()
    msgs = aug.build_messages("¿Algo?", [])
    assert "sin contexto" in msgs[1].content


def test_custom_system_prompt_used() -> None:
    aug = PromptAugmenter(system_prompt="Eres pirata.")
    msgs = aug.build_messages("¿Por qué?", [])
    assert msgs[0].content == "Eres pirata."
