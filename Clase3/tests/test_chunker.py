"""Tests del servicio de chunking (no requieren red ni API key)."""

from __future__ import annotations

from clase3.domain.document import Document
from clase3.services.chunker import TextChunker


def _doc(text: str) -> Document:
    return Document(content=text, metadata={"title": "Test"}, id="test")


def test_short_document_emits_single_chunk() -> None:
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.split(_doc("# Título\n\nPárrafo único."))
    assert len(chunks) == 1
    assert "Título" in chunks[0].content


def test_markdown_sections_become_separate_chunks() -> None:
    text = (
        "# Documento\n\n"
        "## Sección A\n\nContenido A muy breve.\n\n"
        "## Sección B\n\nContenido B también breve.\n\n"
        "## Sección C\n\nContenido C corto.\n"
    )
    chunker = TextChunker(chunk_size=400, chunk_overlap=50)
    chunks = chunker.split(_doc(text))
    sections = [c.metadata.get("section") for c in chunks]
    # Preámbulo + tres secciones H2
    assert "Sección A" in sections
    assert "Sección B" in sections
    assert "Sección C" in sections


def test_long_section_triggers_sliding_window() -> None:
    long_text = "## Larga\n\n" + ("párrafo de prueba. " * 200)
    chunker = TextChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.split(_doc(long_text))
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata.get("section") == "Larga"


def test_chunk_indices_are_sequential() -> None:
    text = "## A\n\n" + "x " * 800 + "\n\n## B\n\n" + "y " * 800
    chunker = TextChunker(chunk_size=400, chunk_overlap=50)
    chunks = chunker.split(_doc(text))
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_invalid_overlap_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        TextChunker(chunk_size=200, chunk_overlap=200)
