"""Servicio: troceado de documentos en chunks.

Estrategia híbrida:
1. Si el documento es markdown con encabezados ``##``, se trocea por sección
   (cada sección H2 se convierte en un chunk independiente, con su título
   propagado al contexto).
2. Si una sección excede ``chunk_size`` caracteres, se aplica una ventana
   deslizante con solapamiento sobre esa sección.
3. Si el documento no tiene encabezados ``##``, se usa la ventana deslizante
   sobre el documento completo.

Esto reduce los falsos negativos de "no lo sé" que aparecen cuando un chunk
parte oraciones a la mitad y pierde el sentido.
"""

from __future__ import annotations

import re

from clase3.domain.document import Chunk, Document

_SECTION_PATTERN = re.compile(r"^##\s+", re.MULTILINE)


class TextChunker:
    """Trocea documentos con respeto a la estructura semántica."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size debe ser positivo")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap debe estar en [0, chunk_size)")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, document: Document) -> list[Chunk]:
        sections = self._split_into_sections(document.content)
        if not sections:
            return self._sliding_window(document.content, document, header=None, start_index=0)

        chunks: list[Chunk] = []
        for header, body in sections:
            if len(body) <= self.chunk_size:
                chunks.append(
                    self._make_chunk(
                        content=self._prefix(header, body),
                        document=document,
                        index=len(chunks),
                        header=header,
                    )
                )
            else:
                chunks.extend(
                    self._sliding_window(
                        body,
                        document,
                        header=header,
                        start_index=len(chunks),
                    )
                )
        return chunks

    def split_many(self, documents: list[Document]) -> list[Chunk]:
        return [chunk for doc in documents for chunk in self.split(doc)]

    # ── internals ─────────────────────────────────────────────────────────────

    def _split_into_sections(self, text: str) -> list[tuple[str | None, str]]:
        """Devuelve una lista de (header, body). Si no hay '##', regresa []."""
        if not _SECTION_PATTERN.search(text):
            return []

        positions = [m.start() for m in _SECTION_PATTERN.finditer(text)]
        sections: list[tuple[str | None, str]] = []

        preamble = text[: positions[0]].strip()
        if preamble:
            header = self._extract_title(preamble)
            sections.append((header, preamble))

        for idx, start in enumerate(positions):
            end = positions[idx + 1] if idx + 1 < len(positions) else len(text)
            block = text[start:end].strip()
            header_line, _, body = block.partition("\n")
            header = header_line.lstrip("# ").strip() or None
            body_text = body.strip()
            if not body_text:
                continue
            sections.append((header, body_text))
        return sections

    def _sliding_window(
        self,
        text: str,
        document: Document,
        *,
        header: str | None,
        start_index: int,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        start = 0
        index = start_index
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            window = text[start:end]
            if end < len(text):
                soft_break = window.rfind("\n\n")
                if soft_break != -1 and soft_break > self.chunk_size // 2:
                    end = start + soft_break
                    window = text[start:end]
            chunks.append(
                self._make_chunk(
                    content=self._prefix(header, window.strip()),
                    document=document,
                    index=index,
                    header=header,
                )
            )
            index += 1
            if end >= len(text):
                break
            start = end - self.chunk_overlap
        return chunks

    def _make_chunk(
        self,
        *,
        content: str,
        document: Document,
        index: int,
        header: str | None,
    ) -> Chunk:
        metadata = dict(document.metadata)
        if header:
            metadata["section"] = header
        return Chunk(
            content=content,
            document_id=document.id,
            chunk_index=index,
            metadata=metadata,
        )

    @staticmethod
    def _prefix(header: str | None, body: str) -> str:
        if not header:
            return body
        if body.lower().startswith(header.lower()):
            return body
        return f"[{header}]\n{body}"

    @staticmethod
    def _extract_title(text: str) -> str | None:
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line.startswith("# "):
                return line.lstrip("# ").strip()
        return None
