"""Port: vector store.

Define las operaciones mínimas que necesita RAG: añadir chunks y consultar
los más similares. La implementación concreta (Chroma, FAISS, Qdrant…) se
inyecta en runtime.
"""

from __future__ import annotations

from typing import Protocol

from clase3.domain.document import Chunk, RetrievedChunk


class VectorStore(Protocol):
    """Contrato para vector stores."""

    @property
    def name(self) -> str: ...

    def add(self, chunks: list[Chunk]) -> None:
        """Indexa los chunks (embedding ya gestionado por el store o el adapter)."""
        ...

    def query(
        self,
        query_text: str,
        top_k: int = 3,
        where: dict | None = None,
    ) -> list[RetrievedChunk]:
        """Devuelve los chunks más similares a la consulta."""
        ...

    def count(self) -> int:
        """Número de chunks indexados."""
        ...
