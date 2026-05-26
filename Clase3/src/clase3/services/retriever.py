"""Servicio: retrieval de contexto desde un VectorStore."""

from __future__ import annotations

from clase3.domain.document import RetrievedChunk
from clase3.ports.vector_store import VectorStore


class Retriever:
    """Recupera los ``top_k`` chunks más similares a una consulta."""

    def __init__(self, store: VectorStore, *, top_k: int = 3) -> None:
        self._store = store
        self._top_k = top_k

    @property
    def top_k(self) -> int:
        return self._top_k

    def retrieve(self, query: str, *, where: dict | None = None) -> list[RetrievedChunk]:
        return self._store.query(query, top_k=self._top_k, where=where)
