"""Servicio: indexación de un corpus en un VectorStore.

Une las piezas (loader → chunker → store) pero respeta SRP: este servicio sólo
orquesta indexación. La carga, el troceado y el almacenamiento son
responsabilidades de otros componentes.
"""

from __future__ import annotations

from dataclasses import dataclass

from clase3.domain.document import Chunk
from clase3.ports.loader import DocumentLoader
from clase3.ports.vector_store import VectorStore
from clase3.services.chunker import TextChunker


@dataclass(frozen=True)
class IndexReport:
    documents: int
    chunks: int
    store_name: str
    total_in_store: int


class CorpusIndexer:
    def __init__(self, loader: DocumentLoader, chunker: TextChunker, store: VectorStore) -> None:
        self._loader = loader
        self._chunker = chunker
        self._store = store

    def index(self) -> IndexReport:
        documents = self._loader.load()
        chunks: list[Chunk] = self._chunker.split_many(documents)
        self._store.add(chunks)
        return IndexReport(
            documents=len(documents),
            chunks=len(chunks),
            store_name=self._store.name,
            total_in_store=self._store.count(),
        )
