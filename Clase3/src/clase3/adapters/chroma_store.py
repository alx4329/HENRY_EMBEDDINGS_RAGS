"""Adapter: vector store basado en ChromaDB (persistente en disco).

Usa el ``EmbeddingProvider`` inyectado para generar los vectores, en vez de
delegar en la función de embeddings interna de Chroma. Esto mantiene una
sola fuente de truth para el modelo de embeddings.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings

from clase3.domain.document import Chunk, RetrievedChunk
from clase3.ports.embedder import EmbeddingProvider


class ChromaVectorStore:
    """Implementación de ``VectorStore`` con persistencia local en disco."""

    def __init__(
        self,
        collection_name: str,
        embedder: EmbeddingProvider,
        *,
        persist_directory: Path | None = None,
        reset: bool = False,
    ) -> None:
        self._collection_name = collection_name
        self._embedder = embedder

        if persist_directory is None:
            self._client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
        else:
            persist_directory = Path(persist_directory)
            persist_directory.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(persist_directory),
                settings=Settings(anonymized_telemetry=False),
            )

        if reset:
            with contextlib.suppress(Exception):
                self._client.delete_collection(collection_name)

        self._collection: Collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "embedding_model": embedder.model_name},
        )

    @property
    def name(self) -> str:
        return self._collection_name

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        embeddings = self._embedder.embed([c.content for c in chunks])
        self._collection.add(
            ids=[c.id for c in chunks],
            documents=[c.content for c in chunks],
            embeddings=embeddings,
            metadatas=[self._serialize_metadata(c) for c in chunks],
        )

    def query(
        self,
        query_text: str,
        top_k: int = 3,
        where: dict | None = None,
    ) -> list[RetrievedChunk]:
        query_embedding = self._embedder.embed([query_text])[0]
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        documents = (result.get("documents") or [[]])[0] or []
        metadatas = (result.get("metadatas") or [[]])[0] or []
        distances = (result.get("distances") or [[]])[0] or []
        ids = (result.get("ids") or [[]])[0] or []

        retrieved: list[RetrievedChunk] = []
        for doc, meta, dist, cid in zip(documents, metadatas, distances, ids, strict=False):
            meta_dict: dict = dict(meta or {})
            raw_index = meta_dict.pop("chunk_index", 0)
            try:
                chunk_index = int(raw_index)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                chunk_index = 0
            chunk = Chunk(
                id=str(cid),
                content=str(doc),
                document_id=str(meta_dict.pop("document_id", cid)),
                chunk_index=chunk_index,
                metadata=meta_dict,
            )
            retrieved.append(RetrievedChunk(chunk=chunk, distance=float(dist)))
        return retrieved

    def count(self) -> int:
        return self._collection.count()

    @staticmethod
    def _serialize_metadata(chunk: Chunk) -> dict:
        """Aplana la metadata a tipos primitivos que Chroma acepta."""
        meta = {
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
        }
        for key, value in chunk.metadata.items():
            if isinstance(value, str | int | float | bool):
                meta[key] = value
            else:
                meta[key] = str(value)
        return meta
