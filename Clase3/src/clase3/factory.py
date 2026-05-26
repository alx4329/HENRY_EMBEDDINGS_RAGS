"""Composition root: ensambla los componentes del pipeline RAG.

Es el ÚNICO lugar donde se conocen los Adapters concretos. El resto del
código depende sólo de Ports.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from clase3.adapters.chroma_store import ChromaVectorStore
from clase3.adapters.markdown_loader import MarkdownDirectoryLoader
from clase3.adapters.openai_embedder import OpenAIEmbedder
from clase3.adapters.openai_llm import OpenAIChatClient
from clase3.config import (
    CHROMA_PERSIST_DIR,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    OPENAI_API_KEY,
)
from clase3.ports.vector_store import VectorStore
from clase3.services.augmenter import DEFAULT_SYSTEM_PROMPT, PromptAugmenter
from clase3.services.chunker import TextChunker
from clase3.services.generator import AnswerGenerator
from clase3.services.indexer import CorpusIndexer
from clase3.services.rag_pipeline import RAGPipeline
from clase3.services.retriever import Retriever


@dataclass
class RAGBundle:
    """Conjunto cohesivo de componentes para un dominio (cómics, música…).

    El campo ``store`` se tipa contra el Protocol ``VectorStore`` y no contra
    la implementación concreta de Chroma: esto es Dependency Inversion en
    acción. Cualquier código que reciba un ``RAGBundle`` puede llamar a
    ``store.count()`` / ``store.query(...)`` sin acoplarse al proveedor.
    """

    name: str
    pipeline: RAGPipeline
    indexer: CorpusIndexer
    store: VectorStore


def build_rag_bundle(
    *,
    collection_name: str,
    data_directory: Path,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    chat_model: str = DEFAULT_CHAT_MODEL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    top_k: int = DEFAULT_TOP_K,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    persist_directory: Path | None = CHROMA_PERSIST_DIR,
    reset_store: bool = False,
    extra_metadata: dict | None = None,
) -> RAGBundle:
    """Construye un pipeline RAG completo a partir de un directorio de .md."""
    embedder = OpenAIEmbedder(model=embedding_model, api_key=OPENAI_API_KEY)
    store = ChromaVectorStore(
        collection_name=collection_name,
        embedder=embedder,
        persist_directory=persist_directory,
        reset=reset_store,
    )
    loader = MarkdownDirectoryLoader(
        directory=data_directory,
        extra_metadata=extra_metadata,
    )
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    indexer = CorpusIndexer(loader=loader, chunker=chunker, store=store)

    llm = OpenAIChatClient(model=chat_model, temperature=temperature, api_key=OPENAI_API_KEY)
    retriever = Retriever(store=store, top_k=top_k)
    augmenter = PromptAugmenter(system_prompt=system_prompt)
    generator = AnswerGenerator(llm=llm)

    pipeline = RAGPipeline(retriever=retriever, augmenter=augmenter, generator=generator)
    return RAGBundle(name=collection_name, pipeline=pipeline, indexer=indexer, store=store)
