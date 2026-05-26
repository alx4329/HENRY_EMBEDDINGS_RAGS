"""Ports (interfaces abstractas) — Principio de Inversión de Dependencias.

Cada módulo expone un Protocol/ABC que define el contrato que los adapters
concretos deben cumplir. El resto del sistema depende de estas abstracciones,
nunca de implementaciones específicas (OpenAI, ChromaDB, etc.).
"""

from clase3.ports.embedder import EmbeddingProvider
from clase3.ports.llm import LLMClient
from clase3.ports.loader import DocumentLoader
from clase3.ports.vector_store import VectorStore

__all__ = [
    "DocumentLoader",
    "EmbeddingProvider",
    "LLMClient",
    "VectorStore",
]
