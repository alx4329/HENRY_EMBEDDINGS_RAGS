"""Entidades de dominio: estructuras de datos sin dependencias externas."""

from clase3.domain.document import Chunk, Document, RetrievedChunk
from clase3.domain.messages import AIMessage, BaseMessage, SystemMessage, UserMessage

__all__ = [
    "AIMessage",
    "BaseMessage",
    "Chunk",
    "Document",
    "RetrievedChunk",
    "SystemMessage",
    "UserMessage",
]
