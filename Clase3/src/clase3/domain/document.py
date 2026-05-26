"""Entidades de dominio para documentos y chunks.

Estas clases son DTOs puros: no conocen ningún detalle de OpenAI, ChromaDB o
disco. Permiten que el resto del pipeline trabaje con tipos estables sin
acoplarse a un proveedor concreto (Principio de Inversión de Dependencias).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    """Documento crudo cargado desde una fuente.

    Atributos:
        id: identificador único (UUID por defecto).
        content: texto plano del documento.
        metadata: diccionario libre de metadatos (fuente, género, etc.).
    """

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass(frozen=True)
class Chunk:
    """Fragmento de un documento, listo para ser embebido y almacenado.

    Cada chunk arrastra el id de su documento padre y su posición relativa.
    """

    content: str
    document_id: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass(frozen=True)
class RetrievedChunk:
    """Chunk recuperado por el vector store con su puntaje de similitud."""

    chunk: Chunk
    distance: float  # menor distancia ⇒ mayor similitud (cosine distance)

    @property
    def similarity(self) -> float:
        """Convierte la distancia coseno a similitud (heurística simple)."""
        return max(0.0, 1.0 - self.distance)
