"""Port: cargador de documentos desde una fuente arbitraria.

Cualquier adapter (markdown, pdf, csv, web, etc.) que cumpla este protocolo
puede ser inyectado en el pipeline RAG sin modificar el resto del código
(Open/Closed Principle).
"""

from __future__ import annotations

from typing import Protocol

from clase3.domain.document import Document


class DocumentLoader(Protocol):
    """Contrato para clases que cargan documentos."""

    def load(self) -> list[Document]:
        """Devuelve la colección completa de documentos cargados."""
        ...
