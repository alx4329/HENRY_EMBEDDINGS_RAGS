"""Port: proveedor de embeddings.

Abstrae el cómputo de vectores densos para textos. Permite cambiar de OpenAI
a sentence-transformers, Cohere o cualquier otro proveedor sin tocar el
pipeline.
"""

from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    """Contrato para proveedores de embeddings."""

    @property
    def model_name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Devuelve un vector por texto, en el mismo orden."""
        ...
