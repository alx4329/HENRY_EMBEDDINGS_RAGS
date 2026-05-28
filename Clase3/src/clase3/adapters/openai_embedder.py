"""Adapter: proveedor de embeddings sobre la API de OpenAI."""

from __future__ import annotations

from openai import OpenAI

# Dimensiones conocidas de los modelos de embeddings de OpenAI.
_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072
}


class OpenAIEmbedder:
    """Implementación de ``EmbeddingProvider`` usando la API de OpenAI."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        api_key: str | None = None,
        batch_size: int = 64,
    ) -> None:
        self._model = model
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self._batch_size = batch_size

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return _DIMENSIONS.get(self._model, 1536)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            response = self._client.embeddings.create(model=self._model, input=batch)
            embeddings.extend(item.embedding for item in response.data)
        return embeddings
