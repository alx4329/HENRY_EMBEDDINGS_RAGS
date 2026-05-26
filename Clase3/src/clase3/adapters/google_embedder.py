"""Adapter: proveedor de embeddings sobre la API de Google (AI Studio).

Soporta los modelos accesibles vía API key (sin Vertex AI):

- ``gemini-embedding-001``        — 3072 dim, modelo flagship GA.
- ``gemini-embedding-2``          — 3072 dim, sucesor (la "v2" oficial).
- ``gemini-embedding-2-preview``  — 3072 dim, variante preview de la v2.

Notas:

- ``text-embedding-004`` ya **no está accesible** vía la API pública de AI
  Studio (deprecado / retirado del catálogo). El SDK devuelve 404.
- ``text-embedding-005`` sólo se expone vía **Vertex AI** con credenciales
  GCP (ADC o service account), no con API key plana.

Cumple el Protocol ``clase3.ports.embedder.EmbeddingProvider``, por lo que es
intercambiable con ``OpenAIEmbedder`` sin tocar el pipeline.

Requiere la variable de entorno ``GCP_API_KEY`` (o ``GOOGLE_API_KEY`` /
``GEMINI_API_KEY``). La forma más simple de obtener una llave válida es
crearla en https://aistudio.google.com/apikey (la API queda habilitada y sin
restricciones automáticamente).
"""

from __future__ import annotations

from google import genai
from google.genai import types

# Dimensiones nominales por modelo (la API también devuelve el tamaño real
# y eso siempre manda).
_DIMENSIONS: dict[str, int] = {
    "gemini-embedding-001": 3072,
    "gemini-embedding-2": 3072,
    "gemini-embedding-2-preview": 3072,
}

# Tamaño de batch máximo confiable por modelo.
#
# Observación empírica (probado a mano): los modelos Gemini de generación 2.x
# **no respetan el batching del endpoint embedContent** — si les mandas N
# textos en una sola llamada devuelven sólo 1 vector. Por seguridad usamos
# batch_size=1 para ellos. ``gemini-embedding-001`` sí batchea correctamente.
_MAX_BATCH_BY_MODEL: dict[str, int] = {
    "gemini-embedding-001": 100,
    "gemini-embedding-2": 1,
    "gemini-embedding-2-preview": 1,
}


class GoogleGenAIEmbedder:
    """Implementación de ``EmbeddingProvider`` usando el SDK ``google-genai``."""

    def __init__(
        self,
        model: str = "gemini-embedding-001",
        *,
        api_key: str | None = None,
        batch_size: int | None = None,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> None:
        if not api_key:
            raise ValueError(
                "GoogleGenAIEmbedder requiere una API key. Configura "
                "GCP_API_KEY (o GOOGLE_API_KEY / GEMINI_API_KEY) en tu .env."
            )
        self._model = model
        self._client = genai.Client(api_key=api_key)
        # Si el caller no fuerza un batch_size, usamos el seguro para ese
        # modelo (1 para Gemini 2.x, 100 para 001). Default conservador: 1.
        self._batch_size = batch_size or _MAX_BATCH_BY_MODEL.get(model, 1)
        self._task_type = task_type

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return _DIMENSIONS.get(self._model, 768)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            response = self._client.models.embed_content(
                model=self._model,
                contents=batch,
                config=types.EmbedContentConfig(task_type=self._task_type),
            )
            for item in response.embeddings:
                embeddings.append(list(item.values))
        return embeddings
