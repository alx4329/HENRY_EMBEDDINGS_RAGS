"""Ejercicio 03 — RAG musical sobre Santana, Celia Cruz y Fabulosos Cadillacs.

Demuestra el reuso del mismo pipeline con un prompt distinto: el sistema actúa
como crítico musical latinoamericano.

Para cada pregunta imprime:
- la respuesta del LLM,
- una **tabla de retrieval** con los chunks recuperados, su similitud y la
  sección Markdown que aportó cada uno (útil para auditar qué fragmento del
  corpus alimentó la respuesta).

Uso:
    uv run python scripts/exercise_03_musical_rag.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
from clase3.config import MUSIC_DIR
from clase3.factory import build_rag_bundle
from clase3.services.rag_pipeline import RAGAnswer

MUSIC_SYSTEM_PROMPT = (
    "Eres un crítico musical latinoamericano. Respondes preguntas sobre "
    "canciones, artistas y contextos históricos usando exclusivamente los "
    "fragmentos provistos. Cita el título de la canción y el artista entre "
    "paréntesis. Si la información no está en el contexto, di que no lo sabes."
)

# Preguntas elegidas para cubrir tres tipos de retrieval distintos:
#   1-2: datos puntuales (un hecho que vive en una sola sección Markdown).
#   3-4: contexto histórico/cultural (la respuesta se sintetiza desde 2+ secciones).
#   5  : pregunta que cruza artistas (testea que el corpus completo dialoga).
MUSIC_QUESTIONS: list[str] = [
    "¿Qué relación hay entre 'Oye Cómo Va' de Santana y Tito Puente?",
    "¿En qué año se grabó 'La Vida Es un Carnaval' y de qué manera dialoga con la biografía de Celia Cruz?",
    "¿Por qué la canción 'Matador' de Los Fabulosos Cadillacs tuvo un resurgimiento global en 2017?",
    "¿Cómo describe el dataset el solo de trompeta de 'Matador' y quién lo grabó?",
    "¿Qué tienen en común 'Quimbara' y 'V Centenario' en su relación con la diáspora latinoamericana?",
]


def _print_retrieval_table(result: RAGAnswer) -> None:
    """Imprime una tabla compacta con título · sección · similitud por chunk."""
    print("\n📚 Retrieval (top-k):")
    print(f"   {'#':>2}  {'sim':>5}  {'canción':<32}  sección")
    print(f"   {'─' * 2}  {'─' * 5}  {'─' * 32}  {'─' * 25}")
    for idx, item in enumerate(result.retrieved, start=1):
        title = item.chunk.metadata.get("title", item.chunk.document_id)[:32]
        section = item.chunk.metadata.get("section", "(preámbulo)")[:25]
        print(f"   {idx:>2}  {item.similarity:>5.2f}  {title:<32}  {section}")


def main() -> None:
    print("═" * 78)
    print("RAG MUSICAL · Santana · Celia Cruz · Fabulosos Cadillacs")
    print("═" * 78)

    bundle = build_rag_bundle(
        collection_name="musica",
        data_directory=MUSIC_DIR,
        system_prompt=MUSIC_SYSTEM_PROMPT,
        extra_metadata={"corpus": "musica"},
        top_k=4,
    )
    if bundle.store.count() == 0:
        print("⚠ La colección 'musica' está vacía; ejecutando ingesta automática.")
        bundle.indexer.index()
        print(f"   indexados {bundle.store.count()} chunks.\n")

    for question in MUSIC_QUESTIONS:
        result = bundle.pipeline.ask(question)
        print("─" * 78)
        print(f"❓ {question}")
        print(f"\n💬 {result.answer}")
        _print_retrieval_table(result)
        print()


if __name__ == "__main__":
    main()
