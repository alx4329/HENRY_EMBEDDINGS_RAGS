"""Ejercicio 03 — RAG musical sobre Santana, Celia Cruz y Fabulosos Cadillacs.

Demuestra el reuso del mismo pipeline con un prompt distinto: el sistema actúa
como crítico musical latinoamericano.

Uso:
    uv run python scripts/exercise_03_musical_rag.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
from clase3.config import MUSIC_DIR
from clase3.factory import build_rag_bundle

MUSIC_SYSTEM_PROMPT = (
    "Eres un crítico musical latinoamericano. Respondes preguntas sobre "
    "canciones, artistas y contextos históricos usando exclusivamente los "
    "fragmentos provistos. Cita el título de la canción y el artista entre "
    "paréntesis. Si la información no está en el contexto, di que no lo sabes."
)

MUSIC_QUESTIONS: list[str] = [
    "¿Qué relación hay entre 'Oye Cómo Va' de Santana y Tito Puente?",
    "¿En qué año se grabó 'La Vida Es un Carnaval' y de qué manera dialoga con la biografía de Celia Cruz?",
    "¿Por qué la canción 'Matador' de Los Fabulosos Cadillacs tuvo un resurgimiento global en 2017?",
    "¿Cómo describe el dataset el solo de trompeta de 'Matador' y quién lo grabó?",
]


def main() -> None:
    print("═" * 70)
    print("RAG MUSICAL · Santana · Celia Cruz · Fabulosos Cadillacs")
    print("═" * 70)

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

    for question in MUSIC_QUESTIONS:
        result = bundle.pipeline.ask(question)
        print("─" * 70)
        print(f"❓ {question}")
        print("\n💬 " + result.answer)
        print("\n📚 Fuentes:")
        for idx, item in enumerate(result.retrieved, start=1):
            title = item.chunk.metadata.get("title", item.chunk.document_id)
            print(f"   {idx}. {title}  (sim≈{item.similarity:.2f})")
        print()


if __name__ == "__main__":
    main()
