"""Ejercicio 02 — RAG básico sobre el corpus de cómics.

Hace tres preguntas sobre los cómics indexados y muestra:
1. La respuesta del LLM.
2. Los chunks recuperados con su puntaje de similitud.

Uso:
    uv run python scripts/exercise_02_basic_rag.py
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
from clase3.config import COMICS_DIR
from clase3.factory import build_rag_bundle
from clase3.services.rag_pipeline import RAGAnswer

COMICS_QUESTIONS: list[str] = [
    "¿Quién es Rorschach y cuál es su frase emblemática cuando lo arrestan?",
    "¿Por qué Marjane Satrapi tuvo que vivir entre Teherán y Viena en Persépolis?",
    "¿Qué decisión toma Ozymandias al final de Watchmen y por qué genera debate moral?",
]


def _print_answer(result: RAGAnswer) -> None:
    print("─" * 70)
    print(f"❓ Pregunta: {result.question}")
    print("\n💬 Respuesta del LLM:\n" + result.answer)
    print("\n📚 Fuentes recuperadas:")
    for idx, item in enumerate(result.retrieved, start=1):
        title = item.chunk.metadata.get("title", item.chunk.document_id)
        print(f"   {idx}. {title}  (sim≈{item.similarity:.2f}, idx={item.chunk.chunk_index})")
    print()


def main() -> None:
    print("═" * 70)
    print("RAG BÁSICO · Cómics")
    print("═" * 70)

    bundle = build_rag_bundle(
        collection_name="comics",
        data_directory=COMICS_DIR,
        extra_metadata={"corpus": "comics"},
    )
    if bundle.store.count() == 0:
        print("⚠ La colección 'comics' está vacía; ejecutando ingesta automática.")
        bundle.indexer.index()

    for question in COMICS_QUESTIONS:
        _print_answer(bundle.pipeline.ask(question))


if __name__ == "__main__":
    main()
