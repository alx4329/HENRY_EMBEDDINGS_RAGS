"""Ejercicio 04 — Router RAG (multi-store).

Este ejercicio combina los dos pipelines (cómics y música) detrás de un
clasificador previo que decide a qué store enviar cada pregunta.
Demuestra los principios:
- **Open/Closed**: añadir un nuevo dominio sólo requiere registrar un
  ``RAGBundle`` adicional; el router no se modifica.
- **Single Responsibility**: la decisión de ruteo está aislada en una clase.

Uso:
    uv run python scripts/exercise_04_router_rag.py
"""

from __future__ import annotations

from dataclasses import dataclass

import _bootstrap  # noqa: F401
from clase3.adapters.openai_llm import OpenAIChatClient
from clase3.config import COMICS_DIR, DEFAULT_CHAT_MODEL, MUSIC_DIR, OPENAI_API_KEY
from clase3.domain.messages import SystemMessage, UserMessage
from clase3.factory import RAGBundle, build_rag_bundle


@dataclass
class RouterRAG:
    bundles: dict[str, RAGBundle]
    classifier: OpenAIChatClient

    def route(self, question: str) -> str:
        keys = ", ".join(sorted(self.bundles.keys()))
        messages = [
            SystemMessage(
                content=(
                    "Eres un clasificador de preguntas. Tu tarea es elegir uno "
                    f"de los siguientes dominios: {keys}. Responde con UNA sola "
                    "palabra: el nombre del dominio."
                )
            ),
            UserMessage(content=question),
        ]
        decision = self.classifier.chat(messages).content.strip().lower()
        for key in self.bundles:
            if key in decision:
                return key
        # Fallback: primer dominio registrado
        return next(iter(self.bundles))

    def ask(self, question: str):
        domain = self.route(question)
        bundle = self.bundles[domain]
        answer = bundle.pipeline.ask(question)
        return domain, answer


QUESTIONS: list[str] = [
    "¿Qué simboliza la máscara de Guy Fawkes en V de Vendetta?",
    "¿Quién produjo el álbum 'Supernatural' de Santana y por qué fue importante?",
    "¿Cómo termina 'Maus' de Art Spiegelman?",
    "¿Qué denuncia la canción 'V Centenario' de Los Fabulosos Cadillacs?",
]


def main() -> None:
    print("═" * 70)
    print("ROUTER RAG · Cómics + Música, decidido por LLM clasificador")
    print("═" * 70)

    bundles = {
        "comics": build_rag_bundle(
            collection_name="comics",
            data_directory=COMICS_DIR,
            extra_metadata={"corpus": "comics"},
        ),
        "musica": build_rag_bundle(
            collection_name="musica",
            data_directory=MUSIC_DIR,
            extra_metadata={"corpus": "musica"},
        ),
    }
    for name, bundle in bundles.items():
        if bundle.store.count() == 0:
            print(f"⚠ Colección '{name}' vacía; ingestando…")
            bundle.indexer.index()

    classifier = OpenAIChatClient(model=DEFAULT_CHAT_MODEL, temperature=0.0, api_key=OPENAI_API_KEY)
    router = RouterRAG(bundles=bundles, classifier=classifier)

    for question in QUESTIONS:
        domain, result = router.ask(question)
        print("─" * 70)
        print(f"❓ {question}")
        print(f"🎯 Dominio detectado: {domain}")
        print("\n💬 " + result.answer)
        sources = ", ".join(
            item.chunk.metadata.get("title", item.chunk.document_id) for item in result.retrieved
        )
        print(f"📚 Fuentes: {sources}\n")


if __name__ == "__main__":
    main()
