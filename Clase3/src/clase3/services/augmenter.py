"""Servicio: construcción del prompt aumentado con el contexto recuperado.

Mantiene la plantilla del system prompt parametrizable, de modo que cada
ejercicio pueda inyectar instrucciones distintas (analista de cómics,
crítico musical, etc.) sin tocar el resto del pipeline.
"""

from __future__ import annotations

from clase3.domain.document import RetrievedChunk
from clase3.domain.messages import BaseMessage, SystemMessage, UserMessage

DEFAULT_SYSTEM_PROMPT = (
    "Eres un asistente experto. Respondes preguntas basándote en el contexto "
    "provisto. Reglas:\n"
    "1. Si la respuesta aparece directa o indirectamente en el contexto, "
    "respóndela con detalle citando el título del documento entre paréntesis.\n"
    "2. Sintetiza información de varios fragmentos cuando sea necesario.\n"
    "3. Sólo di 'no lo sé' cuando ningún fragmento ofrezca información "
    "relevante; nunca lo digas si los fragmentos son del tema correcto.\n"
    "4. No inventes datos que no estén en el contexto."
)


class PromptAugmenter:
    def __init__(self, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> None:
        self._system_prompt = system_prompt

    def build_messages(
        self,
        question: str,
        retrieved: list[RetrievedChunk],
    ) -> list[BaseMessage]:
        context = self._format_context(retrieved)
        user_content = (
            f"## Pregunta\n{question}\n\n"
            f"## Contexto recuperado\n{context}\n\n"
            "## Instrucciones\n"
            "- Responde sólo con la información del contexto.\n"
            "- Cita las fuentes entre paréntesis al final de cada afirmación clave.\n"
            "- Si la información es insuficiente, di explícitamente que no lo sabes."
        )
        return [
            SystemMessage(content=self._system_prompt),
            UserMessage(content=user_content),
        ]

    @staticmethod
    def _format_context(retrieved: list[RetrievedChunk]) -> str:
        if not retrieved:
            return "(sin contexto disponible)"
        blocks: list[str] = []
        for idx, item in enumerate(retrieved, start=1):
            title = item.chunk.metadata.get("title", item.chunk.document_id)
            source = item.chunk.metadata.get("source", "desconocida")
            blocks.append(
                f"[Fragmento {idx} · título: {title} · fuente: {source} · "
                f"similitud≈{item.similarity:.2f}]\n{item.chunk.content}"
            )
        return "\n\n".join(blocks)
