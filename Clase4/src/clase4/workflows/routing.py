"""Workflow 2 — Routing.

Patrón: un agente clasificador decide qué especialista debe responder.

Caso: tres especialistas — un experto en cómics, un especialista en salsa
y un experto en rock latino — se exponen al usuario detrás de un único
"router" que decide a quién dirigir cada pregunta.

Esto demuestra:
- **Open/Closed**: añadir un especialista nuevo (jazz, hip hop, etc.) sólo
  requiere implementar otra clase ``SpecialistAgent`` y registrarla.
- **Dependency Inversion**: el router depende del Protocol ``SpecialistAgent``,
  no de implementaciones concretas.
"""

from __future__ import annotations

from typing import Protocol

from clase4.domain.messages import SystemMessage, UserMessage
from clase4.domain.workflow import WorkflowResult, WorkflowStep
from clase4.ports.llm import LLMClient


class SpecialistAgent(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    def answer(self, question: str) -> str: ...


class _LLMSpecialist:
    """Especialista genérico parametrizado por system prompt."""

    def __init__(
        self,
        name: str,
        description: str,
        system_prompt: str,
        llm: LLMClient,
    ) -> None:
        self._name = name
        self._description = description
        self._system_prompt = system_prompt
        self._llm = llm

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def answer(self, question: str) -> str:
        messages = [
            SystemMessage(content=self._system_prompt),
            UserMessage(content=question),
        ]
        return self._llm.chat(messages, temperature=0.4).content


def build_default_specialists(llm: LLMClient) -> list[SpecialistAgent]:
    return [
        _LLMSpecialist(
            name="comics_expert",
            description=(
                "Experto en cómics canónicos del siglo XX (Watchmen, V de "
                "Vendetta, Maus, Persépolis, Sandman, The Dark Knight Returns). "
                "Usar para preguntas sobre novelas gráficas y sus autores."
            ),
            system_prompt=(
                "Eres un crítico de cómic. Respondes con rigor académico, "
                "citando autores y años de publicación. Si la pregunta no es "
                "sobre cómics, recházala."
            ),
            llm=llm,
        ),
        _LLMSpecialist(
            name="salsa_expert",
            description=(
                "Experto en salsa y son cubano. Especialidad: Celia Cruz, Tito "
                "Puente, Johnny Pacheco y la Fania All-Stars. Usar para "
                "preguntas sobre la diáspora afrocaribeña musical."
            ),
            system_prompt=(
                "Eres un musicólogo especializado en salsa y música caribeña. "
                "Hablas con autoridad sobre el género, autores y arreglos."
            ),
            llm=llm,
        ),
        _LLMSpecialist(
            name="latin_rock_expert",
            description=(
                "Experto en rock latinoamericano: Santana, Los Fabulosos "
                "Cadillacs, Soda Stereo, Caifanes. Usar para preguntas sobre "
                "rock fusión y ska latino."
            ),
            system_prompt=(
                "Eres un crítico de rock latinoamericano. Conoces el latin "
                "rock, el ska argentino y el rock fusión. Sé puntual con "
                "fechas, álbumes y productores."
            ),
            llm=llm,
        ),
    ]


class RoutingWorkflow:
    name = "routing"

    def __init__(self, llm: LLMClient, specialists: list[SpecialistAgent] | None = None) -> None:
        self._llm = llm
        self._specialists = specialists or build_default_specialists(llm)
        self._index = {s.name: s for s in self._specialists}

    def _route(self, question: str) -> str:
        catalog = "\n".join(f"- {s.name}: {s.description}" for s in self._specialists)
        messages = [
            SystemMessage(
                content=(
                    "Eres un router de preguntas. Tienes los siguientes "
                    f"especialistas disponibles:\n{catalog}\n\n"
                    "Devuelve UNA sola palabra: el nombre exacto del "
                    "especialista que debe responder."
                )
            ),
            UserMessage(content=question),
        ]
        decision = self._llm.chat(messages, temperature=0.0).content.strip().lower()
        for key in self._index:
            if key in decision:
                return key
        return next(iter(self._index))  # fallback

    def run(self, user_input: str) -> WorkflowResult:
        chosen = self._route(user_input)
        answer = self._index[chosen].answer(user_input)
        return WorkflowResult(
            input=user_input,
            output=answer,
            steps=[
                WorkflowStep(name="router", output=chosen),
                WorkflowStep(name=chosen, output=answer),
            ],
            metadata={"chosen_specialist": chosen},
        )
