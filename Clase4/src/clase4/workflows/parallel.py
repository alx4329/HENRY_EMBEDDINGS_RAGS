"""Workflow 3 — Parallelization (Map-Reduce).

Patrón: N especialistas trabajan en paralelo sobre la misma pregunta
(fan-out), y un agregador sintetiza sus aportes (fan-in).

Caso: dado un tópico musical/cultural latinoamericano, lanzamos tres
analistas (musical, histórico, social) en hilos paralelos. Un cuarto agente
los une en una respuesta cohesiva.

Esto demuestra:
- **Single Responsibility**: cada especialista habla sólo desde su ángulo.
- **Liskov**: el agregador puede recibir N especialistas; no le importa
  quiénes sean siempre que cumplan el Protocol.
- **Eficiencia**: ``ThreadPoolExecutor`` libera el GIL durante I/O de la
  API, ejecutando las 3 llamadas en paralelo.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Protocol

from clase4.domain.messages import SystemMessage, UserMessage
from clase4.domain.workflow import WorkflowResult, WorkflowStep
from clase4.ports.llm import LLMClient


class Specialist(Protocol):
    @property
    def name(self) -> str: ...

    def analyze(self, topic: str) -> str: ...


@dataclass
class _LLMSpecialist:
    name: str
    system_prompt: str
    llm: LLMClient

    def analyze(self, topic: str) -> str:
        messages = [
            SystemMessage(content=self.system_prompt),
            UserMessage(content=f"Analiza desde tu ángulo el siguiente tema: {topic}"),
        ]
        return self.llm.chat(messages, temperature=0.5).content


def build_default_specialists(llm: LLMClient) -> list[Specialist]:
    return [
        _LLMSpecialist(
            name="musical",
            system_prompt=(
                "Eres un musicólogo. Analiza el tema desde la perspectiva "
                "estrictamente musical: género, ritmos, instrumentación, "
                "tonalidad, influencias sonoras. Máx 120 palabras."
            ),
            llm=llm,
        ),
        _LLMSpecialist(
            name="historico",
            system_prompt=(
                "Eres un historiador cultural. Analiza el tema en su contexto "
                "histórico: década, hechos políticos contemporáneos, "
                "movimientos sociales que lo enmarcan. Máx 120 palabras."
            ),
            llm=llm,
        ),
        _LLMSpecialist(
            name="social",
            system_prompt=(
                "Eres un sociólogo de la cultura popular latinoamericana. "
                "Analiza el tema en clave social: clases sociales, género, "
                "diáspora, identidad afrolatina, herencia colonial. Máx 120 "
                "palabras."
            ),
            llm=llm,
        ),
    ]


class ParallelInsightsWorkflow:
    name = "parallel"

    def __init__(
        self,
        llm: LLMClient,
        specialists: list[Specialist] | None = None,
        *,
        max_workers: int = 4,
    ) -> None:
        self._llm = llm
        self._specialists = specialists or build_default_specialists(llm)
        self._max_workers = max_workers

    def _summarize(self, topic: str, partials: dict[str, str]) -> str:
        body = "\n\n".join(f"### Análisis {name}\n{text}" for name, text in partials.items())
        messages = [
            SystemMessage(
                content=(
                    "Eres un editor cultural. Recibes 3 análisis sobre el "
                    "mismo tema desde ángulos distintos. Sintetízalos en una "
                    "respuesta única, fluida y bien estructurada (250-300 "
                    "palabras), sin repetir información."
                )
            ),
            UserMessage(content=f"Tema: {topic}\n\nAportes:\n\n{body}"),
        ]
        return self._llm.chat(messages, temperature=0.4).content

    def run(self, user_input: str) -> WorkflowResult:
        partials: dict[str, str] = {}
        steps: list[WorkflowStep] = []

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_map = {
                executor.submit(specialist.analyze, user_input): specialist.name
                for specialist in self._specialists
            }
            for future in as_completed(future_map):
                name = future_map[future]
                partials[name] = future.result()
                steps.append(WorkflowStep(name=name, output=partials[name]))

        summary = self._summarize(user_input, partials)
        steps.append(WorkflowStep(name="summary", output=summary))

        return WorkflowResult(
            input=user_input,
            output=summary,
            steps=steps,
            metadata={"specialists": list(partials)},
        )
