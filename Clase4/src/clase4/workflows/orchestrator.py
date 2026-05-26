"""Workflow 5 — Orchestrator + Workers.

Patrón: un orquestador descompone una solicitud compleja en sub-tareas
tipadas, despacha cada sub-tarea a un worker especializado y agrega los
resultados.

Caso: dado un álbum o canción, el orquestador descompone la solicitud en
sub-tareas (datos básicos, contexto histórico, análisis musical, recepción
crítica). Cada worker resuelve su sub-tarea. Un agente final sintetiza
todo en un *long-form essay*.

Esto demuestra:
- **Decomposition**: el orquestador convierte una solicitud difusa en
  trabajo paralelizable.
- **Open/Closed**: añadir un nuevo tipo de worker (ej. ``analisis_lirico``)
  no requiere tocar al orquestador.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import ClassVar

from clase4.domain.messages import SystemMessage, UserMessage
from clase4.domain.workflow import WorkflowResult, WorkflowStep
from clase4.ports.llm import LLMClient

ORCHESTRATOR_PROMPT = (
    "Eres un editor jefe de revista cultural. Recibes el nombre de una "
    "canción o álbum latinoamericano y descompones la cobertura editorial "
    "en 3-5 sub-tareas. Devuelve ESTRICTAMENTE un JSON con la forma:\n"
    "{\n"
    '  "analysis": "<2 frases sobre el enfoque general>",\n'
    '  "tasks": [\n'
    '     {"type": "<datos|contexto|musical|recepcion|legado>", '
    '"description": "<descripción de la sub-tarea>"}\n'
    "  ]\n"
    "}\n"
    "No agregues texto fuera del JSON."
)

SYNTHESIZER_PROMPT = (
    "Eres un editor de cierre. Recibes el tópico original y las respuestas "
    "de 3-5 colaboradores. Sintetiza todo en un ensayo de 350-450 palabras, "
    "fluido, en español neutro, con secciones implícitas y citas a los "
    "datos. No menciones que son colaboradores."
)


@dataclass(frozen=True)
class SubTask:
    type: str
    description: str


class WorkerAgent:
    """Worker genérico parametrizado por tipo de sub-tarea."""

    PROMPTS: ClassVar[dict[str, str]] = {
        "datos": (
            "Eres un fact-checker musical. Devuelves bullets con datos verificables: "
            "artista, álbum, año, sello, compositor, género."
        ),
        "contexto": (
            "Eres un historiador cultural. Explica el contexto histórico en el que "
            "se creó la obra. Máx 130 palabras."
        ),
        "musical": (
            "Eres un musicólogo. Describe el análisis musical: tonalidad, ritmo, "
            "instrumentación, momentos clave. Máx 130 palabras."
        ),
        "recepcion": (
            "Eres un crítico musical. Describe la recepción crítica y comercial: "
            "premios, ventas, repercusión en medios. Máx 130 palabras."
        ),
        "legado": (
            "Eres un historiador de la música popular. Explica el legado e "
            "influencia de la obra en artistas posteriores y la cultura. Máx 130 "
            "palabras."
        ),
    }
    DEFAULT_PROMPT = (
        "Eres un asistente cultural. Responde la sub-tarea con datos concretos en "
        "máximo 130 palabras."
    )

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def run(self, topic: str, task: SubTask) -> str:
        system = self.PROMPTS.get(task.type.lower(), self.DEFAULT_PROMPT)
        messages = [
            SystemMessage(content=system),
            UserMessage(content=f"Tema central: {topic}\nSub-tarea: {task.description}"),
        ]
        return self._llm.chat(messages, temperature=0.4).content


class OrchestratorWorkflow:
    name = "orchestrator"

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self._worker = WorkerAgent(llm)

    def _decompose(self, topic: str) -> tuple[str, list[SubTask]]:
        messages = [
            SystemMessage(content=ORCHESTRATOR_PROMPT),
            UserMessage(content=topic),
        ]
        raw = self._llm.chat(messages, temperature=0.2).content
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError(f"El orquestador no devolvió JSON válido:\n{raw}")
        payload = json.loads(match.group(0))
        tasks = [
            SubTask(type=str(t.get("type", "datos")), description=str(t.get("description", "")))
            for t in payload.get("tasks", [])
        ]
        return payload.get("analysis", ""), tasks

    def _synthesize(self, topic: str, partials: list[tuple[SubTask, str]]) -> str:
        body = "\n\n".join(
            f"### Sub-tarea: {task.type}\n**Descripción:** {task.description}\n\n{output}"
            for task, output in partials
        )
        messages = [
            SystemMessage(content=SYNTHESIZER_PROMPT),
            UserMessage(content=f"Tema: {topic}\n\nAportes:\n\n{body}"),
        ]
        return self._llm.chat(messages, temperature=0.5).content

    def run(self, user_input: str) -> WorkflowResult:
        analysis, tasks = self._decompose(user_input)
        steps: list[WorkflowStep] = [
            WorkflowStep(name="orchestrator", output={"analysis": analysis, "tasks": tasks}),
        ]

        partials: list[tuple[SubTask, str]] = []
        for task in tasks:
            output = self._worker.run(user_input, task)
            partials.append((task, output))
            steps.append(WorkflowStep(name=f"worker_{task.type}", output=output))

        final = self._synthesize(user_input, partials)
        steps.append(WorkflowStep(name="synthesizer", output=final))

        return WorkflowResult(
            input=user_input,
            output=final,
            steps=steps,
            metadata={"analysis": analysis, "sub_tasks": [t.type for t in tasks]},
        )
