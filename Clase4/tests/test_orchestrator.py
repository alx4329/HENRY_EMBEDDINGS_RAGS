"""Tests del OrchestratorWorkflow."""

from __future__ import annotations

import json

from clase4.workflows import OrchestratorWorkflow
from tests._fake_llm import ScriptedLLM


def test_orchestrator_decomposes_and_synthesizes() -> None:
    plan_json = json.dumps(
        {
            "analysis": "Análisis de prueba.",
            "tasks": [
                {"type": "datos", "description": "Lista los datos básicos."},
                {"type": "contexto", "description": "Describe el contexto."},
            ],
        }
    )

    def responder(messages):
        system = messages[0].content.lower()
        if "editor jefe" in system:
            return plan_json
        if "editor de cierre" in system:
            return "Ensayo final."
        return "Aporte del worker."

    workflow = OrchestratorWorkflow(llm=ScriptedLLM(responder=responder))
    result = workflow.run("Supernatural — Santana")

    assert result.output == "Ensayo final."
    assert result.metadata["sub_tasks"] == ["datos", "contexto"]
    assert any(step.name == "worker_datos" for step in result.steps)
    assert any(step.name == "worker_contexto" for step in result.steps)
    assert result.steps[-1].name == "synthesizer"


def test_orchestrator_falls_back_when_no_json() -> None:
    """Si el orquestador no devuelve JSON parseable, usa tareas por defecto."""

    def responder(messages):
        system = messages[0].content.lower()
        if "editor jefe" in system:
            return "Lo siento, no puedo descomponer esa solicitud."
        if "editor de cierre" in system:
            return "Ensayo desde fallback."
        return "Aporte del worker."

    workflow = OrchestratorWorkflow(llm=ScriptedLLM(responder=responder))
    result = workflow.run("Tema desconocido")
    assert result.output == "Ensayo desde fallback."
    assert "datos" in result.metadata["sub_tasks"]
    assert "contexto" in result.metadata["sub_tasks"]


def test_orchestrator_handles_extra_text_around_json() -> None:
    plan_json = (
        "Aquí va el plan:\n"
        + json.dumps(
            {
                "analysis": "Breve análisis.",
                "tasks": [{"type": "datos", "description": "x"}],
            }
        )
        + "\n¡Listo!"
    )

    def responder(messages):
        if "editor jefe" in messages[0].content.lower():
            return plan_json
        if "editor de cierre" in messages[0].content.lower():
            return "Ensayo."
        return "Aporte."

    workflow = OrchestratorWorkflow(llm=ScriptedLLM(responder=responder))
    result = workflow.run("Tema Z")
    assert result.output == "Ensayo."
