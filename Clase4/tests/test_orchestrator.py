"""Tests del OrchestratorWorkflow."""

from __future__ import annotations

import json
import threading
import time

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


def test_orchestrator_runs_workers_in_parallel() -> None:
    """Los workers deben ejecutarse en paralelo, no secuencialmente.

    Cada worker duerme 100 ms. Con N=4 workers secuenciales tomaría >400 ms;
    en paralelo debe acabar bastante por debajo de 400 ms.
    """
    plan_json = json.dumps(
        {
            "analysis": "Plan paralelo.",
            "tasks": [
                {"type": "datos", "description": "d1"},
                {"type": "contexto", "description": "d2"},
                {"type": "musical", "description": "d3"},
                {"type": "legado", "description": "d4"},
            ],
        }
    )

    lock = threading.Lock()
    worker_calls = {"n": 0}

    def responder(messages):
        system = messages[0].content.lower()
        if "editor jefe" in system:
            return plan_json
        if "editor de cierre" in system:
            return "Ensayo."
        with lock:
            worker_calls["n"] += 1
        time.sleep(0.1)
        return "Aporte."

    workflow = OrchestratorWorkflow(llm=ScriptedLLM(responder=responder), max_workers=4)
    start = time.perf_counter()
    result = workflow.run("Tema paralelo")
    elapsed = time.perf_counter() - start

    assert worker_calls["n"] == 4
    assert result.metadata["sub_tasks"] == ["datos", "contexto", "musical", "legado"]
    assert elapsed < 0.35, f"workers no corrieron en paralelo (elapsed={elapsed:.2f}s)"


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
