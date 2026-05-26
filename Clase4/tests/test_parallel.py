"""Tests del ParallelInsightsWorkflow."""

from __future__ import annotations

import threading

from clase4.workflows import ParallelInsightsWorkflow
from tests._fake_llm import ScriptedLLM


def test_parallel_runs_all_specialists_and_synthesizes() -> None:
    counter = {"n": 0}
    lock = threading.Lock()

    def responder(messages):
        with lock:
            counter["n"] += 1
        if "editor cultural" in messages[0].content.lower():
            return "Síntesis final unificada."
        return "Análisis del especialista."

    llm = ScriptedLLM(responder=responder)
    workflow = ParallelInsightsWorkflow(llm=llm)
    result = workflow.run("Tema X")

    # 3 especialistas + 1 síntesis
    assert counter["n"] == 4
    assert result.output == "Síntesis final unificada."
    assert len(result.metadata["specialists"]) == 3


def test_parallel_steps_include_summary() -> None:
    llm = ScriptedLLM(responder=lambda msgs: "summary"
                       if "editor cultural" in msgs[0].content.lower()
                       else "partial")
    workflow = ParallelInsightsWorkflow(llm=llm)
    result = workflow.run("Tema Y")
    step_names = [s.name for s in result.steps]
    assert "summary" in step_names
