"""Tests del ChainingWorkflow usando un FakeLLM."""

from __future__ import annotations

from clase4.workflows import ChainingWorkflow
from tests._fake_llm import ScriptedLLM


def test_chaining_runs_both_steps() -> None:
    counter = {"n": 0}

    def responder(_msgs):
        counter["n"] += 1
        return f"output_{counter['n']}"

    llm = ScriptedLLM(responder=responder)
    workflow = ChainingWorkflow(llm=llm)
    result = workflow.run("Quimbara — Celia Cruz")

    assert counter["n"] == 2  # investigador + redactor
    assert result.steps[0].name == "researcher"
    assert result.steps[1].name == "writer"
    assert result.output == "output_2"


def test_chaining_passes_research_to_writer() -> None:
    seen: dict[str, str] = {}

    def responder(messages):
        last = messages[-1].content
        if "Investiga" in last:
            seen["research_input"] = last
            return "## DATOS\n- Año: 2001"
        seen["writer_input"] = last
        return "Reseña final."

    llm = ScriptedLLM(responder=responder)
    workflow = ChainingWorkflow(llm=llm)
    workflow.run("La Negra Tiene Tumbao — Celia Cruz")

    assert "## DATOS" in seen["writer_input"]
    assert "Año: 2001" in seen["writer_input"]
