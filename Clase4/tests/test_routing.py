"""Tests del RoutingWorkflow."""

from __future__ import annotations

from clase4.workflows import RoutingWorkflow
from tests._fake_llm import ScriptedLLM


def test_routing_selects_specialist_by_keyword() -> None:
    def responder(messages):
        content = messages[-1].content.lower()
        if "watchmen" in content or "rorschach" in content:
            # Caso 1: el router debe responder con el nombre del especialista
            if "router" in messages[0].content.lower():
                return "comics_expert"
            # Caso 2: el especialista responde
            return "Watchmen es una obra cumbre de Alan Moore."
        if "celia" in content or "salsa" in content:
            if "router" in messages[0].content.lower():
                return "salsa_expert"
            return "Quimbara reinventó la salsa de los 70."
        return "comics_expert"

    llm = ScriptedLLM(responder=responder)
    workflow = RoutingWorkflow(llm=llm)

    r1 = workflow.run("¿Qué es Watchmen?")
    assert r1.metadata["chosen_specialist"] == "comics_expert"
    assert "Watchmen" in r1.output


def test_routing_fallback_to_first_specialist_when_unknown() -> None:
    llm = ScriptedLLM(responder=lambda _msgs: "respuesta_invalida")
    workflow = RoutingWorkflow(llm=llm)
    result = workflow.run("pregunta cualquiera")
    # El primer especialista por defecto es 'comics_expert'
    assert result.metadata["chosen_specialist"] == "comics_expert"
