"""Tests del EvaluatorOptimizerWorkflow."""

from __future__ import annotations

from clase4.workflows import EvaluatorOptimizerWorkflow
from tests._fake_llm import ScriptedLLM


def test_approves_on_first_attempt() -> None:
    def responder(messages):
        if "editor de spoilers" in messages[0].content.lower():
            return "VEREDICTO: APROBADO"
        return "Una reseña sin spoilers."

    workflow = EvaluatorOptimizerWorkflow(llm=ScriptedLLM(responder=responder), max_retries=3)
    result = workflow.run("Watchmen")
    assert result.metadata["approved"] is True
    assert result.metadata["attempts"] == 1


def test_iterates_until_approved() -> None:
    state = {"writer_calls": 0, "evaluator_calls": 0}

    def responder(messages):
        role_is_evaluator = "editor de spoilers" in messages[0].content.lower()
        if role_is_evaluator:
            state["evaluator_calls"] += 1
            if state["evaluator_calls"] < 2:
                return "VEREDICTO: RECHAZADO\nFEEDBACK: revela el final."
            return "VEREDICTO: APROBADO"
        state["writer_calls"] += 1
        return f"Reseña intento {state['writer_calls']}"

    workflow = EvaluatorOptimizerWorkflow(llm=ScriptedLLM(responder=responder), max_retries=4)
    result = workflow.run("Maus")

    assert result.metadata["approved"] is True
    assert result.metadata["attempts"] == 2
    assert state["writer_calls"] == 2


def test_gives_up_after_max_retries() -> None:
    def responder(messages):
        if "editor de spoilers" in messages[0].content.lower():
            return "VEREDICTO: RECHAZADO\nFEEDBACK: spoiler persistente."
        return "Reseña con spoiler."

    workflow = EvaluatorOptimizerWorkflow(llm=ScriptedLLM(responder=responder), max_retries=2)
    result = workflow.run("Sandman")
    assert result.metadata["approved"] is False
    assert result.metadata["attempts"] == 2


def test_evaluator_robust_against_format_noise() -> None:
    """Si el modelo añade ruido alrededor de 'VEREDICTO: APROBADO', debe aceptarlo."""

    def responder(messages):
        if "editor de spoilers" in messages[0].content.lower():
            return (
                "Pensando bien la reseña...\n"
                "VEREDICTO: APROBADO\n"
                "Notas adicionales: ninguna observación crítica."
            )
        return "Reseña sin spoilers."

    workflow = EvaluatorOptimizerWorkflow(llm=ScriptedLLM(responder=responder), max_retries=3)
    result = workflow.run("Watchmen")
    assert result.metadata["approved"] is True


def test_evaluator_rejects_when_no_verdict_line() -> None:
    """Si el modelo no emite 'VEREDICTO:', se considera rechazado."""

    def responder(messages):
        if "editor de spoilers" in messages[0].content.lower():
            return "Esto es ambiguo; necesito más contexto."
        return "Reseña ambigua."

    workflow = EvaluatorOptimizerWorkflow(llm=ScriptedLLM(responder=responder), max_retries=1)
    result = workflow.run("Watchmen")
    assert result.metadata["approved"] is False
