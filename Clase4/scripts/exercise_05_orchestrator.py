"""Ejercicio 05 — Orchestrator + Workers (descomposición y síntesis)."""

from __future__ import annotations

import _bootstrap  # noqa: F401
from clase4.factory import build_llm
from clase4.workflows import OrchestratorWorkflow

QUERIES = [
    "Supernatural (1999) — Carlos Santana",
    "La Negra Tiene Tumbao (2001) — Celia Cruz",
]


def main() -> None:
    print("═" * 70)
    print("WORKFLOW 05 · Orchestrator (descomposición → workers → síntesis)")
    print("═" * 70)

    workflow = OrchestratorWorkflow(llm=build_llm())
    for query in QUERIES:
        result = workflow.run(query)
        print("─" * 70)
        print(f"🎼 Tópico: {result.input}")
        print(f"   Sub-tareas detectadas: {result.metadata['sub_tasks']}\n")
        print(f"{result.output}\n")


if __name__ == "__main__":
    main()
