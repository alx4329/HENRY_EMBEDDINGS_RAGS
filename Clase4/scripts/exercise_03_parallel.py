"""Ejercicio 03 — Parallelization (fan-out / fan-in)."""

from __future__ import annotations

import _bootstrap  # noqa: F401
from clase4.factory import build_llm
from clase4.workflows import ParallelInsightsWorkflow

QUERIES = [
    "El impacto cultural de 'V Centenario' de Los Fabulosos Cadillacs en la conmemoración de 1992",
    "La fusión musical en 'Black Magic Woman' de Santana entre blues británico y latin rock",
]


def main() -> None:
    print("═" * 70)
    print("WORKFLOW 03 · Parallelization (3 analistas en paralelo + síntesis)")
    print("═" * 70)

    workflow = ParallelInsightsWorkflow(llm=build_llm())
    for query in QUERIES:
        result = workflow.run(query)
        print("─" * 70)
        print(f"📌 Tema: {result.input}")
        print(f"   Especialistas consultados: {result.metadata['specialists']}\n")
        print(f"{result.output}\n")


if __name__ == "__main__":
    main()
