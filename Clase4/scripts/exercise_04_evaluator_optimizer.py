"""Ejercicio 04 — Evaluator-Optimizer (writer ↔ critic loop)."""

from __future__ import annotations

import _bootstrap  # noqa: F401
from clase4.factory import build_llm
from clase4.workflows import EvaluatorOptimizerWorkflow

QUERIES = [
    "Watchmen — Alan Moore (1986)",
    "V de Vendetta — Alan Moore (1982-1989)",
    "Persépolis — Marjane Satrapi (2000-2003)",
]


def main() -> None:
    print("═" * 70)
    print("WORKFLOW 04 · Evaluator-Optimizer (loop sin spoilers)")
    print("═" * 70)

    workflow = EvaluatorOptimizerWorkflow(llm=build_llm(), max_retries=3)
    for query in QUERIES:
        result = workflow.run(query)
        print("─" * 70)
        print(f"📖 Cómic: {result.input}")
        meta = result.metadata
        print(f"   Aprobado: {meta['approved']}  intentos: {meta['attempts']}\n")
        print(f"{result.output}\n")


if __name__ == "__main__":
    main()
