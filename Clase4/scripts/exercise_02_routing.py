"""Ejercicio 02 — Routing: el LLM elige al especialista correcto."""

from __future__ import annotations

import _bootstrap  # noqa: F401
from clase4.factory import build_llm
from clase4.workflows import RoutingWorkflow

QUERIES = [
    "¿Qué simboliza la máscara de Guy Fawkes en V de Vendetta?",
    "¿Cuál es la importancia del montuno de piano de Papo Lucca en Quimbara?",
    "¿Por qué Matador de Los Fabulosos Cadillacs es considerada un himno político del rock latinoamericano?",
    "¿Cuál es el dilema moral central de Ozymandias al final de Watchmen?",
]


def main() -> None:
    print("═" * 70)
    print("WORKFLOW 02 · Routing (especialistas dinámicos)")
    print("═" * 70)

    workflow = RoutingWorkflow(llm=build_llm())
    for query in QUERIES:
        result = workflow.run(query)
        print("─" * 70)
        print(f"❓ {result.input}")
        print(f"🎯 Especialista elegido: {result.metadata['chosen_specialist']}")
        print(f"\n{result.output}\n")


if __name__ == "__main__":
    main()
