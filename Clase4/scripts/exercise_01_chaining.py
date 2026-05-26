"""Ejercicio 01 — Prompt Chaining: investigador → redactor.

Demuestra el patrón clásico de cadena. Una canción entra como input,
sale una reseña periodística después de pasar por dos agentes.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
from clase4.factory import build_llm
from clase4.workflows import ChainingWorkflow

QUERIES = [
    "Oye Cómo Va — Carlos Santana",
    "La Vida Es un Carnaval — Celia Cruz",
    "Matador — Los Fabulosos Cadillacs",
]


def main() -> None:
    print("═" * 70)
    print("WORKFLOW 01 · Prompt Chaining (investigador → redactor)")
    print("═" * 70)

    workflow = ChainingWorkflow(llm=build_llm())
    for query in QUERIES:
        result = workflow.run(query)
        print("─" * 70)
        print(f"🎵 {result.input}")
        print(f"\n[paso 1 · investigador]\n{result.steps[0].output[:400]}…\n")
        print(f"[paso 2 · reseña final]\n{result.output}\n")


if __name__ == "__main__":
    main()
