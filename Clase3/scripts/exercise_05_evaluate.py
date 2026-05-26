"""Ejercicio 05 — Evaluación cuantitativa del pipeline RAG.

Para cada pregunta calculamos:
- ``hit@k``: si alguno de los chunks recuperados pertenece al documento
  esperado.
- ``mrr``: Mean Reciprocal Rank — 1/posición del primer documento correcto.
- ``avg_similarity``: similitud promedio de los top-k recuperados.

Las preguntas y los documentos esperados son etiquetados manualmente para
poder evaluar de forma objetiva sin ground-truth humano externo.

Uso:
    uv run python scripts/exercise_05_evaluate.py
"""

from __future__ import annotations

from dataclasses import dataclass

import _bootstrap  # noqa: F401
from clase3.config import COMICS_DIR, MUSIC_DIR
from clase3.factory import build_rag_bundle


@dataclass(frozen=True)
class EvalCase:
    question: str
    expected_doc_id: str  # el "stem" del archivo .md (ej: "watchmen")
    collection: str


CASES: list[EvalCase] = [
    EvalCase(
        "¿Cuál es la frase 'Quis custodiet ipsos custodes?' y de qué cómic proviene?",
        "watchmen",
        "comics",
    ),
    EvalCase(
        "¿Qué le ocurre a Valerie en V de Vendetta?",
        "v_for_vendetta",
        "comics",
    ),
    EvalCase(
        "¿Cuál es el dilema moral de Vladek narrando el Holocausto en Maus?",
        "maus",
        "comics",
    ),
    EvalCase(
        "¿Cuál es la frase de la abuela de Marji sobre mantener la espalda recta?",
        "persepolis",
        "comics",
    ),
    EvalCase(
        "¿Cómo describen el solo de guitarra de 'Black Magic Woman'?",
        "santana_black_magic_woman",
        "musica",
    ),
    EvalCase(
        "¿Quién escribió la letra de 'Smooth' de Santana?",
        "santana_smooth",
        "musica",
    ),
    EvalCase(
        "¿Por qué 'Quimbara' fue importante para el regreso de Celia Cruz a los charts?",
        "celia_cruz_quimbara",
        "musica",
    ),
    EvalCase(
        "¿Cuál es la crítica histórica que hace 'V Centenario' a 1492?",
        "cadillacs_v_centenario",
        "musica",
    ),
]


def reciprocal_rank(retrieved_doc_ids: list[str], expected: str) -> float:
    for pos, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id == expected:
            return 1.0 / pos
    return 0.0


def main() -> None:
    print("═" * 70)
    print("EVALUACIÓN · hit@k, MRR, similitud promedio")
    print("═" * 70)

    bundles = {
        "comics": build_rag_bundle(
            collection_name="comics",
            data_directory=COMICS_DIR,
            extra_metadata={"corpus": "comics"},
            top_k=3,
        ),
        "musica": build_rag_bundle(
            collection_name="musica",
            data_directory=MUSIC_DIR,
            extra_metadata={"corpus": "musica"},
            top_k=3,
        ),
    }
    for name, bundle in bundles.items():
        if bundle.store.count() == 0:
            print(f"⚠ Colección '{name}' vacía; ingestando…")
            bundle.indexer.index()

    hits = 0
    rr_total = 0.0
    sim_total = 0.0
    rows: list[tuple[str, str, float, float]] = []

    for case in CASES:
        bundle = bundles[case.collection]
        retrieved = bundle.pipeline.retrieve_only(case.question)
        retrieved_ids = [item.chunk.document_id for item in retrieved]
        hit = case.expected_doc_id in retrieved_ids
        rr = reciprocal_rank(retrieved_ids, case.expected_doc_id)
        avg_sim = (
            sum(item.similarity for item in retrieved) / len(retrieved) if retrieved else 0.0
        )
        hits += int(hit)
        rr_total += rr
        sim_total += avg_sim
        rows.append((case.question, case.expected_doc_id, rr, avg_sim))
        marker = "✔" if hit else "✘"
        print(f"{marker} {case.collection:>6} | esperado={case.expected_doc_id:<28} | rr={rr:.2f} | sim≈{avg_sim:.2f}")

    n = len(CASES)
    print("─" * 70)
    print(f"hit@k     : {hits}/{n} = {hits / n:.0%}")
    print(f"MRR       : {rr_total / n:.3f}")
    print(f"sim media : {sim_total / n:.3f}")


if __name__ == "__main__":
    main()
