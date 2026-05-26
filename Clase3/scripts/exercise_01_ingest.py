"""Ejercicio 01 — Ingesta del corpus en ChromaDB.

Carga los .md de ``data/comics`` y ``data/musica`` en dos colecciones
independientes de Chroma, persistidas en ``.chroma/``.

Uso:
    uv run python scripts/exercise_01_ingest.py
    uv run python scripts/exercise_01_ingest.py --reset
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401  - agrega src/ al sys.path
from clase3.config import COMICS_DIR, MUSIC_DIR
from clase3.factory import build_rag_bundle


def _print_report(label: str, report) -> None:
    print(
        f"  ✔ [{label}] documentos={report.documents} "
        f"chunks={report.chunks} total_en_store={report.total_in_store}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingesta del corpus en ChromaDB")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Borra las colecciones existentes antes de indexar",
    )
    args = parser.parse_args()

    print("═" * 70)
    print("INGESTA · Cómics y canciones latinoamericanas → ChromaDB")
    print("═" * 70)

    comics = build_rag_bundle(
        collection_name="comics",
        data_directory=COMICS_DIR,
        reset_store=args.reset,
        extra_metadata={"corpus": "comics"},
    )
    musica = build_rag_bundle(
        collection_name="musica",
        data_directory=MUSIC_DIR,
        reset_store=args.reset,
        extra_metadata={"corpus": "musica"},
    )

    print("\n▶ Indexando cómics...")
    report_comics = comics.indexer.index()
    _print_report("comics", report_comics)

    print("\n▶ Indexando canciones...")
    report_musica = musica.indexer.index()
    _print_report("musica", report_musica)

    print("\n✅ Ingesta completada.")
    print(f"   Total chunks en 'comics': {comics.store.count()}")
    print(f"   Total chunks en 'musica': {musica.store.count()}")


if __name__ == "__main__":
    main()
