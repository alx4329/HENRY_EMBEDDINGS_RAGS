"""Configuración global del paquete clase3.

Carga variables de entorno desde el .env del repo (raíz del proyecto) y expone
constantes derivadas. Un único punto de truth para todo el resto del código.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True), override=False)


def get_env(key: str, default: str | None = None, *, required: bool = False) -> str | None:
    """Lee una variable de entorno con validación opcional."""
    value = os.getenv(key, default)
    if required and not value:
        raise RuntimeError(f"Falta la variable de entorno {key!r} en el .env")
    return value


# ── API keys ──────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = get_env("OPENAI_API_KEY", required=True)  # type: ignore[assignment]

# ── Rutas ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
COMICS_DIR: Path = DATA_DIR / "comics"
MUSIC_DIR: Path = DATA_DIR / "musica"
CHROMA_PERSIST_DIR: Path = PROJECT_ROOT / ".chroma"

# ── Modelos por defecto ───────────────────────────────────────────────────────
DEFAULT_CHAT_MODEL: str = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL: str = "text-embedding-3-small"
DEFAULT_TEMPERATURE: float = 0.2
DEFAULT_TOP_K: int = 4
DEFAULT_CHUNK_SIZE: int = 900
DEFAULT_CHUNK_OVERLAP: int = 150
