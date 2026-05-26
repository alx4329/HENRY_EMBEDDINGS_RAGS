"""Configuración del paquete clase4. Lee el .env de la raíz del repo."""

from __future__ import annotations

import os

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True), override=False)


def get_env(key: str, default: str | None = None, *, required: bool = False) -> str | None:
    value = os.getenv(key, default)
    if required and not value:
        raise RuntimeError(f"Falta la variable de entorno {key!r} en el .env")
    return value


OPENAI_API_KEY: str = get_env("OPENAI_API_KEY", required=True)  # type: ignore[assignment]

DEFAULT_MODEL: str = "gpt-4o-mini"
DEFAULT_TEMPERATURE: float = 0.4
