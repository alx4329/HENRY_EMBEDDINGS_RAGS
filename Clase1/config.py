import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True), override=False)


def get_env(key: str, default: str | None = None, *, required: bool = False) -> str | None:
    value = os.getenv(key, default)
    if required and not value:
        raise RuntimeError(f"Falta la variable de entorno {key!r} en el .env")
    return value


OPENAI_API_KEY = get_env("OPENAI_API_KEY", required=True)
TVLY_API_KEY = get_env("TVLY_API_KEY")
LANGCHAIN_API_KEY = get_env("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2 = get_env("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT = get_env("LANGCHAIN_ENDPOINT")
