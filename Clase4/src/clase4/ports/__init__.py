"""Ports (Protocols) — abstracciones que los workflows pueden inyectar."""

from clase4.ports.llm import LLMClient
from clase4.ports.workflow import Workflow

__all__ = ["LLMClient", "Workflow"]
