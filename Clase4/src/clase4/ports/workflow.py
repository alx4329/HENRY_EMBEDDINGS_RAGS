"""Port: workflow ejecutable.

Cualquier clase que cumpla este Protocol puede ser orquestada por otros
componentes (un meta-orquestador, una API, una CLI). Permite componer
workflows entre sí (Open/Closed).
"""

from __future__ import annotations

from typing import Protocol

from clase4.domain.workflow import WorkflowResult


class Workflow(Protocol):
    """Contrato mínimo: ``run(input) -> WorkflowResult``."""

    @property
    def name(self) -> str: ...

    def run(self, user_input: str) -> WorkflowResult: ...
