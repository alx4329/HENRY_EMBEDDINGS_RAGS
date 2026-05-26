"""Estructuras compartidas por todos los workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class WorkflowStep:
    """Una observación inmutable de un paso ya ejecutado en un workflow.

    Es el equivalente a un *snapshot* en una state machine: permite auditar
    el flujo completo paso a paso.
    """

    name: str
    output: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class WorkflowResult:
    """Resultado final de la ejecución de un workflow.

    Atributos:
        input: la entrada original del usuario.
        output: la respuesta final del workflow.
        steps: lista cronológica de pasos ejecutados.
        metadata: información extra (tokens, costos, decisiones, etc.).
    """

    input: str
    output: str
    steps: list[WorkflowStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
