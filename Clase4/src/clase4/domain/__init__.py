"""Entidades de dominio (DTOs) para workflows."""

from clase4.domain.messages import AIMessage, BaseMessage, SystemMessage, UserMessage
from clase4.domain.workflow import WorkflowResult, WorkflowStep

__all__ = [
    "AIMessage",
    "BaseMessage",
    "SystemMessage",
    "UserMessage",
    "WorkflowResult",
    "WorkflowStep",
]
