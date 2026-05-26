"""Composition root para Clase 4.

Lugar único donde se eligen los adapters concretos. El resto del código
sólo conoce abstracciones.
"""

from __future__ import annotations

from clase4.adapters.openai_llm import OpenAIChatClient
from clase4.config import DEFAULT_MODEL, DEFAULT_TEMPERATURE, OPENAI_API_KEY
from clase4.ports.llm import LLMClient
from clase4.ports.workflow import Workflow
from clase4.workflows.chaining import ChainingWorkflow
from clase4.workflows.evaluator_optimizer import EvaluatorOptimizerWorkflow
from clase4.workflows.orchestrator import OrchestratorWorkflow
from clase4.workflows.parallel import ParallelInsightsWorkflow
from clase4.workflows.routing import RoutingWorkflow


def build_llm(
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> LLMClient:
    """Construye el adapter LLM por defecto (OpenAI Chat Completions)."""
    return OpenAIChatClient(model=model, temperature=temperature, api_key=OPENAI_API_KEY)


def build_all_workflows(llm: LLMClient | None = None) -> dict[str, Workflow]:
    """Devuelve los cinco workflows canónicos listos para usarse.

    Se tipa contra el Protocol ``Workflow`` (no contra clases concretas) para
    honrar Dependency Inversion: el código cliente sólo necesita
    ``run(user_input) -> WorkflowResult`` y ``name``.
    """
    llm = llm or build_llm()
    return {
        "chaining": ChainingWorkflow(llm=llm),
        "routing": RoutingWorkflow(llm=llm),
        "parallel": ParallelInsightsWorkflow(llm=llm),
        "evaluator_optimizer": EvaluatorOptimizerWorkflow(llm=llm),
        "orchestrator": OrchestratorWorkflow(llm=llm),
    }
