"""Workflows: cinco patrones agenticos canónicos."""

from clase4.workflows.chaining import (
    ChainingWorkflow,
    LyricResearchAgent,
    LyricWriterAgent,
)
from clase4.workflows.evaluator_optimizer import (
    EvaluatorOptimizerWorkflow,
    ReviewEvaluator,
    ReviewWriter,
)
from clase4.workflows.orchestrator import (
    OrchestratorWorkflow,
    WorkerAgent,
)
from clase4.workflows.parallel import (
    ParallelInsightsWorkflow,
    Specialist,
)
from clase4.workflows.routing import (
    RoutingWorkflow,
    SpecialistAgent,
)

__all__ = [
    "ChainingWorkflow",
    "EvaluatorOptimizerWorkflow",
    "LyricResearchAgent",
    "LyricWriterAgent",
    "OrchestratorWorkflow",
    "ParallelInsightsWorkflow",
    "ReviewEvaluator",
    "ReviewWriter",
    "RoutingWorkflow",
    "Specialist",
    "SpecialistAgent",
    "WorkerAgent",
]
