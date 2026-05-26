"""Workflow 4 — Evaluator-Optimizer (loop con criterio de convergencia).

Patrón: un agente genera contenido, otro lo evalúa contra criterios
explícitos. Si no pasa, el evaluador devuelve feedback y el generador
itera. Se sale con un máximo de reintentos para acotar costo.

Caso: redactar una reseña sobre un cómic clásico que NO contenga
spoilers críticos del final. Un evaluador estricto rechaza la versión si
detecta spoilers de tramo final o cierre.

Esto demuestra:
- **Stopping condition**: hardcoded en ``max_retries`` para evitar bucles
  infinitos.
- **Closed feedback loop**: el feedback del evaluador modifica el prompt
  del generador en la siguiente iteración.
"""

from __future__ import annotations

from clase4.domain.messages import SystemMessage, UserMessage
from clase4.domain.workflow import WorkflowResult, WorkflowStep
from clase4.ports.llm import LLMClient

WRITER_PROMPT = (
    "Eres un crítico literario. Recibes el nombre de un cómic y escribes una "
    "reseña de 150-200 palabras que invite a leer la obra. Reglas duras:\n"
    "1. NO reveles cómo termina la obra.\n"
    "2. NO menciones la muerte, suicidio, traición ni decisión final de "
    "ningún personaje principal.\n"
    "3. Habla de tema, contexto histórico, autor, estilo visual y por qué "
    "la obra es relevante."
)

EVALUATOR_PROMPT = (
    "Eres un editor de spoilers. Recibes una reseña de cómic y debes decidir "
    "si contiene spoilers de tramo final (muerte de personajes, decisión "
    "final, revelación de antagonista, etc.). Responde EXACTAMENTE en este "
    "formato:\n"
    "VEREDICTO: APROBADO   (si no hay spoilers)\n"
    "VEREDICTO: RECHAZADO  (si hay spoilers)\n"
    "Si rechazas, agrega después una línea:\n"
    "FEEDBACK: <una frase explicando qué spoiler hay>"
)


class ReviewWriter:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def write(self, comic_name: str, feedback: str | None = None) -> str:
        prompt = f"Cómic: {comic_name}\n\nEscribe la reseña."
        if feedback:
            prompt += f"\n\nFeedback del editor anterior: {feedback}\nCorrige."
        messages = [
            SystemMessage(content=WRITER_PROMPT),
            UserMessage(content=prompt),
        ]
        return self._llm.chat(messages, temperature=0.6).content


class ReviewEvaluator:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def evaluate(self, review: str) -> tuple[bool, str]:
        messages = [
            SystemMessage(content=EVALUATOR_PROMPT),
            UserMessage(content=review),
        ]
        verdict = self._llm.chat(messages, temperature=0.0).content
        approved = self._parse_verdict(verdict)
        return approved, verdict

    @staticmethod
    def _parse_verdict(verdict: str) -> bool:
        """Parsea el veredicto buscando línea ``VEREDICTO:`` y leyendo el token."""
        for raw_line in verdict.splitlines():
            line = raw_line.strip().upper()
            if line.startswith("VEREDICTO:"):
                tokens = line.split(":", 1)[1].strip().split()
                return bool(tokens) and tokens[0] == "APROBADO"
        return False


class EvaluatorOptimizerWorkflow:
    name = "evaluator_optimizer"

    def __init__(self, llm: LLMClient, *, max_retries: int = 4) -> None:
        self._writer = ReviewWriter(llm)
        self._evaluator = ReviewEvaluator(llm)
        self._max_retries = max_retries

    def run(self, user_input: str) -> WorkflowResult:
        steps: list[WorkflowStep] = []
        feedback: str | None = None
        review = ""

        for attempt in range(1, self._max_retries + 1):
            review = self._writer.write(user_input, feedback=feedback)
            steps.append(WorkflowStep(name=f"writer_attempt_{attempt}", output=review))

            approved, verdict = self._evaluator.evaluate(review)
            steps.append(
                WorkflowStep(
                    name=f"evaluator_attempt_{attempt}",
                    output=verdict,
                    metadata={"approved": approved},
                )
            )
            if approved:
                return WorkflowResult(
                    input=user_input,
                    output=review,
                    steps=steps,
                    metadata={"approved": True, "attempts": attempt},
                )
            feedback = verdict

        return WorkflowResult(
            input=user_input,
            output=review,
            steps=steps,
            metadata={"approved": False, "attempts": self._max_retries},
        )
