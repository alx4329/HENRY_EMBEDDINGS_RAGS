# Clase 4 — Workflows agenticos con LLMs (arquitectura SOLID)

> Módulo 4 del programa **HENRY · AI Engineering**.
> Cinco patrones canónicos de orquestación de agentes (Anthropic + Udacity)
> implementados en Python 3.11+ con arquitectura modular SOLID y un caso de
> uso unificado: **cómics canónicos + música latinoamericana** (Santana,
> Celia Cruz, Los Fabulosos Cadillacs).

---

## 1. ¿Qué es un workflow agenticos?

Mientras que un único prompt es estático y monolítico, un **workflow** es
una composición de varios prompts/agentes que cooperan, se evalúan y se
revisan entre sí. Los workflows resuelven cuatro problemas que un prompt
sólo no puede:

1. **Descomposición de tareas complejas** — partir una tarea grande en
   pasos pequeños y especializados (chaining, orchestrator).
2. **Decisión dinámica** — elegir el camino correcto en función del input
   (routing).
3. **Diversidad de perspectivas** — combinar varias miradas independientes
   y sintetizarlas (parallelization).
4. **Calidad por iteración** — usar un crítico para mejorar el resultado
   hasta cumplir criterios (evaluator-optimizer).

---

## 2. Los cinco patrones implementados

```text
┌──────────────────┬────────────────────────────────────────────────────┐
│ chaining         │ A → B  (pipeline lineal)                           │
│ routing          │ router → {A, B, C}  (one-of-N)                     │
│ parallel         │ fan-out [A,B,C] → fan-in (síntesis)                │
│ evaluator_opt    │ writer ⇄ critic  (loop con stopping condition)     │
│ orchestrator     │ planner → workers tipados → synthesizer            │
└──────────────────┴────────────────────────────────────────────────────┘
```

### 2.1 Chaining (`workflows/chaining.py`)

Un agente **investigador** produce una ficha técnica estructurada de una
canción; un agente **redactor** la transforma en reseña periodística.
Demuestra Single Responsibility: cada agente hace una sola cosa.

### 2.2 Routing (`workflows/routing.py`)

Tres especialistas: `comics_expert`, `salsa_expert`, `latin_rock_expert`.
Un **router LLM** decide a quién enviar la pregunta. Añadir un cuarto
especialista (jazz, hip hop…) es escribir una clase que cumpla el
`SpecialistAgent` Protocol y registrarla — el router no cambia (OCP).

### 2.3 Parallelization (`workflows/parallel.py`)

Tres analistas (musical, histórico, social) trabajan **en paralelo**
sobre el mismo tema usando `ThreadPoolExecutor` (el GIL se libera durante
el I/O de la API). Un editor cultural agrega los aportes (map-reduce).

### 2.4 Evaluator-Optimizer (`workflows/evaluator_optimizer.py`)

Un escritor genera una reseña de cómic; un evaluador estricto rechaza
spoilers de final. El feedback del evaluador se reinyecta en el prompt del
escritor en la siguiente iteración. Stopping condition: ``max_retries``.

### 2.5 Orchestrator (`workflows/orchestrator.py`)

Un editor jefe (LLM) **descompone** un álbum en sub-tareas tipadas
(`datos`, `contexto`, `musical`, `recepcion`, `legado`). Cada sub-tarea va
a un worker especializado. Un editor de cierre **sintetiza** todo en un
ensayo largo. Si el JSON del planner es inválido, se aplica un fallback
con un plan por defecto.

---

## 3. Arquitectura SOLID

```text
Clase4/
├── src/clase4/
│   ├── config.py        # carga .env + defaults
│   ├── domain/          # DTOs inmutables (BaseMessage, WorkflowResult, …)
│   ├── ports/           # Protocols: LLMClient, Workflow
│   ├── adapters/        # implementaciones concretas (OpenAIChatClient)
│   ├── workflows/       # los 5 patrones, cada uno aislado y testeable
│   └── factory.py       # composition root: build_llm + build_all_workflows
├── scripts/             # ejercicios end-to-end ejecutables con `uv run`
└── tests/               # 14 tests con FakeLLM (sin red)
```

| Principio | Aplicación en Clase 4 |
|-----------|------------------------|
| **S** | Cada agente tiene UN system prompt y UNA responsabilidad. |
| **O** | Para añadir un workflow nuevo no hay que tocar los existentes. |
| **L** | Cualquier `LLMClient` (OpenAI, Anthropic, fake) se inserta sin cambios. |
| **I** | Los Protocols (`LLMClient`, `Workflow`, `SpecialistAgent`) son pequeños. |
| **D** | Los workflows dependen de Protocols, no de `openai`. Los tests inyectan un `ScriptedLLM` sin red. |

---

## 4. Instalación

Requiere Python 3.11+ y [`uv`](https://github.com/astral-sh/uv). El
`.env` con `OPENAI_API_KEY` debe estar en la raíz del repo
(`HENRY_EMBEDDINGS_RAGS/.env`, ya existente).

```bash
cd Clase4
uv sync
```

---

## 5. Ejecución de los ejercicios

```bash
uv run python scripts/exercise_01_chaining.py
uv run python scripts/exercise_02_routing.py
uv run python scripts/exercise_03_parallel.py
uv run python scripts/exercise_04_evaluator_optimizer.py
uv run python scripts/exercise_05_orchestrator.py
```

### Calidad

```bash
uv run ruff check src scripts tests
uv run pytest tests/ -v             # 14 tests, ~0.05s, sin red
```

---

## 6. Snippet de uso programático

```python
from clase4.factory import build_llm, build_all_workflows

llm = build_llm(model="gpt-4o-mini", temperature=0.4)
workflows = build_all_workflows(llm=llm)

# Reseña en cadena
review = workflows["chaining"].run("Oye Cómo Va — Carlos Santana")
print(review.output)

# Pregunta enrutada
ans = workflows["routing"].run("¿Quién produjo Supernatural y por qué fue importante?")
print(ans.metadata["chosen_specialist"], "→", ans.output)

# Ensayo orquestado
essay = workflows["orchestrator"].run("La Negra Tiene Tumbao — Celia Cruz")
print(essay.metadata["sub_tasks"])  # ['datos', 'contexto', 'musical', 'recepcion', 'legado']
print(essay.output)
```

---

## 7. Anatomía de los workflows (código relevante)

### Chaining

```python
class ChainingWorkflow:
    def run(self, user_input: str) -> WorkflowResult:
        research = self._researcher.run(user_input)
        review   = self._writer.run(user_input, research)
        return WorkflowResult(
            input=user_input, output=review,
            steps=[WorkflowStep("researcher", research), WorkflowStep("writer", review)],
        )
```

### Parallelization (fan-out / fan-in con ThreadPoolExecutor)

```python
with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
    future_map = {
        executor.submit(specialist.analyze, user_input): specialist.name
        for specialist in self._specialists
    }
    for future in as_completed(future_map):
        partials[future_map[future]] = future.result()
summary = self._summarize(user_input, partials)
```

### Evaluator-Optimizer (loop con stopping condition)

```python
for attempt in range(1, self._max_retries + 1):
    review = self._writer.write(user_input, feedback=feedback)
    approved, verdict = self._evaluator.evaluate(review)
    if approved:
        return WorkflowResult(... approved=True, attempts=attempt)
    feedback = verdict
```

### Orchestrator (planner + workers + synthesizer)

```python
analysis, tasks = self._decompose(user_input)         # JSON con sub-tareas
partials = [(t, self._worker.run(user_input, t)) for t in tasks]
final    = self._synthesize(user_input, partials)
```

---

## 8. Ciclos de revisión aplicados

| Ciclo | Diagnóstico | Mejora |
|-------|-------------|--------|
| **1** | Implementación base. Todos los workflows ejecutan con OpenAI real. | 11 tests unitarios con `ScriptedLLM`. |
| **2** | `evaluator._parse_verdict` usaba `"APROBADO" in ...` (frágil); el orquestador no tenía fallback ante JSON malformado. | Parser por línea ``VEREDICTO:`` con token check. Fallback ``_FALLBACK_TASKS`` con plan por defecto. |
| **3** | Faltaban tests para ruido en el veredicto, ausencia de cabecera y JSON inválido. | +3 tests de robustez (total **14**). Verificación final: ``ruff`` limpio, todos los workflows ejecutan end-to-end. |

---

## 9. Cómo extender el proyecto

| Quiero… | Hago… |
|---------|-------|
| añadir un especialista para `bossa_nova_expert` | crear una clase con `name`, `description`, `answer`; registrarla en `build_default_specialists` o pasarla por DI a `RoutingWorkflow`. |
| reemplazar OpenAI por Anthropic | implementar `AnthropicChatClient` que cumpla el Protocol `LLMClient`; nada en `workflows/` cambia. |
| añadir un sexto patrón (e.g. ReAct con tool-use) | crear `workflows/react.py` con su clase, registrarla en `factory.build_all_workflows`. |
| auditar costo / tokens | inyectar un wrapper sobre `LLMClient` que cuente tokens y emita métricas, sin tocar workflows. |
| persistir `WorkflowResult` | serializar `WorkflowResult` a JSON (es un dataclass plano). |

---

## 10. Relación con Clase 3

| | Clase 3 (RAG) | Clase 4 (Workflows) |
|--|--------------|----------------------|
| Foco | recuperar contexto | orquestar agentes |
| Componente clave | `VectorStore` + `Retriever` | `LLMClient` + cadenas de agentes |
| Patrón base | retrieve → augment → generate | chaining, routing, parallel, evaluator, orchestrator |
| Dataset | mismo corpus (cómics + música) | mismo corpus (cómics + música) |
| Composición | `factory.build_rag_bundle` | `factory.build_all_workflows` |

Ambas clases comparten la **misma filosofía de arquitectura**: `domain` /
`ports` / `adapters` / `services` (o `workflows`), composition root
explícito, tests sin red usando fakes que cumplen los Protocols.
