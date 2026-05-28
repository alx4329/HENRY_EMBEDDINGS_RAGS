"""Genera 02_prompt_strategies_langgraph.ipynb a partir de celdas declaradas.

Uso:
    python _build_notebook.py

Salida:
    02_prompt_strategies_langgraph.ipynb

Diseñado para no requerir red ni claves: todos los ejemplos usan un
``ScriptedLLM`` que devuelve respuestas predeterminadas. La versión
con LangGraph se ejecuta sólo si la librería está instalada.
"""

from __future__ import annotations

import json
from pathlib import Path

CELLS: list[dict] = []


def md(source: str) -> None:
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": source})


def code(source: str) -> None:
    CELLS.append(
        {
            "cell_type": "code",
            "metadata": {},
            "source": source,
            "execution_count": None,
            "outputs": [],
        }
    )


# =============================================================================
# 0. Título e índice
# =============================================================================

md(
    """# Cinco patrones de orquestación de prompts — *con y sin LangGraph*

> Cuando un LLM no alcanza con un único prompt, la solución casi nunca es
> *un prompt más largo*: es **componer varios prompts** con una arquitectura
> explícita. Esta notebook recorre los cinco patrones canónicos que viven
> en `Clase4/src/clase4/workflows/` y muestra, lado a lado, cómo se ven
> con código artesanal (Python puro) y con [LangGraph](https://langchain-ai.github.io/langgraph/).

## Lo que vas a recorrer

| # | Patrón | Pregunta que responde |
|---|---|---|
| 1 | **Chaining** | ¿Cómo encadeno la salida de un agente como entrada del siguiente? |
| 2 | **Routing** | ¿Cómo elijo qué especialista responde según el tipo de pregunta? |
| 3 | **Parallel** | ¿Cómo lanzo *N* análisis simultáneos y los fusiono después? |
| 4 | **Evaluator-Optimizer** | ¿Cómo itero con feedback hasta pasar un control de calidad? |
| 5 | **Orchestrator + Workers** | ¿Cómo descompongo una tarea difusa en sub-tareas paralelas y las sintetizo? |

## Convención

* **Sin LangGraph** → usamos las clases ya implementadas en el paquete
  `clase4` (`ChainingWorkflow`, `RoutingWorkflow`, ...). El orquestador
  es Python plano: bucles, `ThreadPoolExecutor`, condicionales.
* **Con LangGraph** → reescribimos cada patrón como un `StateGraph` con
  nodos, aristas (a veces condicionales o cíclicas) y un estado tipado
  que viaja entre nodos.

Ambos enfoques son equivalentes; lo que cambia es **quién lleva la
contabilidad del control de flujo**: vos o un motor de grafos.
"""
)

# =============================================================================
# 1. Setup
# =============================================================================

md(
    """## 1. Setup

> **Kernel:** elegí el entorno `Clase4` (donde está instalado el paquete
> `clase4`). Si no tenés un kernel registrado, podés crearlo con:
> ```bash
> cd Clase4
> uv run python -m ipykernel install --user --name=clase4-henry --display-name="Python (Clase 4 · Workflows)"
> ```

La celda siguiente instala `langgraph` *sólo* si no está disponible. La
notebook funciona sin él: cada sección "sin LangGraph" es autónoma.
"""
)

code(
    """import importlib
import subprocess
import sys

def _ensure(pkg: str, import_name: str | None = None) -> None:
    name = import_name or pkg
    try:
        importlib.import_module(name)
    except ImportError:
        print(f"Instalando {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

_ensure("langgraph")

# Las versiones \"sin LangGraph\" vienen del paquete del proyecto.
from clase4.domain.messages import AIMessage, BaseMessage, SystemMessage, UserMessage
from clase4.domain.workflow import WorkflowResult, WorkflowStep
from clase4.ports.llm import LLMClient
from clase4.workflows.chaining import ChainingWorkflow
from clase4.workflows.routing import RoutingWorkflow, build_default_specialists
from clase4.workflows.parallel import ParallelInsightsWorkflow
from clase4.workflows.evaluator_optimizer import EvaluatorOptimizerWorkflow
from clase4.workflows.orchestrator import OrchestratorWorkflow, SubTask

print("OK — paquete clase4 disponible.")
"""
)

# =============================================================================
# 2. ScriptedLLM
# =============================================================================

md(
    """## 2. Un LLM *scripted* para no depender de la red

Para que la notebook corra sin clave de OpenAI definimos un cliente
que implementa el `Protocol` `LLMClient` y responde según el contenido
del último mensaje. Es el mismo truco que usa la suite de tests del
paquete (`Clase4/tests/_fake_llm.py`).

Lo importante para el patrón arquitectónico es:

* Los workflows **no conocen al LLM concreto**, sólo el `Protocol`.
* Eso nos deja inyectar OpenAI en producción y `ScriptedLLM` en clase.
"""
)

code(
    '''from dataclasses import dataclass, field
from typing import Callable

@dataclass
class ScriptedLLM:
    """LLM determinístico para demos sin red."""

    responder: Callable[[list[BaseMessage]], str]
    model_name_value: str = "scripted"
    calls: list[list[BaseMessage]] = field(default_factory=list)

    @property
    def model_name(self) -> str:
        return self.model_name_value

    def chat(
        self,
        messages: list[BaseMessage],
        *,
        temperature: float | None = None,
    ) -> AIMessage:
        self.calls.append(messages)
        return AIMessage(content=self.responder(messages))


def make_responder(rules: list[tuple[str, str]], default: str = "(sin respuesta)") -> Callable:
    """Devuelve un responder que matchea por substring en el último user message."""

    def responder(messages: list[BaseMessage]) -> str:
        last = messages[-1].content.lower()
        for needle, reply in rules:
            if needle.lower() in last:
                return reply
        return default

    return responder
'''
)

# =============================================================================
# 3. PATTERN 1 — CHAINING
# =============================================================================

md(
    """---

## Patrón 1 — **Prompt Chaining**

> *La salida del agente A es la entrada del agente B.*

**Cuándo usarlo:** cuando una tarea se descompone naturalmente en
**etapas secuenciales** donde cada etapa enriquece el contexto. Ejemplo
del paquete: dado un título de canción, primero un **investigador**
arma una ficha técnica (año, sello, contexto histórico) y después un
**redactor** transforma esa ficha en una reseña periodística.

### Arquitectura — sin LangGraph

```
   user_input
       │
       ▼
 ┌─────────────┐    research brief
 │ Researcher  │───────────────┐
 └─────────────┘               │
                               ▼
                         ┌──────────┐
                         │  Writer  │── reseña final ─▶ output
                         └──────────┘
```

Es simplemente una llamada después de otra. El "motor" cabe en tres
líneas:

```python
research = researcher.run(user_input)
review = writer.run(user_input, research)
return WorkflowResult(input=user_input, output=review, steps=[...])
```
"""
)

code(
    '''# --- Sin LangGraph: usamos ChainingWorkflow de clase4 con un ScriptedLLM ---

chaining_llm = ScriptedLLM(
    responder=make_responder(
        rules=[
            (
                "investiga la canción",
                "# DATOS BÁSICOS\\n- Año: 1970\\n- Álbum: Abraxas\\n- Sello: Columbia\\n"
                "# CONTEXTO HISTÓRICO\\n- Versión latina del clásico de Tito Puente.",
            ),
            (
                "ficha técnica",
                "Reseña: Oye Cómo Va condensa el cruce entre la timba cubana y el rock psicodélico... (180 palabras)",
            ),
        ],
    )
)

chaining = ChainingWorkflow(llm=chaining_llm)
result = chaining.run("Oye Cómo Va — Carlos Santana")

print("=== STEPS ===")
for step in result.steps:
    print(f"\\n[{step.name}]")
    print(step.output[:160], "...")
print("\\n=== OUTPUT FINAL ===")
print(result.output)
'''
)

md(
    """### Arquitectura — con LangGraph

LangGraph reemplaza el "encadenamiento manual" por un `StateGraph`:

* **Estado**: un `TypedDict` que viaja entre nodos. Cada nodo retorna
  un *patch* (diccionario parcial) que se mergea contra el estado.
* **Nodos**: funciones puras `state -> partial_state`.
* **Aristas**: `add_edge("researcher", "writer")` declara el orden.

```
   START ──▶ researcher ──▶ writer ──▶ END
```

El gráfico se *compila* y queda como un objeto invocable: `graph.invoke({...})`.
"""
)

code(
    '''from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class ChainState(TypedDict):
    song: str
    research: str
    review: str


def researcher_node(state: ChainState) -> dict:
    msgs = [SystemMessage(content="(researcher prompt)"),
            UserMessage(content=f"Investiga la canción: {state[\'song\']}")]
    return {"research": chaining_llm.chat(msgs).content}


def writer_node(state: ChainState) -> dict:
    msgs = [SystemMessage(content="(writer prompt)"),
            UserMessage(content=f"Canción: {state[\'song\']}\\nficha técnica:\\n{state[\'research\']}")]
    return {"review": chaining_llm.chat(msgs).content}


builder = StateGraph(ChainState)
builder.add_node("researcher", researcher_node)
builder.add_node("writer", writer_node)
builder.add_edge(START, "researcher")
builder.add_edge("researcher", "writer")
builder.add_edge("writer", END)
chain_graph = builder.compile()

final_state = chain_graph.invoke({"song": "Oye Cómo Va — Carlos Santana"})
print("research:", final_state["research"][:80], "...")
print("review  :", final_state["review"][:80], "...")

# Diagrama (si Mermaid está disponible)
try:
    print("\\nGrafo (Mermaid):")
    print(chain_graph.get_graph().draw_mermaid())
except Exception as e:
    print("(diagrama no disponible:", e, ")")
'''
)

md(
    """**Diferencia conceptual.** En el código sin LangGraph el control
de flujo está *implícito* en el orden de las líneas; en LangGraph está
**reificado**: los nodos son entidades de primera clase, el grafo se
puede serializar, dibujar, *checkpointear* y reanudar."""
)

# =============================================================================
# 4. PATTERN 2 — ROUTING
# =============================================================================

md(
    """---

## Patrón 2 — **Routing**

> *Una pregunta → un solo especialista entre N posibles.*

**Cuándo usarlo:** cuando tenés agentes con dominios muy distintos
(cómics, salsa, rock latino) y querés mandar la pregunta al que
realmente sabe. Evita el "experto Frankenstein" que sabe poco de todo.

### Arquitectura — sin LangGraph

```
                ┌──────────────────────────────┐
                │           Router LLM          │
                │ (temperature=0.0, devuelve   │
                │   el nombre del experto)      │
                └──────────────────────────────┘
                              │
              ┌───────────────┼────────────────┐
              ▼               ▼                ▼
        comics_expert   salsa_expert   latin_rock_expert
              │               │                │
              └─────────── (uno solo responde) ┘
                              │
                              ▼
                           output
```

El router es un LLM con `temperature=0.0` cuyo único trabajo es decir
*"a quién le toca"*. El despacho luego es un `dict[name → agent]`.
"""
)

code(
    '''# --- Sin LangGraph: RoutingWorkflow ---
# Despachamos por system prompt: router devuelve el nombre del experto,
# cada especialista devuelve una respuesta acorde a su dominio.

def routing_responder(messages: list[BaseMessage]) -> str:
    system = messages[0].content.lower()
    question = messages[-1].content.lower()
    if "router de preguntas" in system:
        if "watchmen" in question or "cómic" in question or "comic" in question:
            return "comics_expert"
        if "celia" in question or "salsa" in question:
            return "salsa_expert"
        return "latin_rock_expert"
    if "crítico de cómic" in system:
        return "Watchmen (1986-87) de Alan Moore es una deconstrucción del superhéroe..."
    if "musicólogo" in system and "salsa" in system:
        return "Celia Cruz dialogó con la Fania All-Stars en los 70..."
    if "rock latinoamericano" in system:
        return "Santana fusionó blues, jazz y ritmos afro-caribeños desde 1969..."
    return "(respuesta genérica)"

routing_llm = ScriptedLLM(responder=routing_responder)
routing = RoutingWorkflow(llm=routing_llm, specialists=build_default_specialists(routing_llm))
res = routing.run("Explica qué hace Watchmen tan icónico")
print("Especialista elegido:", res.metadata.get("chosen_specialist"))
print("Respuesta:", res.output[:120], "...")
'''
)

md(
    """### Arquitectura — con LangGraph

En LangGraph el "despacho" se expresa con
[`add_conditional_edges`](https://langchain-ai.github.io/langgraph/concepts/low_level/#conditional-edges):
una función mira el estado y devuelve el nombre del siguiente nodo.

```
                  START
                    │
                    ▼
                ┌────────┐
                │ router │
                └────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   comics      salsa     latin_rock
        │           │           │
        └─────── END ───────────┘
```
"""
)

code(
    '''from typing import Literal

class RouteState(TypedDict):
    question: str
    specialist: str
    answer: str


def router_node(state: RouteState) -> dict:
    decision = routing_llm.chat(
        [SystemMessage(content="Eres un router de preguntas. Devolvé el nombre del especialista."),
         UserMessage(content=state["question"])]
    ).content.strip().lower()
    for key in ("comics_expert", "salsa_expert", "latin_rock_expert"):
        if key in decision:
            return {"specialist": key}
    return {"specialist": "comics_expert"}  # fallback


def choose_route(state: RouteState) -> Literal["comics_expert", "salsa_expert", "latin_rock_expert"]:
    return state["specialist"]  # type: ignore[return-value]


SPECIALIST_SYSTEMS = {
    "comics_expert": "Eres un crítico de cómic, riguroso con autores y fechas.",
    "salsa_expert":  "Eres un musicólogo especializado en salsa y son cubano.",
    "latin_rock_expert": "Eres un crítico de rock latinoamericano.",
}

def make_specialist_node(name: str):
    def node(state: RouteState) -> dict:
        ans = routing_llm.chat(
            [SystemMessage(content=SPECIALIST_SYSTEMS[name]),
             UserMessage(content=state["question"])]
        ).content
        return {"answer": ans}
    return node


b = StateGraph(RouteState)
b.add_node("router", router_node)
b.add_node("comics_expert", make_specialist_node("comics_expert"))
b.add_node("salsa_expert", make_specialist_node("salsa_expert"))
b.add_node("latin_rock_expert", make_specialist_node("latin_rock_expert"))
b.add_edge(START, "router")
b.add_conditional_edges("router", choose_route,
                        {"comics_expert": "comics_expert",
                         "salsa_expert": "salsa_expert",
                         "latin_rock_expert": "latin_rock_expert"})
for name in ("comics_expert", "salsa_expert", "latin_rock_expert"):
    b.add_edge(name, END)

route_graph = b.compile()
out = route_graph.invoke({"question": "Explica qué hace Watchmen tan icónico"})
print("Elegido:", out["specialist"])
print("Respuesta:", out["answer"][:120], "...")
'''
)

# =============================================================================
# 5. PATTERN 3 — PARALLEL
# =============================================================================

md(
    """---

## Patrón 3 — **Parallelization** (fan-out / fan-in)

> *Lanzá N análisis simultáneos sobre el mismo tema y agregalos al final.*

**Cuándo usarlo:** cuando querés **múltiples perspectivas independientes**
(un musicólogo, un historiador, un sociólogo analizando la misma canción)
y luego una síntesis. Cada análisis es **stateless** respecto del otro,
así que el wall-clock es el del más lento.

### Arquitectura — sin LangGraph

```
                user_input
                    │
        ┌───────────┼───────────┐   (ThreadPoolExecutor)
        ▼           ▼           ▼
   musicólogo  historiador  sociólogo
        │           │           │
        └─────┬─────┴────┬──────┘
              ▼          ▼
          ┌──────────────────┐
          │   synthesizer    │
          └──────────────────┘
                   │
                   ▼
                output
```

El paquete usa `ThreadPoolExecutor` porque la llamada HTTP al LLM libera
el GIL: 3 llamadas en paralelo cuestan ≈ lo que cuesta la más lenta.
"""
)

code(
    '''# --- Sin LangGraph: ParallelInsightsWorkflow ---
# El responder mira el system prompt para distinguir qué especialista llama.

def parallel_responder(messages: list[BaseMessage]) -> str:
    system = messages[0].content.lower() if messages else ""
    if "musicólogo" in system:
        return "Análisis musical: clave Am, montuno cubano, guitarra sobre clave 2-3."
    if "historiador cultural" in system:
        return "Análisis histórico: NY 1970, salsa boom, Fania, contracultura latina."
    if "sociólogo" in system:
        return "Análisis social: la canción atraviesa diásporas caribeñas e identidad afrolatina."
    if "editor cultural" in system:
        return "Síntesis editorial (250 palabras) que cohesiona los tres análisis previos."
    return "(análisis genérico)"

parallel_llm = ScriptedLLM(responder=parallel_responder)
parallel_wf = ParallelInsightsWorkflow(llm=parallel_llm)
res = parallel_wf.run("Oye Cómo Va — Santana")
for step in res.steps:
    print(f"[{step.name}] {step.output[:80]}")
print("\\nFINAL:", res.output[:80], "...")
'''
)

md(
    """### Arquitectura — con LangGraph

En LangGraph el fan-out se logra con **múltiples aristas salientes** desde
`START` y un nodo `synthesizer` con múltiples entrantes. Los nodos
independientes se ejecutan en paralelo automáticamente (LangGraph
detecta que no dependen entre sí).

Para que tres nodos escriban distintas partes del estado sin colisionar
declaramos un *reducer* en el campo `analyses` (lista que se concatena).

```
              START
              ╱ │ ╲
             ╱  │  ╲
            ▼   ▼   ▼
        music hist socio
            ╲   │   ╱
             ╲  │  ╱
              ▼ ▼ ▼
           synthesizer
              │
              ▼
             END
```
"""
)

code(
    '''from operator import add
from typing import Annotated

class ParallelState(TypedDict):
    topic: str
    analyses: Annotated[list[str], add]   # reducer: concatena listas
    summary: str


def musicologo(state: ParallelState) -> dict:
    out = parallel_llm.chat(
        [SystemMessage(content="Eres un musicólogo. Análisis musical estricto."),
         UserMessage(content=state["topic"])]
    ).content
    return {"analyses": [f"musicologo: {out}"]}


def historiador(state: ParallelState) -> dict:
    out = parallel_llm.chat(
        [SystemMessage(content="Eres un historiador cultural. Contexto histórico."),
         UserMessage(content=state["topic"])]
    ).content
    return {"analyses": [f"historiador: {out}"]}


def sociologo(state: ParallelState) -> dict:
    out = parallel_llm.chat(
        [SystemMessage(content="Eres un sociólogo de la cultura popular latinoamericana."),
         UserMessage(content=state["topic"])]
    ).content
    return {"analyses": [f"sociologo: {out}"]}


def synthesizer(state: ParallelState) -> dict:
    body = "\\n".join(state["analyses"])
    out = parallel_llm.chat(
        [SystemMessage(content="Eres un editor cultural. Sintetiza los aportes."),
         UserMessage(content=f"Tema: {state[\'topic\']}\\n\\nAportes:\\n{body}")]
    ).content
    return {"summary": out}


g = StateGraph(ParallelState)
g.add_node("musicologo", musicologo)
g.add_node("historiador", historiador)
g.add_node("sociologo", sociologo)
g.add_node("synthesizer", synthesizer)
for name in ("musicologo", "historiador", "sociologo"):
    g.add_edge(START, name)
    g.add_edge(name, "synthesizer")
g.add_edge("synthesizer", END)
parallel_graph = g.compile()

out = parallel_graph.invoke({"topic": "Oye Cómo Va — Santana", "analyses": []})
print(f"Recolectados {len(out[\'analyses\'])} análisis en paralelo.")
print("Síntesis:", out["summary"][:80], "...")
'''
)

# =============================================================================
# 6. PATTERN 4 — EVALUATOR-OPTIMIZER
# =============================================================================

md(
    """---

## Patrón 4 — **Evaluator-Optimizer** (loop con feedback)

> *Generar → evaluar → si falla, mejorar con feedback → repetir.*

**Cuándo usarlo:** cuando un único intento del LLM no garantiza calidad
y tenés un criterio **automatizable** para juzgarlo (ej.: "no debe
contener spoilers"). Se itera hasta aprobar o hasta `max_retries`.

### Arquitectura — sin LangGraph

```
              user_input
                  │
                  ▼
            ┌──────────┐  ← feedback
            │  Writer  │ ─────────┐
            └──────────┘          │
                  │               │
                  ▼               │
            ┌──────────┐          │
            │ Evaluator│          │
            └──────────┘          │
                  │               │
        ┌─────────┴────┐          │
       APROBADO    RECHAZADO ─────┘  (hasta max_retries)
        │
        ▼
      output
```

En el paquete eso es un bucle `for` con corte cuando el evaluador
devuelve `VEREDICTO: APROBADO`.
"""
)

code(
    '''# --- Sin LangGraph: EvaluatorOptimizerWorkflow ---
# Distinguimos writer / evaluator por el system prompt y simulamos
# un primer intento con spoilers que el evaluador rechaza.

attempts = {"count": 0}
def writer_then_evaluator(messages: list[BaseMessage]) -> str:
    system = messages[0].content.lower()
    if "crítico literario" in system:               # ← writer
        attempts["count"] += 1
        if attempts["count"] == 1:
            return "La reseña revela el final: Rorschach muere al cierre."   # spoiler
        return "Reseña limpia: foco en tema, contexto, autor y estilo visual."
    if "editor de spoilers" in system:              # ← evaluator
        text = messages[-1].content.lower()
        if "rorschach muere" in text or "al cierre" in text:
            return "VEREDICTO: RECHAZADO\\nFEEDBACK: contiene spoilers explícitos del final."
        return "VEREDICTO: APROBADO"
    return "VEREDICTO: APROBADO"

eval_llm = ScriptedLLM(responder=writer_then_evaluator)
eo = EvaluatorOptimizerWorkflow(llm=eval_llm, max_retries=3)
res = eo.run("Reseña de Watchmen sin spoilers")
print(f"Aprobado: {res.metadata[\'approved\']}  Intentos: {res.metadata[\'attempts\']}")
print("Final:", res.output)
'''
)

md(
    """### Arquitectura — con LangGraph

La gracia de LangGraph en este patrón es la **arista cíclica
condicional**: `evaluator → writer` si fue rechazado, `evaluator → END`
si aprobó. El bucle se vuelve declarativo.

```
   START ──▶ writer ──▶ evaluator
                          │
                ┌─────────┴─────────┐
              RECHAZADO          APROBADO
                │                   │
                ▲                   ▼
                └─── (writer)      END
```
"""
)

code(
    '''class EvalState(TypedDict):
    topic: str
    draft: str
    feedback: str
    attempts: int
    approved: bool


def writer_node_eo(state: EvalState) -> dict:
    prompt = f"Cómic: {state[\'topic\']}\\nFeedback previo: {state.get(\'feedback\') or \'(ninguno)\'}"
    out = eval_llm.chat(
        [SystemMessage(content="Eres un crítico literario que escribe reseñas sin spoilers."),
         UserMessage(content=prompt)]
    ).content
    return {"draft": out, "attempts": state.get("attempts", 0) + 1}


def evaluator_node(state: EvalState) -> dict:
    verdict = eval_llm.chat(
        [SystemMessage(content="Eres un editor de spoilers. Veredicto APROBADO/RECHAZADO."),
         UserMessage(content=state["draft"])]
    ).content
    approved = "APROBADO" in verdict.split("\\n", 1)[0].upper()
    feedback = "" if approved else verdict.split("FEEDBACK:", 1)[-1].strip()
    return {"approved": approved, "feedback": feedback}


def should_retry(state: EvalState) -> Literal["writer", "end"]:
    if state["approved"]:
        return "end"
    if state["attempts"] >= 3:
        return "end"
    return "writer"


b = StateGraph(EvalState)
b.add_node("writer", writer_node_eo)
b.add_node("evaluator", evaluator_node)
b.add_edge(START, "writer")
b.add_edge("writer", "evaluator")
b.add_conditional_edges("evaluator", should_retry, {"writer": "writer", "end": END})
eo_graph = b.compile()

attempts["count"] = 0  # reset del contador del responder
out = eo_graph.invoke({"topic": "Watchmen sin spoilers",
                       "draft": "", "feedback": "", "attempts": 0, "approved": False})
print(f"Aprobado: {out[\'approved\']}  Intentos: {out[\'attempts\']}")
print("Final  :", out["draft"])
'''
)

# =============================================================================
# 7. PATTERN 5 — ORCHESTRATOR
# =============================================================================

md(
    """---

## Patrón 5 — **Orchestrator + Workers**

> *Un planner descompone la tarea → workers paralelos la ejecutan → un sintetizador agrega.*

**Cuándo usarlo:** cuando la tarea no se puede pre-definir como un
pipeline fijo. El planner *decide en runtime* qué sub-tareas hay que
hacer. Es la diferencia con "parallel" (donde los analistas son fijos):
acá la cantidad y tipo de workers depende de la entrada.

### Arquitectura — sin LangGraph

```
                user_input
                    │
                    ▼
              ┌───────────┐
              │ Planner   │   (devuelve JSON con N sub-tareas)
              └───────────┘
                    │
        ┌───────────┼───────────┐   (ThreadPoolExecutor)
        ▼           ▼           ▼
    worker_1    worker_2    worker_N
        │           │           │
        └─────┬─────┴────┬──────┘
              ▼          ▼
          ┌─────────────────┐
          │  Synthesizer    │
          └─────────────────┘
                  │
                  ▼
               output
```

Robustez: si el planner devuelve JSON inválido, el paquete cae en
`_FALLBACK_TASKS` (datos / contexto / musical / legado) para que la
ejecución no muera.
"""
)

code(
    '''# --- Sin LangGraph: OrchestratorWorkflow ---
# Dispatch por system prompt:
#  * "editor jefe"        → planner JSON
#  * "fact-checker"       → worker datos
#  * "musicólogo"         → worker musical
#  * "historiador de la música popular" → worker legado
#  * "editor de cierre"   → synthesizer

def orch_responder(messages: list[BaseMessage]) -> str:
    system = messages[0].content.lower()
    if "editor jefe" in system:
        return (\'\'\'{"analysis": "cobertura editorial multi-ángulo",
                    "tasks": [
                      {"type": "datos",   "description": "ficha técnica"},
                      {"type": "musical", "description": "análisis musical"},
                      {"type": "legado",  "description": "impacto cultural"}
                    ]}\'\'\')
    if "fact-checker" in system:
        return "- Artista: Santana\\n- Álbum: Abraxas\\n- Año: 1970\\n- Sello: Columbia"
    if "musicólogo" in system:
        return "Tonalidad Am, montuno cubano, guitarra sobre clave 2-3."
    if "historiador de la música popular" in system:
        return "Influencia decisiva en el latin rock posterior."
    if "editor de cierre" in system:
        return "Ensayo final (~400 palabras) que entrelaza datos, análisis musical y legado cultural."
    return "(worker genérico)"

orch_llm = ScriptedLLM(responder=orch_responder)
orch = OrchestratorWorkflow(llm=orch_llm)
res = orch.run("Oye Cómo Va — Santana")
print("Sub-tareas planeadas:", res.metadata["sub_tasks"])
print("\\nEnsayo final:")
print(res.output)
'''
)

md(
    """### Arquitectura — con LangGraph (`Send` API)

LangGraph tiene un mecanismo específico para *fan-out dinámico*: el
objeto [`Send`](https://langchain-ai.github.io/langgraph/concepts/low_level/#send).
Una arista condicional puede devolver `[Send("worker", payload1), Send("worker", payload2), ...]`
y el grafo dispara *N* invocaciones del mismo nodo, una por cada `Send`.

Eso permite reproducir el patrón orquestador sin saber de antemano
cuántos workers se van a lanzar.

```
              START
                │
                ▼
            ┌────────┐
            │ planner│
            └────────┘
                │  (Send xN, dinámico)
                ▼
            ┌────────┐
            │ worker │ × N (paralelo)
            └────────┘
                │
                ▼
          synthesizer
                │
                ▼
               END
```
"""
)

code(
    '''import json
import re

from langgraph.constants import Send


class OrchState(TypedDict):
    topic: str
    tasks: list[dict]
    partials: Annotated[list[str], add]
    final: str


def planner_node(state: OrchState) -> dict:
    raw = orch_llm.chat(
        [SystemMessage(content="Eres un editor jefe. Devolvé JSON con tasks."),
         UserMessage(content=state["topic"])]
    ).content
    m = re.search(r"\\{.*\\}", raw, re.DOTALL)
    payload = json.loads(m.group(0)) if m else {"tasks": []}
    return {"tasks": payload.get("tasks", [])}


def worker_node(state: dict) -> dict:
    # state acá es el payload de cada Send, no el estado global
    persona = {"datos":   "Eres un fact-checker musical.",
               "musical": "Eres un musicólogo.",
               "legado":  "Eres un historiador de la música popular."}.get(
                  state["type"], "Eres un asistente cultural.")
    out = orch_llm.chat(
        [SystemMessage(content=persona),
         UserMessage(content=f"Tema: {state[\'topic\']}\\nSub-tarea: {state[\'description\']}")]
    ).content
    return {"partials": [f"[{state[\'type\']}] {out}"]}


def synth_node(state: OrchState) -> dict:
    body = "\\n".join(state["partials"])
    out = orch_llm.chat(
        [SystemMessage(content="Eres un editor de cierre. Sintetizá los aportes."),
         UserMessage(content=f"Tema: {state[\'topic\']}\\nAportes:\\n{body}")]
    ).content
    return {"final": out}


def fan_out(state: OrchState) -> list:
    return [Send("worker", {**task, "topic": state["topic"]}) for task in state["tasks"]]


b = StateGraph(OrchState)
b.add_node("planner", planner_node)
b.add_node("worker", worker_node)
b.add_node("synth", synth_node)
b.add_edge(START, "planner")
b.add_conditional_edges("planner", fan_out, ["worker"])
b.add_edge("worker", "synth")
b.add_edge("synth", END)
orch_graph = b.compile()

out = orch_graph.invoke({"topic": "Oye Cómo Va — Santana",
                         "tasks": [], "partials": [], "final": ""})
print("Sub-tareas:", [t["type"] for t in out["tasks"]])
print(f"Workers que respondieron: {len(out[\'partials\'])}")
print("\\nEnsayo final:")
print(out["final"])
'''
)

# =============================================================================
# 8. Cierre
# =============================================================================

md(
    """---

## Comparativa final — ¿cuándo conviene cada enfoque?

| Aspecto | Sin LangGraph (Python plano) | Con LangGraph |
|---|---|---|
| **Curva de aprendizaje** | mínima | conceptos nuevos (StateGraph, reducers, Send) |
| **Lectura del control de flujo** | implícita (orden de las líneas) | explícita (grafo serializable) |
| **Bucles y feedback** | `for` con `break` | aristas condicionales cíclicas |
| **Paralelismo** | `ThreadPoolExecutor` manual | nodos sin dependencias corren en paralelo |
| **Fan-out dinámico** | construcción de futures en runtime | `Send` API |
| **Observabilidad** | logs ad-hoc, `WorkflowResult.steps` | tracing automático + integraciones con LangSmith |
| **Checkpointing / resume** | hay que armarlo | `MemorySaver` / `SqliteSaver` built-in |
| **Streaming token-a-token** | manual | nativo (`graph.stream(...)`) |
| **Testabilidad unitaria** | trivial (es Python plano) | requiere mockear el `Runnable` |
| **Costo en líneas** | menos boilerplate | más boilerplate, más estructura |

### Heurística práctica

* **Empezá sin LangGraph** mientras la topología sea fija y simple
  (chaining, parallel con fan-out estático). Es código que cualquier
  developer Python lee sin documentación.
* **Migrá a LangGraph** cuando aparezcan **ciclos con condiciones**
  (evaluator-optimizer), **fan-out dinámico** (orchestrator) o cuando
  empieces a necesitar **persistir / reanudar / streamear** ejecuciones.
* **El `Protocol` `LLMClient`** del paquete es la mejor abstracción que
  vas a tener: sobrevive a la decisión de framework, porque tanto el
  workflow artesanal como cualquier nodo de LangGraph se construyen
  encima de ella.

### Para profundizar

* Código fuente: `Clase4/src/clase4/workflows/`
* Tests (15, ~0.2 s, sin red): `Clase4/tests/`
* Scripts ejecutables uno por patrón: `Clase4/scripts/exercise_0[1-5]_*.py`
* Docs LangGraph: <https://langchain-ai.github.io/langgraph/>
"""
)

# =============================================================================
# Serializar
# =============================================================================

notebook = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

# nbformat 4.5 requiere id por celda
import uuid

for c in notebook["cells"]:
    c["id"] = uuid.uuid4().hex[:8]

out_path = Path(__file__).parent / "02_prompt_strategies_langgraph.ipynb"
out_path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
print(f"Escrito: {out_path}  ({len(CELLS)} celdas)")
