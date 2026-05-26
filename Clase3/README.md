# Clase 3 — RAG (Retrieval-Augmented Generation) con arquitectura SOLID

> Módulo 3 del programa **HENRY · AI Engineering**.
> Construir un sistema RAG industrial, modular y testeable sobre un corpus de
> **cómics canónicos** y **música latinoamericana** (Santana, Celia Cruz, Los
> Fabulosos Cadillacs).

---

## 1. ¿Qué es RAG y por qué importa?

Un modelo de lenguaje (LLM) sólo "sabe" lo que vio durante su entrenamiento.
Si le preguntas por información reciente, privada o de nicho — la letra de
*Quimbara*, la trama de *Maus* — el modelo intentará completar la respuesta
con datos plausibles pero inventados. A eso se le llama **alucinación**.

**Retrieval-Augmented Generation** mitiga el problema dándole al modelo un
"manual abierto" antes de responder: una base de conocimiento indexada por
**embeddings** que se consulta en cada turno para inyectar el contexto
relevante en el prompt.

El pipeline canónico tiene tres pasos: **R-A-G**:

```text
                ┌─────────────────────────────────────────┐
   pregunta ─▶  │  R  Retrieve  →  vector DB ↦ top-k     │
                │  A  Augment   →  contexto + pregunta    │
                │  G  Generate  →  LLM ↦ respuesta citada │
                └─────────────────────────────────────────┘
```

- **Retrieve** convierte la pregunta en un vector y busca los chunks más
  cercanos en el vector store.
- **Augment** ensambla un prompt que contiene los fragmentos recuperados y
  reglas explícitas para el LLM.
- **Generate** invoca al LLM, que produce la respuesta final apoyado en el
  contexto.

---

## 2. Arquitectura SOLID del proyecto

Este repositorio aplica los cinco principios SOLID de forma deliberada:

| Principio | Aplicación concreta |
|-----------|---------------------|
| **S** — Single Responsibility | `TextChunker` sólo trocea, `Retriever` sólo recupera, `PromptAugmenter` sólo construye prompts, `AnswerGenerator` sólo llama al LLM. |
| **O** — Open/Closed | Añadir un nuevo cargador (PDF, web…) es escribir una clase que cumpla `DocumentLoader`. No se toca ni el pipeline ni los scripts. |
| **L** — Liskov Substitution | Cualquier `VectorStore` o `LLMClient` es intercambiable: el `RAGPipeline` no distingue Chroma de FAISS, ni OpenAI de un fake en tests. |
| **I** — Interface Segregation | Los `Protocol` (en `ports/`) tienen 1–3 métodos cada uno; nadie depende de capacidades que no usa. |
| **D** — Dependency Inversion | El pipeline depende exclusivamente de los `Protocol` en `ports/`. Los `adapters/` concretos sólo se conocen en `factory.py` (composition root). |

### Estructura de carpetas

```text
Clase3/
├── data/
│   ├── comics/          # 6 documentos .md (Watchmen, V de Vendetta, Maus, …)
│   └── musica/          # 9 canciones (Santana, Celia Cruz, F. Cadillacs)
├── src/clase3/
│   ├── config.py        # variables de entorno, paths, defaults
│   ├── domain/          # entidades inmutables (Document, Chunk, mensajes)
│   ├── ports/           # Protocols: DocumentLoader, EmbeddingProvider, …
│   ├── adapters/        # implementaciones concretas (OpenAI, Chroma, MD)
│   ├── services/        # lógica de negocio: chunker, retriever, pipeline…
│   └── factory.py       # composition root: build_rag_bundle(...)
├── scripts/             # ejercicios end-to-end
│   ├── exercise_01_ingest.py
│   ├── exercise_02_basic_rag.py
│   ├── exercise_03_musical_rag.py
│   ├── exercise_04_router_rag.py
│   └── exercise_05_evaluate.py
├── tests/               # 15 unit + integration tests (sin red)
└── pyproject.toml       # uv + ruff + pytest
```

### Diagrama de dependencias

```text
   scripts/  ─────────────► factory.build_rag_bundle()
                                 │
                                 ▼
                          ┌──────────────────┐
                          │   RAGPipeline    │  (services)
                          └──────────────────┘
                                 │ depende sólo de Protocols
                                 ▼
        ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
        │ Retriever    │  │ Augmenter    │  │ Generator    │
        └──────────────┘  └──────────────┘  └──────────────┘
                 │                  │                │
                 ▼                  ▼                ▼
         VectorStore (port)   <plantilla>     LLMClient (port)
                 ▲                                    ▲
   adapters: ChromaVectorStore              OpenAIChatClient
             FAISSVectorStore  ←—— extensible ——→   AnthropicClient
```

---

## 3. Dataset

### Cómics (`data/comics/`)

1. **Watchmen** — Alan Moore (1986)
2. **V de Vendetta** — Alan Moore (1982-1989)
3. **Maus** — Art Spiegelman (1980-1991)
4. **Persépolis** — Marjane Satrapi (2000-2003)
5. **The Sandman** — Neil Gaiman (1989-1996)
6. **The Dark Knight Returns** — Frank Miller (1986)

### Música latinoamericana (`data/musica/`)

- **Carlos Santana**: *Oye Cómo Va*, *Black Magic Woman*, *Smooth (ft. Rob Thomas)*
- **Celia Cruz**: *La Vida Es un Carnaval*, *La Negra Tiene Tumbao*, *Quimbara*
- **Los Fabulosos Cadillacs**: *Matador*, *Mal Bicho*, *V Centenario*

Cada documento Markdown incluye **contexto histórico, análisis musical,
personal de grabación, anécdotas, recepción crítica e importancia cultural**.
Los chunks se construyen respetando los encabezados H2 del Markdown.

---

## 4. Instalación

Requiere **Python 3.11+**, [`uv`](https://github.com/astral-sh/uv) y un
`.env` con `OPENAI_API_KEY` en la raíz del repositorio (`HENRY_EMBEDDINGS_RAGS/.env`).

```bash
cd Clase3
uv sync                # instala dependencias y crea .venv
```

Eso es todo. `uv` se encarga del entorno virtual, lockfile y resolución
determinista.

---

## 5. Ejecución de los ejercicios

```bash
# 1) Ingesta del corpus (escribe en ./.chroma)
uv run python scripts/exercise_01_ingest.py --reset

# 2) RAG básico sobre cómics
uv run python scripts/exercise_02_basic_rag.py

# 3) RAG musical (system prompt: crítico musical latinoamericano)
uv run python scripts/exercise_03_musical_rag.py

# 4) Router RAG: el LLM decide entre 'comics' y 'musica'
uv run python scripts/exercise_04_router_rag.py

# 5) Evaluación cuantitativa (hit@k, MRR, similitud media)
uv run python scripts/exercise_05_evaluate.py
```

### Calidad de código

```bash
uv run ruff check src scripts tests   # linting (PEP-8, pyupgrade, isort, …)
uv run ruff format src scripts tests  # formateo automático
uv run pytest tests/ -v               # 15 tests (chunker, loader, augmenter, pipeline)
```

---

## 6. Snippet de uso programático

```python
from clase3.config import COMICS_DIR
from clase3.factory import build_rag_bundle

# 1. Composición — el único lugar donde se eligen los adapters concretos.
bundle = build_rag_bundle(
    collection_name="comics",
    data_directory=COMICS_DIR,
    extra_metadata={"corpus": "comics"},
    top_k=4,
)

# 2. Indexación idempotente — sólo escribe si la colección está vacía.
if bundle.store.count() == 0:
    bundle.indexer.index()

# 3. Pregunta a través del pipeline.
result = bundle.pipeline.ask(
    "¿Qué simboliza la máscara de Guy Fawkes en V de Vendetta?"
)
print(result.answer)

# 4. Las fuentes recuperadas siguen disponibles para auditar.
for item in result.retrieved:
    print(item.chunk.metadata["title"], "→ similitud≈", round(item.similarity, 2))
```

Para sustituir el LLM o el vector store, basta con escribir un adapter que
cumpla el `Protocol` correspondiente y pasarlo al constructor del pipeline.
El resto del código no necesita cambios.

---

## 7. Anatomía del pipeline (código relevante)

```python
# src/clase3/services/rag_pipeline.py
class RAGPipeline:
    def ask(self, question: str, *, where: dict | None = None) -> RAGAnswer:
        retrieved = self._retriever.retrieve(question, where=where)         # R
        messages  = self._augmenter.build_messages(question, retrieved)     # A
        ai_msg    = self._generator.generate(messages)                      # G
        return RAGAnswer(
            question=question, answer=ai_msg.content,
            retrieved=retrieved, messages=[*messages, ai_msg],
        )
```

### Chunking semántico por secciones Markdown

```python
# src/clase3/services/chunker.py — extracto
sections = self._split_into_sections(document.content)
for header, body in sections:
    if len(body) <= self.chunk_size:
        chunks.append(self._make_chunk(content=self._prefix(header, body), ...))
    else:
        chunks.extend(self._sliding_window(body, document, header=header, ...))
```

Cada sección `##` del Markdown se convierte en un chunk independiente con su
título propagado al texto. Esto reduce las respuestas espurias de "no lo sé"
porque el chunk recuperado conserva sentido autocontenido.

---

## 8. Resultados de la evaluación

Sobre 8 preguntas etiquetadas (`scripts/exercise_05_evaluate.py`):

| Métrica         | Valor   |
|-----------------|---------|
| `hit@k`         | **100 %** (8/8)  |
| **MRR**         | **1.000**        |
| similitud media | 0.567            |

Los 15 tests unitarios + integración pasan sin red.

---

## 9. Ciclos de revisión aplicados

1. **Ciclo 1**: implementación base. Métricas OK pero el LLM devolvía
   "no lo sé" en 3 preguntas porque los chunks recuperados perdían el
   contexto local.
2. **Ciclo 2**: chunking semántico por encabezados Markdown, prompt menos
   conservador, exposición pública de `retrieve_only` y `retriever` en el
   pipeline (deja de violar encapsulamiento), `DEFAULT_TOP_K` de 3 → 4. Las
   tres preguntas problemáticas se resuelven con respuestas detalladas y
   citadas.
3. **Ciclo 3**: tests de integración con fakes (cumplen el `Protocol` sin
   pegarse a OpenAI ni Chroma), ajustes de tipado, validación holística
   (`ruff` + `pytest` + 4 ejercicios end-to-end con OpenAI real).

---

## 10. Cómo extender el proyecto

| Quiero…                                  | Hago…                                                                                          |
|------------------------------------------|------------------------------------------------------------------------------------------------|
| añadir un cargador PDF                   | crear `PDFLoader` que devuelva `list[Document]`, usarlo en `factory.build_rag_bundle`.        |
| cambiar a embeddings locales             | implementar `EmbeddingProvider` con `sentence-transformers`, inyectarlo en `ChromaVectorStore`.|
| reemplazar Chroma por FAISS              | escribir `FAISSVectorStore` que cumpla el `Protocol`; cero cambios en `RAGPipeline`.          |
| cambiar OpenAI por Anthropic             | implementar `AnthropicChatClient` que cumpla `LLMClient`.                                     |
| agregar reranking                        | un nuevo servicio entre `Retriever` y `PromptAugmenter` (composición, no herencia).           |
