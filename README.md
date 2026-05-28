# Clase 3: RAG — Repaso del material de la clase

Este directorio contiene el código y los recursos para la Clase 3, dedicada a
**RAG (Retrieval-Augmented Generation)**.

## 🎯 Objetivos del módulo

Al finalizar esta clase podrás:

- Implementar RAG básico con recuperación por similitud de embeddings
- Usar la API de OpenAI para generar embeddings y texto
- Evaluar el rendimiento de un sistema RAG con métricas estándar
- Construir un router semántico basado en centroides y prefiltrado
- Diagnosticar y mejorar la calidad del chunking y la selección de contexto
- Visualizar el espacio de embeddings para auditoría y debugging

## 🛠️ Requisitos

- Python 3.13+
- Entorno virtual con uv:
  ```bash
  uv sync
  ```
- Clave de OpenAI en `.env`:
  ```env
  OPENAI_API_KEY="sk-..."
  OPENAI_BASE_URL="https://api.openai.com/v1/"
  OPENAI_MODEL="gpt-4o-mini"
  ```

## 📁 Estructura del proyecto

```
Clase3/
├── .env                  # Tus claves de OpenAI (no subir a Git)
├── src/
│   ├── clase3/
│   │   ├── __init__.py
│   │   ├── config.py       # Configuración y constantes
│   │   ├── adapters/       # Adaptadores a sistemas externos
│   │   │   ├── __init__.py
│   │   │   ├── openai_embedder.py
│   │   │   ├── openai_llm.py
│   │   │   ├── markdown_loader.py
│   │   ├── services/       # Lógica de negocio
│   │   │   ├── __init__.py
│   │   │   ├── chunker.py
│   │   │   ├── retriever.py
│   │   │   ├── generator.py
│   │   │   ├── router.py
│   │   ├── utils/          # Utilidades compartidas
│   │   │   ├── __init__.py
│   │   │   ├── metrics.py
│   │   │   ├── prompt_utils.py
├── data/
│   ├── comics/             # Comics para el ejercicio 1
│   └── music/              # Música para ejercicios 2, 3, 5 y notebook
├── notebooks/
│   ├── 01_embeddings_visual_tour.ipynb
├── scripts/
│   ├── exercise_01_basic_rag.py
│   ├── exercise_02_basic_rag.py
│   ├── exercise_03_musical_rag.py
│   ├── exercise_04_router.py
│   ├── exercise_05_evaluate.py
├── .gitignore
├── pyproject.toml
├── README.md
└── uv.lock
```

## 🏃 Ejecución

Desde la raíz del repositorio:

```bash
# 1) RAG de comics (preguntas y respuestas cortas)
uv run python scripts/exercise_01_basic_rag.py

# 2) RAG musical básico
uv run python scripts/exercise_02_basic_rag.py

# 3) RAG musical (system prompt: crítico musical latinoamericano)
#    Imprime también una tabla de retrieval por pregunta (título · sección · sim).
uv run python scripts/exercise_03_musical_rag.py

# 4) Router RAG: el LLM decide entre 'comics' y 'musica'
uv run python scripts/exercise_04_router.py

# 5) Evaluar RAG musical con precisión/recall/f1 y recuperación
uv run python scripts/exercise_05_evaluate.py
```

### 5.1 Notebook — *Geometría del significado*

`notebooks/01_embeddings_visual_tour.ipynb` es un tour visual estilo
*Jay Alammar* sobre el espacio de embeddings, construido **sobre el mismo corpus
musical** que usa el RAG. Cada visualización está atada a un caso de uso de
RAG real:

| Sección | Pregunta que responde | Conexión con RAG |
|---------|----------------------|-------------------|
| Anatomía del vector | ¿Norma 1? ¿Distribución? | Validar la promesa del proveedor de embeddings |
| Heatmap de similitud | ¿El modelo agrupa por estilo? | Diagnóstico de overlap entre corpora |
| PCA + t-SNE | ¿Cómo se ve el "mapa" del corpus? | Auditoría del espacio antes de tunear prompts |
| k-NN a mano | ¿Qué hace `Retriever.retrieve`? | Simulación del paso "R" del pipeline |
| Centroides por artista | ¿Qué tan separados están los estilos? | Base de routers semánticos sin LLM |
| Aritmética de vectores | ¿Existen ejes culturales? | Soft-routing sin metadata |
| Chunking visual | ¿Los chunks orbitan al documento? | Diagnóstico del chunker |

Para abrirlo:

```bash
uv run jupyter notebook notebooks/01_embeddings_visual_tour.ipynb
```

Los embeddings se cachean en `.cache/music_embeddings.pkl` (hash por
contenido), así que re-ejecutar la notebook es gratis.

### Calidad de código

```bash
uv run ruff check
uv run ruff format
uv run mypy
```

## 📚 Recursos de la clase

- [Apuntes en Notion](https://notion.so/...)
- [PPT](data/material/...) (si aplica)
- Documentación oficial de OpenAI: embeddings, Chat Completions, modelos
- Biblioteca Tiktoken para tokenización

¡Cualquier duda, consulta los materiales o pregunta por los canales del curso!
