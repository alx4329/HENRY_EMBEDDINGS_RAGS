"""Services: la lógica de negocio del pipeline RAG.

Cada servicio tiene UNA responsabilidad (SRP) y depende sólo de Ports, nunca
de Adapters concretos.
"""

from clase3.services.augmenter import PromptAugmenter
from clase3.services.chunker import TextChunker
from clase3.services.generator import AnswerGenerator
from clase3.services.indexer import CorpusIndexer
from clase3.services.rag_pipeline import RAGAnswer, RAGPipeline
from clase3.services.retriever import Retriever

__all__ = [
    "AnswerGenerator",
    "CorpusIndexer",
    "PromptAugmenter",
    "RAGAnswer",
    "RAGPipeline",
    "Retriever",
    "TextChunker",
]
