"""Adapters: implementaciones concretas de los Ports."""

from clase3.adapters.chroma_store import ChromaVectorStore
from clase3.adapters.google_embedder import GoogleGenAIEmbedder
from clase3.adapters.markdown_loader import MarkdownDirectoryLoader
from clase3.adapters.openai_embedder import OpenAIEmbedder
from clase3.adapters.openai_llm import OpenAIChatClient

__all__ = [
    "ChromaVectorStore",
    "GoogleGenAIEmbedder",
    "MarkdownDirectoryLoader",
    "OpenAIChatClient",
    "OpenAIEmbedder",
]
