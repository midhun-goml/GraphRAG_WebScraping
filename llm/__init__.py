"""
llm — LLM Client & Embedding Model
=====================================

Groq-compatible LLM client for text/JSON generation and
sentence-transformer embedding model for entity vectors.
"""

from llm.generator import EmbeddingModel, LLMClient

__all__ = ["LLMClient", "EmbeddingModel"]
