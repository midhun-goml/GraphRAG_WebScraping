"""
exceptions.py — Shared Exception Hierarchy
============================================

Typed exception classes for the entire Graph RAG pipeline.
Every module raises exceptions from this hierarchy instead of bare
``Exception`` or ``RuntimeError``.

Hierarchy
---------
::

    GraphRAGError
    ├── DocumentLoadError
    ├── ChunkingError
    ├── LLMGenerationError
    │   └── LLMJsonParseError
    ├── EntityResolutionError
    ├── GraphConstructionError
    └── RetrievalError
"""


class GraphRAGError(Exception):
    """Base exception for the Graph RAG pipeline."""


class DocumentLoadError(GraphRAGError):
    """Failed to load or parse a document."""


class ChunkingError(GraphRAGError):
    """Failed during text chunking."""


class LLMGenerationError(GraphRAGError):
    """LLM call failed or returned unusable output."""


class LLMJsonParseError(LLMGenerationError):
    """LLM returned non-parseable JSON."""


class EntityResolutionError(GraphRAGError):
    """Failed during entity merging or deduplication."""


class GraphConstructionError(GraphRAGError):
    """Failed to build or query the Neo4j graph."""


class RetrievalError(GraphRAGError):
    """Failed during the retrieval or search pipeline."""
