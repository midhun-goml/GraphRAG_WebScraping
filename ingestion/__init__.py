"""
ingestion — Document Ingestion Pipeline
========================================

Production-grade pipeline for converting raw PDF documents into
validated, token-aware text chunks ready for LLM extraction and
Knowledge Graph construction.

Pipeline stages:
    1. DocumentLoader       — PDF → markdown text + metadata (Docling).
    2. TextCleaner          — Normalise and clean raw text.
    3. MetadataExtractor    — Enrich document metadata.
    4. MarkdownChunker      — Heading-aware, token-counted chunking.
    5. ChunkValidator       — Filter bad / boilerplate chunks.
"""

from ingestion.chunker import MarkdownChunker, TokenAwareChunker
from ingestion.cleaner import TextCleaner
from ingestion.loader import DocumentLoader
from ingestion.metadata import MetadataExtractor
from ingestion.models import (
    ChunkExtraction,
    ChunksOutput,
    DocumentMetadata,
    Entity,
    GraphData,
    Relationship,
    TextChunk,
)
from ingestion.validator import ChunkValidator

__all__ = [
    "DocumentLoader",
    "TextCleaner",
    "MetadataExtractor",
    "MarkdownChunker",
    "TokenAwareChunker",
    "ChunkValidator",
    "DocumentMetadata",
    "TextChunk",
    "ChunksOutput",
    "Entity",
    "Relationship",
    "ChunkExtraction",
    "GraphData",
]
