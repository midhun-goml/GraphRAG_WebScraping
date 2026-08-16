"""
models.py — Pipeline-Wide Pydantic Models
===========================================

Single source of truth for every data structure that crosses a module
boundary in the Graph RAG pipeline.

Models
------
- **DocumentMetadata** — Source document identity and provenance.
- **TextChunk**        — Token-aware text chunk with character offsets.
- **Entity**           — Resolved knowledge-graph entity.
- **Relationship**     — Directed, weighted edge between two entities.
- **ChunkExtraction**  — Per-chunk LLM extraction output.
- **GraphData**        — Global pipeline output aggregating all artefacts.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator #type: ignore


# ======================================================================
# Document-level metadata
# ======================================================================
class DocumentMetadata(BaseModel):
   

    model_config = ConfigDict(populate_by_name=True)

    source_file: str = Field(..., min_length=1)
    title: str | None = Field(default=None)
    author: str | None = Field(default=None)
    page_count: int = Field(default=0, ge=0)
    creation_date: str | None = Field(default=None)
    file_hash: str = Field(..., min_length=1)
    word_count: int = Field(default=0, ge=0)
    language: str = Field(default="en")
    markdown_path: str = Field(default="")


# ======================================================================
# Chunk-level model
# ======================================================================
class TextChunk(BaseModel):
    

    model_config = ConfigDict(populate_by_name=True)

    chunk_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    token_count: int = Field(..., gt=0)
    chunk_index: int = Field(..., ge=0)
    start_char: int = Field(..., ge=0)
    end_char: int = Field(..., ge=0)
    heading_hierarchy: list[str] = Field(default_factory=list)
    metadata: DocumentMetadata

    @field_validator("text")
    @classmethod
    def _text_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Chunk text must be non-empty.")
        return v


# ======================================================================
# Entity model
# ======================================================================
class Entity(BaseModel):
    

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(..., min_length=1)
    type: str = Field(..., min_length=1)
    description: str = Field(default="")
    aliases: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    embedding: list[float] | None = Field(default=None)

    @field_validator("name")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        return v.strip()

    @field_validator("type")
    @classmethod
    def _strip_type(cls, v: str) -> str:
        return v.strip()


# ======================================================================
# Relationship model
# ======================================================================
class Relationship(BaseModel):
    

    model_config = ConfigDict(populate_by_name=True)

    source_entity: str = Field(..., min_length=1)
    target_entity: str = Field(..., min_length=1)
    relation_type: str = Field(..., min_length=1)
    description: str = Field(default="")
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    source_chunk_ids: list[str] = Field(default_factory=list)

    @field_validator("relation_type")
    @classmethod
    def _normalize_relation(cls, v: str) -> str:
        return v.strip().upper().replace(" ", "_")

    @field_validator("weight")
    @classmethod
    def _clamp_weight(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


# ======================================================================
# Per-chunk extraction result
# ======================================================================
class ChunkExtraction(BaseModel):
    """LLM extraction output for a single chunk.

    Attributes
    ----------
    chunk_id : str
        The deterministic chunk identifier.
    entities : list[Entity]
        Entities found in this chunk.
    relationships : list[Relationship]
        Relationships found in this chunk.
    summary : str
        LLM-generated chunk summary.
    """

    model_config = ConfigDict(populate_by_name=True)

    chunk_id: str = Field(..., min_length=1)
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    summary: str = Field(default="")


# ======================================================================
# Chunk-only serialisation wrapper
# ======================================================================
class ChunksOutput(BaseModel):
    """Wrapper for serialising validated chunks to ``chunks.json``.

    Attributes
    ----------
    document_metadata : DocumentMetadata
        The document-level metadata.
    total_chunks : int
        Number of chunks in the list.
    chunks : list[TextChunk]
        All validated chunks.
    """

    model_config = ConfigDict(populate_by_name=True)

    document_metadata: DocumentMetadata
    total_chunks: int = Field(default=0, ge=0)
    chunks: list[TextChunk] = Field(default_factory=list)


# ======================================================================
# Global pipeline output
# ======================================================================
class GraphData(BaseModel):
    """Aggregated pipeline output containing all resolved artefacts.

    Attributes
    ----------
    entities : list[Entity]
        Globally resolved and deduplicated entities.
    relationships : list[Relationship]
        Globally resolved relationships.
    chunks : list[TextChunk]
        All text chunks from the document.
    extractions : list[ChunkExtraction]
        Per-chunk extraction results (before resolution).
    metadata : DocumentMetadata
        Source document metadata.
    """

    model_config = ConfigDict(populate_by_name=True)

    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    chunks: list[TextChunk] = Field(default_factory=list)
    extractions: list[ChunkExtraction] = Field(default_factory=list)
    metadata: DocumentMetadata
