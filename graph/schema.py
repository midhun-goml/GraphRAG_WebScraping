"""
schema.py — Graph Schema Definitions
======================================

Canonical, frozen definitions of node labels, entity types, and
structural relationship labels used in the Knowledge Graph.

Dynamic LLM-generated relationship types (e.g. ``FOUNDED_BY``,
``WORKS_AT``) are **not** validated against a whitelist — the LLM is
trusted to produce semantically accurate labels.

Only structural relationship constants (``EXTRACTED_FROM``,
``NEXT_CHUNK``) are defined here.
"""

from __future__ import annotations

# ======================================================================
# Node Labels
# ======================================================================
CHUNK_NODE: str = "Chunk"
"""Neo4j label for document-chunk nodes (evidence layer)."""

ENTITY_NODE: str = "Entity"
"""Neo4j label for extracted entity nodes (knowledge layer)."""

DOCUMENT_NODE: str = "Document"
"""Neo4j label for source-document nodes."""

# ======================================================================
# Entity Types (reference vocabulary for prompts)
# ======================================================================
ENTITY_TYPES: frozenset[str] = frozenset({
    "PERSON",
    "ORGANIZATION",
    "CONCEPT",
    "TECHNOLOGY",
    "LOCATION",
    "EVENT",
    "DATE",
    "METRIC",
    "DOCUMENT",
    "LAW",
    "PRODUCT",
})

# ======================================================================
# Structural Relationship Labels
# ======================================================================
EXTRACTED_FROM: str = "EXTRACTED_FROM"
"""(:Entity)-[:EXTRACTED_FROM]->(:Chunk) — entity provenance link."""

NEXT_CHUNK: str = "NEXT_CHUNK"
"""(:Chunk)-[:NEXT_CHUNK]->(:Chunk) — sequential chunk ordering."""

PART_OF: str = "PART_OF"
"""(:Chunk)-[:PART_OF]->(:Document) — chunk to document provenance."""


# ======================================================================
# Validation helpers
# ======================================================================
def is_valid_entity_type(entity_type: str) -> bool:
    """Check whether *entity_type* is in the reference vocabulary.

    Parameters
    ----------
    entity_type : str

    Returns
    -------
    bool
    """
    return entity_type.upper() in ENTITY_TYPES


def sanitize_relationship_label(label: str) -> str:
    """Sanitize a dynamic relationship label for safe Cypher injection.

    Removes characters that could break Cypher syntax and ensures the
    label is a valid Neo4j relationship type.

    Parameters
    ----------
    label : str

    Returns
    -------
    str
    """
    import re
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", label.strip().upper())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "RELATED_TO"