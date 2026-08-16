"""
graph — Knowledge Graph Construction
======================================

Schema definitions, Neo4j connection management, and graph builder
for converting extracted knowledge into a Neo4j Knowledge Graph
with vector-indexed entity embeddings.
"""

from graph.builder import GraphBuilder
from graph.neo4j_client import Neo4jClient
from graph.schema import (
    CHUNK_NODE,
    DOCUMENT_NODE,
    ENTITY_NODE,
    ENTITY_TYPES,
    EXTRACTED_FROM,
    NEXT_CHUNK,
    PART_OF,
    is_valid_entity_type,
    sanitize_relationship_label,
)

__all__ = [
    "Neo4jClient",
    "GraphBuilder",
    "ENTITY_NODE",
    "CHUNK_NODE",
    "DOCUMENT_NODE",
    "ENTITY_TYPES",
    "EXTRACTED_FROM",
    "NEXT_CHUNK",
    "PART_OF",
    "is_valid_entity_type",
    "sanitize_relationship_label",
]
