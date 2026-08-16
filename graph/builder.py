"""
builder.py — Knowledge Graph Builder
======================================

Takes a ``GraphData`` object and populates a Neo4j graph using efficient
batch writes.

Graph Architecture
------------------
- **Chunk nodes**   — evidence layer (text, summary, metadata).
- **Entity nodes**  — knowledge layer (name, type, description, embedding).
- **Dynamic edges** — LLM-generated relationship types between entities.
- **EXTRACTED_FROM** — entity → source chunk provenance.
- **NEXT_CHUNK**     — sequential chunk ordering for context expansion.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from exceptions import GraphConstructionError
from graph.neo4j_client import Neo4jClient
from graph.schema import (
    CHUNK_NODE,
    DOCUMENT_NODE,
    ENTITY_NODE,
    EXTRACTED_FROM,
    NEXT_CHUNK,
    PART_OF,
    sanitize_relationship_label,
)
from ingestion.models import ChunksOutput, GraphData

logger = logging.getLogger(__name__)


class GraphBuilder:

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        clear_existing: bool = False,
    ) -> None:
        self.client = neo4j_client
        self.clear_existing = clear_existing
        logger.info("GraphBuilder initialised (clear=%s).", clear_existing)

    def build_graph(self, graph_data: GraphData) -> None:
        
        if not graph_data.chunks and not graph_data.entities:
            logger.warning("GraphData is empty — nothing to build.")
            return

        embedding_dim = int(os.getenv("EMBEDDING_DIMENSION", "384"))

        # Step 0
        if self.clear_existing:
            logger.info("Clearing existing graph data.")
            self.client.execute_write("MATCH (n) DETACH DELETE n")

        # Step 1
        logger.info("Creating indexes and constraints.")
        self.client.create_indexes(embedding_dimension=embedding_dim)

        # Step 2
        self._create_chunk_nodes(graph_data)

        # Step 3
        self._create_entity_nodes(graph_data)

        # Step 4
        self._create_relationships(graph_data)

        # Step 5
        self._create_extracted_from_links(graph_data)

        # Step 6
        self._create_next_chunk_links(graph_data)

        logger.info(
            "Graph build complete — %d chunks, %d entities, "
            "%d relationships.",
            len(graph_data.chunks),
            len(graph_data.entities),
            len(graph_data.relationships),
        )

    # ------------------------------------------------------------------
    # Step 2: Chunk nodes
    # ------------------------------------------------------------------
    def _create_chunk_nodes(self, graph_data: GraphData) -> None:
        """Batch-create Chunk nodes."""
        # Build a lookup for chunk summaries from extractions
        summary_map: dict[str, str] = {}
        for ext in graph_data.extractions:
            summary_map[ext.chunk_id] = ext.summary

        batch = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "token_count": c.token_count,
                "chunk_index": c.chunk_index,
                "source_file": c.metadata.source_file,
                "summary": summary_map.get(c.chunk_id, ""),
            }
            for c in graph_data.chunks
        ]

        query = f"""
        UNWIND $batch AS row
        MERGE (c:{CHUNK_NODE} {{chunk_id: row.chunk_id}})
        SET c.text = row.text,
            c.token_count = row.token_count,
            c.chunk_index = row.chunk_index,
            c.source_file = row.source_file,
            c.summary = row.summary
        """
        self.client.execute_batch(query, batch)
        logger.info("Created %d Chunk node(s).", len(batch))

    # ------------------------------------------------------------------
    # Step 3: Entity nodes
    # ------------------------------------------------------------------
    def _create_entity_nodes(self, graph_data: GraphData) -> None:
        """Batch-create Entity nodes with embeddings."""
        batch = [
            {
                "name": e.name,
                "type": e.type,
                "description": e.description,
                "aliases": e.aliases,
                "embedding": e.embedding,
            }
            for e in graph_data.entities
        ]

        query = f"""
        UNWIND $batch AS row
        MERGE (e:{ENTITY_NODE} {{name: row.name}})
        SET e.type = row.type,
            e.description = row.description,
            e.aliases = row.aliases,
            e.embedding = row.embedding
        """
        self.client.execute_batch(query, batch)
        logger.info("Created %d Entity node(s).", len(batch))

    # ------------------------------------------------------------------
    # Step 4: Inter-entity relationships (dynamic types)
    # ------------------------------------------------------------------
    def _create_relationships(self, graph_data: GraphData) -> None:
        """Create inter-entity relationships with dynamic types.

        Uses string interpolation with sanitized labels (safe after
        ``sanitize_relationship_label``).  Falls back to ``RELATES_TO``
        with a ``relation_type`` property if the sanitized label is
        empty.
        """
        # Group by relation_type for batch execution per type
        from collections import defaultdict

        groups: dict[str, list[dict]] = defaultdict(list)
        for r in graph_data.relationships:
            label = sanitize_relationship_label(r.relation_type)
            groups[label].append(
                {
                    "source_entity": r.source_entity,
                    "target_entity": r.target_entity,
                    "description": r.description,
                    "weight": r.weight,
                    "source_chunk_ids": r.source_chunk_ids,
                }
            )

        for label, batch in groups.items():
            query = f"""
            UNWIND $batch AS row
            MATCH (s:{ENTITY_NODE} {{name: row.source_entity}})
            MATCH (t:{ENTITY_NODE} {{name: row.target_entity}})
            MERGE (s)-[r:{label}]->(t)
            SET r.description = row.description,
                r.weight = row.weight,
                r.source_chunk_ids = row.source_chunk_ids
            """
            self.client.execute_batch(query, batch)
            logger.info(
                "Created %d '%s' relationship(s).", len(batch), label
            )

    # ------------------------------------------------------------------
    # Step 5: EXTRACTED_FROM links
    # ------------------------------------------------------------------
    def _create_extracted_from_links(self, graph_data: GraphData) -> None:
        """Link entities to their source chunks."""
        batch: list[dict] = []
        for entity in graph_data.entities:
            for chunk_id in entity.source_chunk_ids:
                batch.append(
                    {"entity_name": entity.name, "chunk_id": chunk_id}
                )

        if not batch:
            return

        query = f"""
        UNWIND $batch AS row
        MATCH (e:{ENTITY_NODE} {{name: row.entity_name}})
        MATCH (c:{CHUNK_NODE} {{chunk_id: row.chunk_id}})
        MERGE (e)-[:{EXTRACTED_FROM}]->(c)
        """
        self.client.execute_batch(query, batch)
        logger.info("Created %d EXTRACTED_FROM link(s).", len(batch))

    # ------------------------------------------------------------------
    # Step 6: NEXT_CHUNK sequential links
    # ------------------------------------------------------------------
    def _create_next_chunk_links(self, graph_data: GraphData) -> None:
        """Create sequential chunk links for neighbor expansion."""
        sorted_chunks = sorted(
            graph_data.chunks, key=lambda c: c.chunk_index
        )

        batch: list[dict] = []
        for i in range(len(sorted_chunks) - 1):
            batch.append(
                {
                    "current_chunk_id": sorted_chunks[i].chunk_id,
                    "next_chunk_id": sorted_chunks[i + 1].chunk_id,
                }
            )

        if not batch:
            return

        query = f"""
        UNWIND $batch AS row
        MATCH (c1:{CHUNK_NODE} {{chunk_id: row.current_chunk_id}})
        MATCH (c2:{CHUNK_NODE} {{chunk_id: row.next_chunk_id}})
        MERGE (c1)-[:{NEXT_CHUNK}]->(c2)
        """
        self.client.execute_batch(query, batch)
        logger.info("Created %d NEXT_CHUNK link(s).", len(batch))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(client={self.client!r})"

    # ==================================================================
    # Chunk-only graph builder (new pipeline)
    # ==================================================================
    def build_chunk_graph(self, chunks_output: ChunksOutput) -> None:
        """Build the chunk-layer graph from a ``ChunksOutput``.

        Steps
        -----
        1. Create chunk-layer indexes/constraints.
        2. MERGE the :Document node.
        3. Batch-create :Chunk nodes.
        4. Link chunks to document via :PART_OF.
        5. Create sequential :NEXT_CHUNK relationships.
        6. Log final graph counts.

        Parameters
        ----------
        chunks_output : ChunksOutput
        """
        meta = chunks_output.document_metadata

        if not chunks_output.chunks:
            logger.warning("ChunksOutput is empty — nothing to build.")
            return

        # Step 1 — Create indexes
        logger.info("Creating chunk-layer indexes.")
        self.client.create_chunk_indexes()

        # Step 2 — Create Document node
        logger.info("Creating Document node.")
        doc_query = f"""
        MERGE (d:{DOCUMENT_NODE} {{file_hash: $file_hash}})
        SET d.source_file   = $source_file,
            d.title         = $title,
            d.author        = $author,
            d.page_count    = $page_count,
            d.creation_date = $creation_date,
            d.word_count    = $word_count,
            d.language      = $language,
            d.markdown_path = $markdown_path,
            d.total_chunks  = $total_chunks
        """
        self.client.execute_write(
            doc_query,
            {
                "file_hash": meta.file_hash,
                "source_file": meta.source_file,
                "title": meta.title,
                "author": meta.author,
                "page_count": meta.page_count,
                "creation_date": meta.creation_date,
                "word_count": meta.word_count,
                "language": meta.language,
                "markdown_path": meta.markdown_path,
                "total_chunks": chunks_output.total_chunks,
            },
        )

        # Step 3 — Create Chunk nodes in batch
        logger.info("Creating %d Chunk node(s).", len(chunks_output.chunks))
        chunk_batch = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "token_count": c.token_count,
                "chunk_index": c.chunk_index,
                "start_char": c.start_char,
                "end_char": c.end_char,
                "heading_hierarchy": c.heading_hierarchy,
                "source_file": c.metadata.source_file,
            }
            for c in chunks_output.chunks
        ]

        chunk_query = f"""
        UNWIND $batch AS row
        MERGE (c:{CHUNK_NODE} {{chunk_id: row.chunk_id}})
        SET c.text              = row.text,
            c.token_count       = row.token_count,
            c.chunk_index       = row.chunk_index,
            c.start_char        = row.start_char,
            c.end_char          = row.end_char,
            c.heading_hierarchy = row.heading_hierarchy,
            c.source_file       = row.source_file
        """
        self.client.execute_batch(chunk_query, chunk_batch)

        # Step 4 — Link Chunks to Document via PART_OF
        logger.info("Creating PART_OF links.")
        part_of_query = f"""
        UNWIND $batch AS row
        MATCH (d:{DOCUMENT_NODE} {{file_hash: $file_hash}})
        MATCH (c:{CHUNK_NODE} {{chunk_id: row.chunk_id}})
        MERGE (c)-[:{PART_OF}]->(d)
        """
        part_of_batch = [
            {"chunk_id": c.chunk_id} for c in chunks_output.chunks
        ]
        for i in range(0, len(part_of_batch), 100):
            batch_slice = part_of_batch[i : i + 100]
            self.client.execute_write(
                part_of_query,
                {"batch": batch_slice, "file_hash": meta.file_hash},
            )

        # Step 5 — Create sequential NEXT_CHUNK relationships
        seq_params = []
        sorted_chunks = sorted(
            chunks_output.chunks, key=lambda c: c.chunk_index
        )
        for i in range(len(sorted_chunks) - 1):
            seq_params.append(
                {
                    "current_chunk_id": sorted_chunks[i].chunk_id,
                    "next_chunk_id": sorted_chunks[i + 1].chunk_id,
                }
            )

        if seq_params:
            logger.info("Creating %d NEXT_CHUNK link(s).", len(seq_params))
            seq_query = f"""
            UNWIND $batch AS row
            MATCH (c1:{CHUNK_NODE} {{chunk_id: row.current_chunk_id}})
            MATCH (c2:{CHUNK_NODE} {{chunk_id: row.next_chunk_id}})
            MERGE (c1)-[:{NEXT_CHUNK}]->(c2)
            """
            self.client.execute_batch(seq_query, seq_params)

        # Step 6 — Log results
        result = self.client.execute_query(
            "MATCH (c:Chunk) RETURN count(c) AS chunk_count"
        )
        logger.info(
            "Chunk graph built: %d Chunk nodes.", result[0]["chunk_count"]
        )

