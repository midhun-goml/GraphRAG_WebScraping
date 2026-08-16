

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from exceptions import EntityResolutionError, LLMGenerationError
from extraction.prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    build_entity_relationship_extraction_prompt_json,
)
from ingestion.models import (
    ChunkExtraction,
    Entity,
    GraphData,
    Relationship,
    TextChunk,
)

logger = logging.getLogger(__name__)


class KnowledgeExtractor:
    

    def __init__(
        self,
        llm_client: Any,
        embedding_model: Any,
        output_dir: str | Path | None = None,
    ) -> None:
        self.llm = llm_client
        self.embedding_model = embedding_model
        self.output_dir = Path(
            output_dir or os.getenv("OUTPUT_DIR", "output")
        )
        logger.info("KnowledgeExtractor initialised.")

    # ==================================================================
    # Main entry point
    # ==================================================================
    def extract_from_chunks(self, chunks: list[TextChunk]) -> GraphData:
        """Run the full extraction pipeline on *chunks*.

        Parameters
        ----------
        chunks : list[TextChunk]

        Returns
        -------
        GraphData
        """
        logger.info("Starting extraction for %d chunk(s).", len(chunks))

        # Step 1: per-chunk extraction
        extractions = self._extract_all_chunks(chunks)

        # Step 2: global entity resolution
        entities = self._resolve_entities(extractions)

        # Step 3: relationship resolution
        relationships = self._resolve_relationships(extractions, entities)

        # Step 4: generate embeddings
        entities = self._generate_embeddings(entities)

        # Step 5: validation & logging
        self._validate(entities, relationships)

        # Assemble GraphData
        graph_data = GraphData(
            entities=entities,
            relationships=relationships,
            chunks=chunks,
            extractions=extractions,
            metadata=chunks[0].metadata if chunks else None,
        )

        # Step 6: serialise
        self._save(graph_data)

        logger.info(
            "Extraction complete — %d entities, %d relationships, "
            "%d chunks, %d extractions.",
            len(entities),
            len(relationships),
            len(chunks),
            len(extractions),
        )
        return graph_data

    # ==================================================================
    # Step 1: Per-chunk extraction
    # ==================================================================
    def _extract_all_chunks(
        self, chunks: list[TextChunk]
    ) -> list[ChunkExtraction]:
        """Call the LLM for every chunk and parse the JSON response."""
        extractions: list[ChunkExtraction] = []

        for chunk in tqdm(chunks, desc="Extracting chunks"):
            try:
                extraction = self._extract_single(chunk)
                extractions.append(extraction)
                logger.info(
                    "Chunk %s — %d entities, %d relationships.",
                    chunk.chunk_id,
                    len(extraction.entities),
                    len(extraction.relationships),
                )
            except (LLMGenerationError, Exception) as exc:
                logger.error(
                    "Failed to extract chunk %s: %s", chunk.chunk_id, exc
                )

        logger.info(
            "Per-chunk extraction — %d/%d succeeded.",
            len(extractions),
            len(chunks),
        )
        return extractions

    def _extract_single(self, chunk: TextChunk) -> ChunkExtraction:
        """Extract knowledge from a single chunk via the LLM."""
        prompt = build_entity_relationship_extraction_prompt_json(
            chunk.text, chunk.chunk_id
        )
        raw = self.llm.generate_json(prompt, EXTRACTION_SYSTEM_PROMPT)

        # Parse entities
        entities: list[Entity] = []
        for e in raw.get("entities", []):
            try:
                entities.append(
                    Entity(
                        name=e.get("name", "").strip(),
                        type=e.get("type", "CONCEPT").strip(),
                        description=e.get("description", ""),
                        aliases=e.get("aliases", []),
                        source_chunk_ids=[chunk.chunk_id],
                    )
                )
            except Exception as exc:
                logger.warning("Skipping malformed entity %s: %s", e, exc)

        # Parse relationships
        relationships: list[Relationship] = []
        for r in raw.get("relationships", []):
            try:
                weight_raw = r.get("weight", 0.8)
                if isinstance(weight_raw, (int, float)):
                    weight = float(weight_raw)
                    if weight > 1.0:
                        weight = weight / 10.0
                else:
                    weight = 0.8

                relationships.append(
                    Relationship(
                        source_entity=r.get("source_entity", "").strip(),
                        target_entity=r.get("target_entity", "").strip(),
                        relation_type=r.get("relation_type", "RELATED_TO"),
                        description=r.get("description", ""),
                        weight=max(0.0, min(1.0, weight)),
                        source_chunk_ids=[chunk.chunk_id],
                    )
                )
            except Exception as exc:
                logger.warning("Skipping malformed relationship %s: %s", r, exc)

        return ChunkExtraction(
            chunk_id=chunk.chunk_id,
            entities=entities,
            relationships=relationships,
            summary=raw.get("summary", ""),
        )

    # ==================================================================
    # Step 2: Global entity resolution
    # ==================================================================
    def _resolve_entities(
        self, extractions: list[ChunkExtraction]
    ) -> list[Entity]:
        """Normalize, canonicalize, merge aliases, deduplicate entities."""
        all_entities: list[Entity] = []
        for ext in extractions:
            all_entities.extend(ext.entities)

        if not all_entities:
            return []

        # Group by normalised name
        groups: dict[str, list[Entity]] = defaultdict(list)
        for ent in all_entities:
            key = self._normalize_name(ent.name)
            groups[key].append(ent)

        # Merge each group
        resolved: list[Entity] = []
        for norm_name, group in groups.items():
            merged = self._merge_entity_group(group)
            resolved.append(merged)

        # Alias-based merging
        resolved = self._merge_by_aliases(resolved)

        logger.info(
            "Entity resolution — %d raw → %d resolved.",
            len(all_entities),
            len(resolved),
        )
        return resolved

    @staticmethod
    def _normalize_name(name: str) -> str:
        name = name.lower().strip()
        name = re.sub(r"[^\w\s-]", "", name)
        return re.sub(r"\s+", " ", name).strip()

    def _merge_entity_group(self, group: list[Entity]) -> Entity:
        name_counts = Counter(e.name for e in group)
        canonical_name = name_counts.most_common(1)[0][0]

        type_counts = Counter(e.type for e in group)
        canonical_type = type_counts.most_common(1)[0][0]
        if len(type_counts) > 1:
            logger.warning(
                canonical_name,
                dict(type_counts),
                canonical_type,
            )

        descriptions = [e.description for e in group if e.description]
        best_desc = max(descriptions, key=len) if descriptions else ""

        all_aliases: set[str] = set()
        for e in group:
            all_aliases.update(e.aliases)
            if e.name != canonical_name:
                all_aliases.add(e.name)
        all_aliases.discard(canonical_name)

        all_chunks: set[str] = set()
        for e in group:
            all_chunks.update(e.source_chunk_ids)

        return Entity(
            name=canonical_name,
            type=canonical_type,
            description=best_desc,
            aliases=sorted(all_aliases),
            source_chunk_ids=sorted(all_chunks),
        )

    def _merge_by_aliases(self, entities: list[Entity]) -> list[Entity]:
        name_map: dict[str, Entity] = {e.name: e for e in entities}
        alias_to_name: dict[str, str] = {}
        for e in entities:
            for alias in e.aliases:
                norm = self._normalize_name(alias)
                alias_to_name[norm] = e.name

        merged_away: set[str] = set()
        for entity in entities:
            norm = self._normalize_name(entity.name)
            if norm in alias_to_name and alias_to_name[norm] != entity.name:
                target_name = alias_to_name[norm]
                if target_name in name_map and entity.name not in merged_away:
                    target = name_map[target_name]
                    merged = self._merge_entity_group([target, entity])
                    name_map[target_name] = merged
                    merged_away.add(entity.name)

        return [e for name, e in name_map.items() if name not in merged_away]

    # ==================================================================
    # Step 3: Relationship resolution
    # ==================================================================
    def _resolve_relationships(
        self,
        extractions: list[ChunkExtraction],
        resolved_entities: list[Entity],
    ) -> list[Relationship]:
        """Remap, merge, and validate relationships."""
        canonical_map: dict[str, str] = {}
        for ent in resolved_entities:
            canonical_map[self._normalize_name(ent.name)] = ent.name
            for alias in ent.aliases:
                canonical_map[self._normalize_name(alias)] = ent.name

        entity_names = {e.name for e in resolved_entities}

        all_rels: list[Relationship] = []
        for ext in extractions:
            for rel in ext.relationships:
                src = canonical_map.get(
                    self._normalize_name(rel.source_entity),
                    rel.source_entity,
                )
                tgt = canonical_map.get(
                    self._normalize_name(rel.target_entity),
                    rel.target_entity,
                )

                if src == tgt:
                    continue

                if src not in entity_names or tgt not in entity_names:
                    logger.debug(
                        "Dropping relationship '%s' → '%s' "
                        "(missing endpoint).",
                        src,
                        tgt,
                    )
                    continue

                all_rels.append(
                    Relationship(
                        source_entity=src,
                        target_entity=tgt,
                        relation_type=rel.relation_type,
                        description=rel.description,
                        weight=rel.weight,
                        source_chunk_ids=list(rel.source_chunk_ids),
                    )
                )

        merged: dict[tuple, Relationship] = {}
        for rel in all_rels:
            key = (rel.source_entity, rel.target_entity, rel.relation_type)
            if key in merged:
                existing = merged[key]
                new_weight = (existing.weight + rel.weight) / 2.0
                new_chunks = sorted(
                    set(existing.source_chunk_ids + rel.source_chunk_ids)
                )
                desc = (
                    rel.description
                    if len(rel.description) > len(existing.description)
                    else existing.description
                )
                merged[key] = Relationship(
                    source_entity=rel.source_entity,
                    target_entity=rel.target_entity,
                    relation_type=rel.relation_type,
                    description=desc,
                    weight=new_weight,
                    source_chunk_ids=new_chunks,
                )
            else:
                merged[key] = rel

        resolved = list(merged.values())
        logger.info(
            "Relationship resolution — %d raw → %d resolved.",
            len(all_rels),
            len(resolved),
        )
        return resolved

    # ==================================================================
    # Step 4: Embedding generation
    # ==================================================================
    def _generate_embeddings(self, entities: list[Entity]) -> list[Entity]:
        """Batch-generate embeddings for all entities."""
        if not entities:
            return entities

        texts = [
            f"{e.name}: {e.type}. {e.description}" for e in entities
        ]

        logger.info("Generating embeddings for %d entities.", len(entities))
        embeddings = self.embedding_model.embed_batch(texts)

        updated: list[Entity] = []
        for ent, emb in zip(entities, embeddings):
            updated.append(ent.model_copy(update={"embedding": emb}))

        logger.info("Embeddings generated — dimension=%d.", len(embeddings[0]))
        return updated

    # ==================================================================
    # Step 5: Validation
    # ==================================================================
    def _validate(
        self, entities: list[Entity], relationships: list[Relationship]
    ) -> None:
        """Log validation stats and warnings."""
        entity_names = {e.name for e in entities}

        related = set()
        for r in relationships:
            related.add(r.source_entity)
            related.add(r.target_entity)

        orphans = entity_names - related
        if orphans:
            logger.warning(
                "%d orphan entities (no relationships): %s",
                len(orphans),
                list(orphans)[:10],
            )

        for r in relationships:
            if r.source_entity not in entity_names:
                logger.warning("Dangling source: '%s'.", r.source_entity)
            if r.target_entity not in entity_names:
                logger.warning("Dangling target: '%s'.", r.target_entity)

        logger.info(
            "Validation — %d entities, %d relationships, %d orphans.",
            len(entities),
            len(relationships),
            len(orphans),
        )

    # ==================================================================
    # Step 6: Serialisation
    # ==================================================================
    def _save(self, graph_data: GraphData) -> None:
        """Write ``GraphData`` to ``output/graph_data.json``."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "graph_data.json"

        with open(path, "w", encoding="utf-8") as fh:
            fh.write(graph_data.model_dump_json(indent=2))

        logger.info("Saved graph data to %s.", path)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(llm={self.llm!r})"
