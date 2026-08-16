"""
graph_search.py — Hybrid Graph Search Engine
==============================================

Implements dual retrieval: graph-structural matching (exact + fuzzy
Cypher queries) combined with vector similarity search via Neo4j's
native vector index.

Pipeline
--------
1. Query understanding (LLM → entities, keywords, intent).
2. Entity normalisation.
3. Dual retrieval:
   - 3a. Exact / fuzzy entity matching (Cypher CONTAINS + aliases).
   - 3b. Vector similarity search (entity_embedding_index).
4. Merge & deduplicate with combined scoring.
5. Graph traversal (1-hop neighbours, connected chunks).
6. Neighbour chunk expansion (NEXT_CHUNK).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from exceptions import RetrievalError
from graph.neo4j_client import Neo4jClient
from graph.schema import CHUNK_NODE, ENTITY_NODE, EXTRACTED_FROM, NEXT_CHUNK

# Stopwords for local keyword extraction (no LLM needed)
_STOPWORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must", "ought",
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
    "she", "her", "it", "its", "they", "them", "their", "this", "that",
    "these", "those", "what", "which", "who", "whom", "whose", "where",
    "when", "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "some", "any", "no", "not", "only", "own", "same", "so",
    "than", "too", "very", "just", "about", "above", "after", "again",
    "against", "along", "also", "among", "and", "as", "at", "because",
    "before", "below", "between", "but", "by", "down", "during", "for",
    "from", "if", "in", "into", "nor", "of", "on", "or", "out", "over",
    "per", "then", "through", "to", "under", "until", "up", "upon",
    "with", "without", "tell", "explain", "describe", "give", "show",
    "please", "know", "find", "get", "make", "go", "see", "look",
    "come", "think", "say", "take", "want", "use", "used",
}

logger = logging.getLogger(__name__)


class GraphSearchEngine:
    

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        llm_client: Any,
        embedding_model: Any,
    ) -> None:
        self.client = neo4j_client
        self.llm = llm_client
        self.embedding_model = embedding_model
        logger.info("GraphSearchEngine initialised.")

    # ==================================================================
    # Main entry point
    # ==================================================================
    def search(
        self, user_question: str, top_k: int = 10
    ) -> dict[str, Any]:
       
        # Step 1: Local n-gram keyword extraction (fast, no LLM)
        query_info = self._extract_keywords(user_question)
        keywords = query_info.get("keywords", [])
        intent = query_info.get("intent", "")

        logger.info(
            "Query analysis — keywords=%s, intent='%s'.",
            keywords,
            intent,
        )

        # Step 2: Embedding-based entity discovery (vector search)
        #   This IS the smart entity extraction — the embedding model
        #   embeds the question and finds semantically similar entities
        #   in the graph.  No LLM round-trip required.
        vector_entities = self._vector_entity_match(
            user_question, top_k
        )

        # Step 3: Graph-based entity matching
        #   Feed both vector-discovered entity names AND n-gram keywords
        #   into exact/fuzzy Cypher matching for high recall.
        discovered_names = [e["name"] for e in vector_entities]
        search_entities = [
            self._normalize(n) for n in discovered_names
        ]
        search_keywords = [
            self._normalize(k) for k in keywords
        ]

        graph_entities = self._graph_entity_match(
            search_entities, search_keywords, top_k
        )

        # Step 4: Merge & deduplicate
        merged_entities = self._merge_results(
            graph_entities, vector_entities
        )

        entity_names = [e["name"] for e in merged_entities]

        # Step 5: Graph traversal
        neighbours = self._find_neighbours(entity_names)
        relationships = self._find_relationships(entity_names)
        chunks = self._find_connected_chunks(entity_names)

        # Step 6: Neighbour chunk expansion
        chunk_ids = [c["chunk_id"] for c in chunks]
        neighbour_chunks = self._expand_chunks(chunk_ids)

        # Deduplicate chunks
        seen_chunk_ids: set[str] = set()
        all_chunks: list[dict] = []
        for c in chunks + neighbour_chunks:
            cid = c.get("chunk_id", "")
            if cid not in seen_chunk_ids:
                seen_chunk_ids.add(cid)
                all_chunks.append(c)

        # Merge neighbour entities
        seen_entity_names: set[str] = {e["name"] for e in merged_entities}
        for n in neighbours:
            if n["name"] not in seen_entity_names:
                merged_entities.append(n)
                seen_entity_names.add(n["name"])

        result = {
            "entities": merged_entities,
            "relationships": relationships,
            "chunks": all_chunks,
            "metadata": {
                "query_keywords": keywords,
                "intent": intent,
            },
        }

        logger.info(
            "Search complete — %d entities, %d relationships, %d chunks.",
            len(merged_entities),
            len(relationships),
            len(all_chunks),
        )
        return result

    # ------------------------------------------------------------------
    # Step 1: N-gram keyword extraction (local, no LLM)
    # ------------------------------------------------------------------
    def _extract_keywords(self, question: str) -> dict[str, Any]:
        """Extract n-gram keywords and classify intent locally.

        Builds unigram, bigram, and trigram candidates from the
        question after stopword removal.  Entity discovery is handled
        separately by embedding-based vector search (see ``search()``).
        """
        # Tokenise and filter stopwords
        tokens = re.findall(r"[a-zA-Z0-9_'-]+", question)
        clean = [
            t for t in tokens
            if t.lower() not in _STOPWORDS and len(t) > 1
        ]

        # Build n-gram candidates: unigrams + bigrams + trigrams
        candidates: list[str] = list(clean)
        for i in range(len(clean) - 1):
            candidates.append(f"{clean[i]} {clean[i + 1]}")
        for i in range(len(clean) - 2):
            candidates.append(
                f"{clean[i]} {clean[i + 1]} {clean[i + 2]}"
            )

        # Deduplicate while preserving order
        seen: set[str] = set()
        keywords: list[str] = []
        for kw in candidates:
            kw_lower = kw.lower()
            if kw_lower not in seen:
                seen.add(kw_lower)
                keywords.append(kw)

        # Classify intent with simple heuristics
        q_lower = question.lower()
        if any(w in q_lower for w in ("how many", "count", "total", "all the")):
            intent = "aggregation"
        elif any(w in q_lower for w in ("how does", "how do", "how to", "steps", "process")):
            intent = "procedural"
        elif any(w in q_lower for w in ("compare", "differ", "versus", "vs")):
            intent = "comparative"
        elif any(w in q_lower for w in ("what is", "explain", "describe", "tell me about")):
            intent = "exploratory"
        else:
            intent = "factual"

        return {"keywords": keywords, "intent": intent}

    # ------------------------------------------------------------------
    # Step 2: Normalisation
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase and strip whitespace."""
        return text.lower().strip()

    # ------------------------------------------------------------------
    # Step 3a: Graph entity matching
    # ------------------------------------------------------------------
    def _graph_entity_match(
        self,
        entity_names: list[str],
        keywords: list[str],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Exact + fuzzy entity matching via Cypher."""
        results: list[dict] = []

        # Exact + alias matching
        if entity_names:
            query = f"""
            UNWIND $entity_names AS name
            MATCH (e:{ENTITY_NODE})
            WHERE toLower(e.name) = toLower(name)
               OR toLower(name) IN [alias IN e.aliases | toLower(alias)]
            RETURN DISTINCT e.name AS name, e.type AS type,
                   e.description AS description
            """
            records = self.client.execute_query(
                query, {"entity_names": entity_names}
            )
            for r in records:
                r["match_source"] = "exact"
                r["score"] = 1.0
            results.extend(records)

        # Fuzzy keyword matching
        if keywords:
            query = f"""
            UNWIND $keywords AS keyword
            MATCH (e:{ENTITY_NODE})
            WHERE toLower(e.name) CONTAINS toLower(keyword)
               OR toLower(e.description) CONTAINS toLower(keyword)
            RETURN DISTINCT e.name AS name, e.type AS type,
                   e.description AS description
            LIMIT $top_k
            """
            records = self.client.execute_query(
                query, {"keywords": keywords, "top_k": top_k}
            )
            for r in records:
                r["match_source"] = "fuzzy"
                r["score"] = 0.7
            results.extend(records)

        return results

    # ------------------------------------------------------------------
    # Step 3b: Vector similarity search
    # ------------------------------------------------------------------
    def _vector_entity_match(
        self, question: str, top_k: int
    ) -> list[dict[str, Any]]:
        """Embed the question and query the vector index."""
        try:
            query_embedding = self.embedding_model.embed(question)
            query = """
            CALL db.index.vector.queryNodes(
                'entity_embedding_index', $top_k, $query_embedding
            )
            YIELD node, score
            RETURN node.name AS name, node.type AS type,
                   node.description AS description, score
            ORDER BY score DESC
            """
            # Note: db.index.vector.queryNodes works on Neo4j 5.11+.
            # If your Neo4j version supports the newer VECTOR SEARCH
            # syntax, consider migrating. The current form is functional.
            records = self.client.execute_query(
                query,
                {"top_k": top_k, "query_embedding": query_embedding},
            )
            for r in records:
                r["match_source"] = "vector"
            return records
        except Exception as exc:
            logger.warning("Vector search failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Step 4: Merge & deduplicate
    # ------------------------------------------------------------------
    def _merge_results(
        self,
        graph_entities: list[dict],
        vector_entities: list[dict],
    ) -> list[dict[str, Any]]:
        """Merge graph + vector results with combined scoring."""
        merged: dict[str, dict] = {}

        for e in graph_entities:
            name = e.get("name", "")
            if name not in merged:
                merged[name] = {
                    "name": name,
                    "type": e.get("type", ""),
                    "description": e.get("description", ""),
                    "graph_score": e.get("score", 0.7),
                    "vector_score": 0.0,
                }
            else:
                merged[name]["graph_score"] = max(
                    merged[name]["graph_score"], e.get("score", 0.7)
                )

        for e in vector_entities:
            name = e.get("name", "")
            if name not in merged:
                merged[name] = {
                    "name": name,
                    "type": e.get("type", ""),
                    "description": e.get("description", ""),
                    "graph_score": 0.0,
                    "vector_score": e.get("score", 0.0),
                }
            else:
                merged[name]["vector_score"] = max(
                    merged[name]["vector_score"], e.get("score", 0.0)
                )

        # Combined score: graph * 0.6 + vector * 0.4
        for entry in merged.values():
            entry["combined_score"] = (
                entry["graph_score"] * 0.6 + entry["vector_score"] * 0.4
            )

        # Sort by combined score
        sorted_entities = sorted(
            merged.values(), key=lambda x: x["combined_score"], reverse=True
        )
        return sorted_entities

    # ------------------------------------------------------------------
    # Step 5: Graph traversal
    # ------------------------------------------------------------------
    def _find_neighbours(
        self, entity_names: list[str]
    ) -> list[dict[str, Any]]:
        """1-hop traversal: entities related to matched entities."""
        if not entity_names:
            return []

        query = f"""
        MATCH (e:{ENTITY_NODE})-[r]-(n:{ENTITY_NODE})
        WHERE e.name IN $names
        RETURN DISTINCT n.name AS name, n.type AS type,
               n.description AS description
        """
        return self.client.execute_query(query, {"names": entity_names})

    def _find_relationships(
        self, entity_names: list[str]
    ) -> list[dict[str, Any]]:
        """Find all relationships involving matched entities."""
        if not entity_names:
            return []

        query = f"""
        MATCH (s:{ENTITY_NODE})-[r]->(t:{ENTITY_NODE})
        WHERE s.name IN $names OR t.name IN $names
        RETURN DISTINCT s.name AS source, type(r) AS relation,
               t.name AS target,
               r.description AS description,
               r.weight AS weight
        """
        return self.client.execute_query(query, {"names": entity_names})

    def _find_connected_chunks(
        self, entity_names: list[str]
    ) -> list[dict[str, Any]]:
        """Find chunks connected to matched entities via EXTRACTED_FROM."""
        if not entity_names:
            return []

        query = f"""
        MATCH (e:{ENTITY_NODE})-[:{EXTRACTED_FROM}]->(c:{CHUNK_NODE})
        WHERE e.name IN $names
        RETURN DISTINCT c.chunk_id AS chunk_id,
               c.text AS text,
               c.source_file AS source_file,
               c.summary AS summary,
               c.chunk_index AS chunk_index
        ORDER BY c.chunk_index
        """
        return self.client.execute_query(query, {"names": entity_names})

    # ------------------------------------------------------------------
    # Step 6: Chunk expansion
    # ------------------------------------------------------------------
    def _expand_chunks(
        self, chunk_ids: list[str]
    ) -> list[dict[str, Any]]:
        """Expand chunks by retrieving NEXT_CHUNK neighbours."""
        if not chunk_ids:
            return []

        query = f"""
        UNWIND $chunk_ids AS cid
        MATCH (c:{CHUNK_NODE} {{chunk_id: cid}})-[:{NEXT_CHUNK}]-(n:{CHUNK_NODE})
        RETURN DISTINCT n.chunk_id AS chunk_id,
               n.text AS text,
               n.source_file AS source_file,
               n.summary AS summary,
               n.chunk_index AS chunk_index
        """
        return self.client.execute_query(query, {"chunk_ids": chunk_ids})

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(client={self.client!r})"
