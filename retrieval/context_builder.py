

from __future__ import annotations

import logging
import os
from typing import Any

import tiktoken #type: ignore

logger = logging.getLogger(__name__)


class ContextBuilder:
    

    def __init__(self, max_context_tokens: int = 4000) -> None:
        self.max_context_tokens = max_context_tokens
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        logger.info(
            "ContextBuilder initialised — budget=%d tokens.",
            max_context_tokens,
        )

    def _count_tokens(self, text: str) -> int:
        """Return the token count for *text*."""
        return len(self.tokenizer.encode(text))

    def build_context(
        self, search_results: dict[str, Any], user_question: str = ""
    ) -> str:
        """Construct the final context string.

        Parameters
        ----------
        search_results : dict
            Output of ``GraphSearchEngine.search()``.
        user_question : str
            The original question (for relevance scoring).

        Returns
        -------
        str
        """
        entities = search_results.get("entities", [])
        relationships = search_results.get("relationships", [])
        chunks = search_results.get("chunks", [])

        parts: list[str] = []
        tokens_used = 0

        # --- Section 1: Entities (~reserved budget) --------------------
        entity_section = self._build_entity_section(entities)
        entity_tokens = self._count_tokens(entity_section)
        parts.append(entity_section)
        tokens_used += entity_tokens

        # --- Section 2: Relationships ---------------------------------
        rel_section = self._build_relationship_section(relationships)
        rel_tokens = self._count_tokens(rel_section)
        parts.append(rel_section)
        tokens_used += rel_tokens

        # --- Section 3: Chunks (fill remaining budget) ----------------
        remaining = max(0, self.max_context_tokens - tokens_used)
        chunk_section = self._build_chunk_section(chunks, remaining)
        parts.append(chunk_section)

        context = "\n\n".join(parts)
        total_tokens = self._count_tokens(context)

        # --- Hard budget enforcement ----------------------------------
        # Entities + relationships may already exceed the budget.
        # Truncate to guarantee the context fits within the limit.
        if total_tokens > self.max_context_tokens:
            encoded = self.tokenizer.encode(context)
            context = self.tokenizer.decode(
                encoded[: self.max_context_tokens]
            )
            total_tokens = self.max_context_tokens

        logger.info(
            "Context built — %d entities, %d rels, %d chunks, "
            "%d tokens.",
            len(entities),
            len(relationships),
            len(chunks),
            total_tokens,
        )
        return context

    def _build_entity_section(
        self, entities: list[dict[str, Any]]
    ) -> str:
        """Format the entity section."""
        lines = ["=== RELEVANT ENTITIES ==="]
        for e in entities:
            name = e.get("name", "")
            etype = e.get("type", "")
            desc = e.get("description", "")
            lines.append(f"- {name} ({etype}): {desc}")
        return "\n".join(lines)

    def _build_relationship_section(
        self, relationships: list[dict[str, Any]]
    ) -> str:
        """Format the relationship section."""
        lines = ["=== KEY RELATIONSHIPS ==="]
        for r in relationships:
            src = r.get("source", "")
            rel = r.get("relation", "")
            tgt = r.get("target", "")
            desc = r.get("description", "")
            weight = r.get("weight", "")
            line = f"- {src} --[{rel}]--> {tgt}"
            if desc:
                line += f": {desc}"
            lines.append(line)
        return "\n".join(lines)

    def _build_chunk_section(
        self, chunks: list[dict[str, Any]], token_budget: int
    ) -> str:
        """Format chunk text within the token budget."""
        lines = ["=== RELEVANT TEXT PASSAGES ==="]
        tokens_used = self._count_tokens(lines[0])

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "")
            text = chunk.get("text", "")
            source = chunk.get("source_file", "")

            entry = f"[Chunk {chunk_id}] (Source: {source}):\n{text}"
            entry_tokens = self._count_tokens(entry)

            if tokens_used + entry_tokens > token_budget:
                logger.debug(
                    "Token budget exhausted at chunk %s.", chunk_id
                )
                break

            lines.append(entry)
            tokens_used += entry_tokens

        return "\n\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"max_tokens={self.max_context_tokens})"
        )
