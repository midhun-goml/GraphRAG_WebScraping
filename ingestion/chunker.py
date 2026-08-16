"""
chunker.py — Markdown-Aware Heading-Based Token Chunker
=========================================================

Splits cleaned markdown text into overlapping, token-counted chunks
that respect heading boundaries.  Each chunk carries a deterministic
``chunk_id`` derived from the document's file hash and a sequential index.

Key Design Decisions
--------------------
1. **Heading-aware sections** — text is first split by ``#`` headings;
   no chunk ever spans two different sections.
2. **Heading context prefix** — every chunk text starts with
   ``[H1 > H2 > H3]`` so it's self-contained for an LLM.
3. **Sentence-level granularity** — within each section, text is split
   at sentence boundaries.
4. **Token-aware sizing** — limits measured in tokens via ``tiktoken``.
5. **Overlap** — trailing sentences from the previous chunk seed the next.
6. **Tables are atomic** — markdown tables are never split mid-row.
7. **Character offsets** — ``start_char`` / ``end_char`` enable provenance
   mapping back to the cleaned text.

The legacy ``TokenAwareChunker`` name is kept as an alias for backward
compatibility.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

import tiktoken #type: ignore

from exceptions import ChunkingError
from ingestion.models import DocumentMetadata, TextChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------
_RE_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")
"""Match a markdown heading line and capture (hashes, title)."""

_RE_SENTENCE_SPLIT = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z])|(?<=\n\n)"
)
"""Split on sentence-ending punctuation + whitespace or paragraph breaks."""

_RE_TABLE_BLOCK = re.compile(
    r"(?:^\|.+\|$\n?)+", re.MULTILINE
)
"""Match a contiguous block of markdown table lines."""

_RE_LIST_ITEM = re.compile(r"^(?:\s*[-*]|\s*\d+\.)\s", re.MULTILINE)
"""Match the start of a markdown list item."""


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------
@dataclass
class _Section:
    """A parsed markdown section."""

    heading_hierarchy: list[str] = field(default_factory=list)
    body_text: str = ""
    start_char: int = 0


# ---------------------------------------------------------------------------
# MarkdownChunker
# ---------------------------------------------------------------------------
class MarkdownChunker:
    """Split cleaned markdown into overlapping, heading-aware token chunks.

    Parameters
    ----------
    chunk_size : int
        Maximum tokens per chunk.
    chunk_overlap : int
        Number of overlap tokens carried into the next chunk.
    tokenizer_name : str
        ``tiktoken`` encoding name.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        tokenizer_name: str = "cl100k_base",
    ) -> None:
        self.chunk_size = chunk_size or int(
            os.getenv("CHUNK_SIZE_TOKENS", "500")
        )
        self.chunk_overlap = chunk_overlap or int(
            os.getenv("CHUNK_OVERLAP_TOKENS", "80")
        )
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)

        if self.chunk_size <= 0:
            raise ChunkingError(
                f"chunk_size must be positive, got {self.chunk_size}."
            )
        if self.chunk_overlap >= self.chunk_size:
            raise ChunkingError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})."
            )

        logger.info(
            "MarkdownChunker — size=%d, overlap=%d, encoding='%s'.",
            self.chunk_size,
            self.chunk_overlap,
            tokenizer_name,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def count_tokens(self, text: str) -> int:
        """Return the token count for *text*."""
        return len(self.tokenizer.encode(text))

    def chunk(
        self, text: str, metadata: DocumentMetadata
    ) -> list[TextChunk]:
        """Split *text* into heading-aware, token-counted chunks.

        Parameters
        ----------
        text : str
            Cleaned markdown text.
        metadata : DocumentMetadata
            Source document metadata (provides ``file_hash``).

        Returns
        -------
        list[TextChunk]

        Raises
        ------
        ChunkingError
            If the text is empty.
        """
        if not text or not text.strip():
            raise ChunkingError("Cannot chunk empty text.")

        # Step 1: Parse into sections by heading
        sections = self._parse_sections(text)
        logger.info(
            "Parsed %d section(s) from markdown.", len(sections)
        )

        # Step 2 & 3: Chunk each section
        chunks: list[TextChunk] = []
        chunk_index = 0

        for section in sections:
            # Build heading context prefix
            heading_ctx = " > ".join(section.heading_hierarchy)
            prefix = f"[{heading_ctx}]\n" if heading_ctx else ""

            body = section.body_text.strip()
            if not body:
                continue

            section_text = prefix + body

            # Split into atomic units (sentences / table blocks / list items)
            units = self._split_into_units(section_text)
            if not units:
                continue

            # Accumulate units into chunks
            i = 0
            while i < len(units):
                current_units: list[str] = []
                current_tokens = 0

                while i < len(units):
                    unit = units[i]
                    unit_tokens = self.count_tokens(unit)

                    if (
                        current_units
                        and current_tokens + unit_tokens > self.chunk_size
                    ):
                        break

                    # Force-split oversized single unit
                    if not current_units and unit_tokens > self.chunk_size:
                        force_chunks = self._force_split(unit)
                        for fc_text in force_chunks:
                            fc_tokens = self.count_tokens(fc_text)
                            if fc_tokens <= 0:
                                continue

                            # Compute character offsets
                            s_char = text.find(
                                fc_text[:50],
                                section.start_char,
                            )
                            if s_char == -1:
                                s_char = section.start_char
                            e_char = s_char + len(fc_text)

                            chunk_id = (
                                f"{metadata.file_hash[:12]}"
                                f"_{chunk_index:04d}"
                            )
                            chunks.append(
                                TextChunk(
                                    chunk_id=chunk_id,
                                    text=fc_text,
                                    token_count=fc_tokens,
                                    chunk_index=chunk_index,
                                    start_char=s_char,
                                    end_char=e_char,
                                    heading_hierarchy=list(
                                        section.heading_hierarchy
                                    ),
                                    metadata=metadata,
                                )
                            )
                            chunk_index += 1
                        i += 1
                        continue

                    current_units.append(unit)
                    current_tokens += unit_tokens
                    i += 1

                if not current_units:
                    continue

                chunk_text = "\n".join(current_units)
                c_tokens = self.count_tokens(chunk_text)

                if c_tokens <= 0:
                    continue

                # Compute character offsets in the full cleaned text
                first_unit_sample = current_units[0][:50]
                s_char = text.find(
                    first_unit_sample, section.start_char
                )
                if s_char == -1:
                    s_char = section.start_char
                e_char = s_char + len(chunk_text)

                chunk_id = (
                    f"{metadata.file_hash[:12]}_{chunk_index:04d}"
                )
                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        token_count=c_tokens,
                        chunk_index=chunk_index,
                        start_char=s_char,
                        end_char=e_char,
                        heading_hierarchy=list(
                            section.heading_hierarchy
                        ),
                        metadata=metadata,
                    )
                )
                chunk_index += 1

                # Compute overlap: back up units whose cumulative
                # tokens ≤ chunk_overlap
                overlap_units: list[str] = []
                overlap_tokens = 0
                for unit in reversed(current_units):
                    ut = self.count_tokens(unit)
                    if overlap_tokens + ut > self.chunk_overlap:
                        break
                    overlap_tokens += ut
                    overlap_units.insert(0, unit)

                if overlap_units and i < len(units):
                    i -= len(overlap_units)
                    if i < 0:
                        i = 0

        logger.info(
            "Chunking complete — %d chunk(s) produced.", len(chunks)
        )
        return chunks

    # ------------------------------------------------------------------
    # Step 1: Parse markdown into sections by heading
    # ------------------------------------------------------------------
    def _parse_sections(self, text: str) -> list[_Section]:
        """Split *text* into sections delimited by heading lines.

        Each section records the heading hierarchy (stack) and the body
        text that follows the heading.
        """
        lines = text.split("\n")
        sections: list[_Section] = []

        heading_stack: list[tuple[int, str]] = []  # (level, title)
        current_body_lines: list[str] = []
        current_start = 0
        current_hierarchy: list[str] = []

        char_offset = 0

        for line in lines:
            match = _RE_HEADING.match(line.strip())
            if match:
                # Flush current section
                if current_body_lines or current_hierarchy:
                    sections.append(
                        _Section(
                            heading_hierarchy=list(current_hierarchy),
                            body_text="\n".join(current_body_lines),
                            start_char=current_start,
                        )
                    )
                    current_body_lines = []

                # Determine heading level
                level = len(match.group(1))
                title = match.group(2).strip()

                # Pop headings with level >= current
                while (
                    heading_stack
                    and heading_stack[-1][0] >= level
                ):
                    heading_stack.pop()

                heading_stack.append((level, title))
                current_hierarchy = [h[1] for h in heading_stack]
                current_start = char_offset
            else:
                current_body_lines.append(line)

            char_offset += len(line) + 1  # +1 for \n

        # Flush final section
        if current_body_lines or current_hierarchy:
            sections.append(
                _Section(
                    heading_hierarchy=list(current_hierarchy),
                    body_text="\n".join(current_body_lines),
                    start_char=current_start,
                )
            )

        # If no headings at all, treat entire text as one section
        if not sections:
            sections.append(
                _Section(
                    heading_hierarchy=[],
                    body_text=text,
                    start_char=0,
                )
            )

        return sections

    # ------------------------------------------------------------------
    # Step 2: Split section text into atomic units
    # ------------------------------------------------------------------
    def _split_into_units(self, text: str) -> list[str]:
        """Split *text* into atomic units for chunking.

        - Markdown tables are kept as single atomic blocks.
        - List items are kept intact.
        - Remaining prose is split at sentence boundaries.
        """
        units: list[str] = []

        # First extract table blocks as atomic units
        remaining = text
        table_matches = list(_RE_TABLE_BLOCK.finditer(text))

        if table_matches:
            last_end = 0
            for m in table_matches:
                # Process text before the table
                before = remaining[last_end : m.start()]
                if before.strip():
                    units.extend(self._split_prose(before))
                # Add table as atomic unit
                table_text = m.group().strip()
                if table_text:
                    units.append(table_text)
                last_end = m.end()
            # Process text after last table
            after = remaining[last_end:]
            if after.strip():
                units.extend(self._split_prose(after))
        else:
            units.extend(self._split_prose(text))

        return [u for u in units if u.strip()]

    def _split_prose(self, text: str) -> list[str]:
        """Split prose text into sentences, keeping list items atomic."""
        parts = _RE_SENTENCE_SPLIT.split(text)
        result: list[str] = []
        for part in parts:
            stripped = part.strip()
            if stripped:
                result.append(stripped)
        return result

    # ------------------------------------------------------------------
    # Force-split oversized units at token boundaries
    # ------------------------------------------------------------------
    def _force_split(self, text: str) -> list[str]:
        """Force-split a single unit that exceeds ``chunk_size`` tokens."""
        tokens = self.tokenizer.encode(text)
        pieces: list[str] = []
        for i in range(0, len(tokens), self.chunk_size):
            segment_tokens = tokens[i : i + self.chunk_size]
            segment_text = self.tokenizer.decode(segment_tokens)
            if segment_text.strip():
                pieces.append(segment_text.strip())
        logger.warning(
            "Force-split oversized unit (%d tokens) into %d pieces.",
            len(tokens),
            len(pieces),
        )
        return pieces

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
        )


# ---------------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------------
TokenAwareChunker = MarkdownChunker
"""Legacy alias so existing code importing ``TokenAwareChunker`` continues
to work.  New code should use ``MarkdownChunker``."""
