"""
validator.py — Chunk Validation Gate
=====================================

Validates chunks before they enter the expensive LLM extraction step.
Filters out chunks that are purely heading context with
no body, consist primarily of boilerplate content, are mostly numeric,
or are exact duplicates.
"""

from __future__ import annotations

import logging
import os
import re

from ingestion.models import TextChunk

logger = logging.getLogger(__name__)

_RE_TOC = re.compile(
    r"(?:table\s+of\s+contents|contents|list\s+of\s+(?:figures|tables))",
    re.IGNORECASE,
)
_RE_REFERENCES = re.compile(
    r"^\s*(?:\[\d+\]|\d+\.)\s+[A-Z]", re.MULTILINE
)
_RE_BOILERPLATE_ONLY = re.compile(
    r"^\s*(?:table\s+of\s+contents|references|bibliography|index)\s*$",
    re.IGNORECASE,
)
_RE_HEADING_CONTEXT = re.compile(r"^\[.+\]\s*$")
"""Matches a chunk that is only a heading context line with no body."""


class ChunkValidator:
    

    def __init__(self, min_tokens: int | None = None) -> None:
        self.min_tokens = min_tokens or int(
            os.getenv("MIN_CHUNK_TOKENS", "30")
        )

    def validate(self, chunks: list[TextChunk]) -> list[TextChunk]:
        
        if chunks is None:
            raise ValueError("chunks must not be None.")

        validated: list[TextChunk] = []
        reasons: dict[str, int] = {}
        seen_texts: set[str] = set()

        for chunk in chunks:
            issues = self._check(chunk, seen_texts)
            if issues:
                for issue in issues:
                    reasons[issue] = reasons.get(issue, 0) + 1
                logger.debug(
                    "Chunk %s rejected: %s",
                    chunk.chunk_id,
                    "; ".join(issues),
                )
            else:
                # Track for duplicate detection
                seen_texts.add(chunk.text.strip())
                validated.append(chunk)

        rejected = len(chunks) - len(validated)
        logger.info(
            "Validated %d/%d chunks (%d removed).",
            len(validated),
            len(chunks),
            rejected,
        )
        if reasons:
            for reason, count in reasons.items():
                logger.info(
                    "  Rejection reason: %s (%d chunks)", reason, count
                )

        return validated

    def _check(
        self, chunk: TextChunk, seen_texts: set[str]
    ) -> list[str]:
        """Run all rules against a chunk. Return issue descriptions."""
        issues: list[str] = []

        # Rule 1: Minimum token count
        if chunk.token_count < self.min_tokens:
            issues.append(
                f"too short ({chunk.token_count} tokens < {self.min_tokens})"
            )

        # Rule 2: Only a heading context line with no body
        text_stripped = chunk.text.strip()
        if _RE_HEADING_CONTEXT.match(text_stripped):
            issues.append("heading context only (no body text)")

        # Rule 3: Pure boilerplate
        if self._is_boilerplate(text_stripped):
            issues.append("boilerplate content")

        # Rule 4: >80% numeric
        if self._is_mostly_numeric(text_stripped):
            issues.append(">80% numeric content")

        # Rule 5: Exact duplicate
        if text_stripped in seen_texts:
            issues.append("exact duplicate")

        return issues

    @staticmethod
    def _is_boilerplate(text: str) -> bool:
        """Detect table-of-contents, reference-list, or single-keyword chunks."""
        lines = text.strip().split("\n")
        if not lines:
            return False

        # Check if the only meaningful content is a boilerplate keyword
        # Strip heading context prefix first
        content = text
        if content.startswith("[") and "]\n" in content:
            content = content.split("]\n", 1)[1].strip()
        if _RE_BOILERPLATE_ONLY.match(content):
            return True

        # Check first few lines for TOC patterns
        header = "\n".join(lines[:3])
        if _RE_TOC.search(header):
            return True

        # Reference section: majority of lines start with [N] or N.
        ref_lines = sum(1 for line in lines if _RE_REFERENCES.match(line))
        if len(lines) > 3 and ref_lines / len(lines) > 0.6:
            return True

        return False

    @staticmethod
    def _is_mostly_numeric(text: str) -> bool:
        """Return True if >80% of non-whitespace characters are digits."""
        non_ws = re.sub(r"\s", "", text)
        if not non_ws:
            return False
        digit_count = sum(1 for c in non_ws if c.isdigit())
        return digit_count / len(non_ws) > 0.8

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(min_tokens={self.min_tokens})"
