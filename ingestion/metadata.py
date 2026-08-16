"""
metadata.py — Document Metadata Enrichment
============================================

Takes cleaned text and base ``DocumentMetadata`` from the loader, and
enriches it with information derivable from the text itself (inferred
title, word count, language detection).
"""

from __future__ import annotations

import logging
import re

from ingestion.models import DocumentMetadata

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Enrich document metadata with text-derived information.

    Responsibilities
    ----------------
    - Infer the document title from the first ``# H1`` heading if missing.
    - Estimate word count.
    - Detect language (graceful fallback to ``"en"``).
    """

    def extract(
        self, text: str, base_metadata: DocumentMetadata
    ) -> DocumentMetadata:
        """Enrich *base_metadata* with information derived from *text*.

        Parameters
        ----------
        text : str
            Cleaned document text.
        base_metadata : DocumentMetadata
            Metadata produced by the loader.

        Returns
        -------
        DocumentMetadata
            A copy of the input metadata with enriched fields.
        """
        updates: dict = {}

        # Infer title from first H1 heading or first non-empty line
        if not base_metadata.title:
            inferred = self._infer_title(text)
            if inferred:
                updates["title"] = inferred
                logger.info("Inferred title: '%s'.", inferred)

        # Compute word count
        word_count = len(text.split())
        updates["word_count"] = word_count

        # Detect language
        language = self._detect_language(text)
        updates["language"] = language

        enriched = base_metadata.model_copy(update=updates)

        logger.info(
            "Metadata enriched — title='%s', words=%d, lang='%s', pages=%d.",
            enriched.title,
            enriched.word_count,
            enriched.language,
            enriched.page_count,
        )
        return enriched

    @staticmethod
    def _infer_title(text: str) -> str | None:
        """Return the document title.

        Prefers the first ``# H1`` heading.  Falls back to the first
        non-empty, non-trivial line.
        """
        # First pass: look for an H1 heading (# ...)
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("# ") and not stripped.startswith("## "):
                title = stripped.lstrip("#").strip()
                if 3 <= len(title) <= 200:
                    return title

        # Fallback: first non-empty, non-trivial line
        for line in text.split("\n"):
            stripped = line.strip().lstrip("#").strip()
            # Skip HTML comments / image placeholders
            if stripped.startswith("<!--") or stripped.endswith("-->"):
                continue
            if stripped and 3 <= len(stripped) <= 200:
                return stripped
        return None

    @staticmethod
    def _detect_language(text: str) -> str:
        """Detect the language of *text*.

        Uses ``langdetect`` if available, otherwise defaults to ``"en"``.
        Only examines the first 2000 characters for performance.
        """
        try:
            from langdetect import detect  # type: ignore[import-untyped]

            return detect(text[:2000])
        except ImportError:
            logger.debug(
                "langdetect not installed — defaulting to 'en'."
            )
            return "en"
        except Exception as exc:
            logger.debug("Language detection failed: %s — defaulting to 'en'.", exc)
            return "en"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
