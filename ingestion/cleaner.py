"""
cleaner.py — Text Cleaning Pipeline
=====================================

Takes raw extracted text and returns cleaned, normalised text ready for
chunking.  All cleaning steps are composable static methods.

Cleaning preserves semantic structure (headings, paragraphs, lists) and
does **not** lowercase or remove stop words — entity casing must survive.
"""

from __future__ import annotations

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pre-compiled regex patterns
# ---------------------------------------------------------------------------
_RE_TABS = re.compile(r"\t")
_RE_MULTI_SPACE = re.compile(r"[ ]{2,}")
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")
_RE_ISOLATED_PAGE_NUM = re.compile(r"^\s*\d{1,5}\s*$", re.MULTILINE)
_RE_HEADER_FOOTER = re.compile(
    r"^\s*(?:page\s+\d+(?:\s+of\s+\d+)?|confidential|draft|©.*\d{4})\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_RE_HYPHENATION = re.compile(r"(\w+)-\n(?!-)(\w+)")
_RE_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class TextCleaner:
    """Clean raw extracted text while preserving semantic formatting.

    Cleaning Steps (in order)
    -------------------------
    1. Unicode normalisation   (NFKC)
    2. Control character removal (keep ``\\n``, ``\\t``)
    3. Broken hyphenation fix  (``knowl-\\nedge`` → ``knowledge``;
       but NOT markdown horizontal rules like ``---``)
    4. Tab replacement         (→ single space)
    5. Header/footer removal
    6. Isolated page-number removal
    7. Repeated-line removal   (same line appearing > 3 times)
    8. Inline whitespace collapse
    9. Blank-line collapse     (≥ 3 → 2)
    10. Empty-line removal
    11. Leading / trailing trim
    """

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Apply NFKC Unicode normalisation."""
        return unicodedata.normalize("NFKC", text)

    @staticmethod
    def remove_control_chars(text: str) -> str:
        """Remove null bytes and control characters (keep ``\\n``, ``\\t``)."""
        return _RE_CONTROL_CHARS.sub("", text)

    @staticmethod
    def fix_hyphenation(text: str) -> str:
        """Rejoin words broken across line boundaries."""
        return _RE_HYPHENATION.sub(r"\1\2", text)

    @staticmethod
    def remove_tabs(text: str) -> str:
        """Replace tab characters with a single space."""
        return _RE_TABS.sub(" ", text)

    @staticmethod
    def remove_header_footer_artifacts(text: str) -> str:
        """Remove common header/footer patterns."""
        return _RE_HEADER_FOOTER.sub("", text)

    @staticmethod
    def remove_page_numbers(text: str) -> str:
        """Remove isolated page-number lines."""
        return _RE_ISOLATED_PAGE_NUM.sub("", text)

    @staticmethod
    def remove_extra_spaces(text: str) -> str:
        """Collapse 2+ inline spaces to one."""
        return _RE_MULTI_SPACE.sub(" ", text)

    @staticmethod
    def remove_extra_newlines(text: str) -> str:
        """Collapse 3+ newlines to exactly two."""
        return _RE_MULTI_NEWLINE.sub("\n\n", text)

    @staticmethod
    def remove_empty_lines(text: str) -> str:
        """Remove lines that contain only whitespace."""
        return "\n".join(line for line in text.split("\n") if line.strip())

    @staticmethod
    def strip_text(text: str) -> str:
        """Strip leading / trailing whitespace."""
        return text.strip()

    @staticmethod
    def remove_repeated_lines(text: str, threshold: int = 3) -> str:
        """Remove lines that appear more than *threshold* times.

        These are typically repeated headers or footers injected per-page.
        """
        lines = text.split("\n")
        line_counts: dict[str, int] = {}
        for line in lines:
            stripped = line.strip()
            if stripped:
                line_counts[stripped] = line_counts.get(stripped, 0) + 1

        repeated = {ln for ln, cnt in line_counts.items() if cnt > threshold}
        if repeated:
            lines = [
                line for line in lines if line.strip() not in repeated
            ]
            logger.debug(
                "Removed %d repeated line patterns.", len(repeated)
            )
        return "\n".join(lines)

    def clean(self, raw_text: str) -> str:
        """Apply the full cleaning pipeline to raw text.

        Parameters
        ----------
        raw_text : str
            Raw text from the document loader.

        Returns
        -------
        str
            Cleaned text with semantic structure preserved.
        """
        text = self.normalize_unicode(raw_text)
        text = self.remove_control_chars(text)
        text = self.fix_hyphenation(text)
        text = self.remove_tabs(text)
        text = self.remove_header_footer_artifacts(text)
        text = self.remove_page_numbers(text)
        text = self.remove_repeated_lines(text)
        text = self.remove_extra_spaces(text)
        text = self.remove_extra_newlines(text)
        text = self.remove_empty_lines(text)
        text = self.strip_text(text)

        logger.info(
            "Cleaned text: %d → %d chars (removed %d).",
            len(raw_text),
            len(text),
            len(raw_text) - len(text),
        )
        return text

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
