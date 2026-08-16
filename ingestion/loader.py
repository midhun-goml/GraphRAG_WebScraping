"""
loader.py — Docling-Based Document Loader
==========================================

Accepts a PDF file path, uses the Docling ``DocumentConverter`` to parse
the document, and returns the full extracted **markdown** as a single
string alongside a ``DocumentMetadata`` object containing provenance
information.

The loader:
  - Converts the PDF to Markdown via Docling.
  - Saves the ``.md`` file to ``{output_dir}/{stem}.md``.
  - Performs **no cleaning and no chunking** — it is a
    single-responsibility component.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from docling.datamodel.pipeline_options import PdfPipelineOptions #type: ignore
from docling.document_converter import DocumentConverter, PdfFormatOption #type: ignore

from exceptions import DocumentLoadError
from ingestion.models import DocumentMetadata

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Load a PDF document using Docling and return markdown text + metadata.

    Parameters
    ----------
    output_dir : str
        Directory where the converted ``.md`` file is saved.
    enable_ocr : bool
        Whether to enable OCR for scanned pages (default ``False``).
    """

    def __init__(
        self, output_dir: str = "output", enable_ocr: bool = False
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = enable_ocr

        self._converter = DocumentConverter(
            format_options={
                "pdf": PdfFormatOption(pipeline_options=pipeline_options),
            }
        )
        logger.info(
            "DocumentLoader initialised (output_dir=%s, ocr=%s).",
            output_dir,
            enable_ocr,
        )

    def load(self, file_path: str) -> tuple[str, DocumentMetadata]:
        """Parse a PDF and return ``(markdown_text, metadata)``.

        The markdown is also saved to ``{output_dir}/{stem}.md``.

        Parameters
        ----------
        file_path : str
            Filesystem path to the PDF file.

        Returns
        -------
        tuple[str, DocumentMetadata]

        Raises
        ------
        DocumentLoadError
            If the file does not exist, is not a PDF, or cannot be parsed.
        """
        path = Path(file_path).resolve()

        if not path.exists():
            raise DocumentLoadError(f"File not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise DocumentLoadError(
                f"Expected a .pdf file, got '{path.suffix}': {path}"
            )

        # Compute SHA-256 hash
        file_hash = self._hash_file(path)

        try:
            result = self._converter.convert(str(path))
            doc = result.document
        except Exception as exc:
            raise DocumentLoadError(
                f"Docling failed to parse '{path.name}': {exc}"
            ) from exc

        # Extract full text as markdown
        markdown_text = doc.export_to_markdown()
        if not markdown_text or not markdown_text.strip():
            logger.warning("No text extracted from '%s'.", path.name)
            markdown_text = ""

        # Save markdown to disk
        md_path = self.output_dir / f"{path.stem}.md"
        md_path.write_text(markdown_text, encoding="utf-8")
        logger.info("Markdown saved to: %s", md_path)

        # Build metadata from Docling document
        if hasattr(doc, "num_pages") and callable(doc.num_pages):
            page_count = doc.num_pages()
        elif hasattr(doc, "pages") and isinstance(doc.pages, dict):
            page_count = len(doc.pages)
        else:
            page_count = 0

        title: str | None = None
        author: str | None = None
        creation_date: str | None = None

        # Attempt to read metadata from docling document properties
        doc_meta = getattr(doc, "metadata", None) or {}
        if isinstance(doc_meta, dict):
            title = doc_meta.get("title")
            author = doc_meta.get("author")
            creation_date = doc_meta.get("creation_date")

        metadata = DocumentMetadata(
            source_file=path.name,
            title=title,
            author=author,
            page_count=page_count,
            creation_date=creation_date,
            file_hash=file_hash,
            word_count=0,         # enriched later by MetadataExtractor
            language="",          # enriched later by MetadataExtractor
            markdown_path=str(md_path),
        )

        logger.info(
            "Loaded '%s' — %d pages, %d chars, hash=%s.",
            path.name,
            metadata.page_count,
            len(markdown_text),
            file_hash[:12],
        )
        return markdown_text, metadata

    @staticmethod
    def _hash_file(path: Path) -> str:
        """Compute SHA-256 hex digest of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(output_dir='{self.output_dir}')"