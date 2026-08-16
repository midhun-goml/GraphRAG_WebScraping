"""
app.py — GraphRAG Pipeline Orchestrator
========================================

Orchestrates the full pipeline with three main flows:

- **Ingest**: PDF → Clean → Metadata → Chunk → Validate → Extract
  (LLM + entity resolution + embeddings) → Build Graph in Neo4j.
- **Ingest-Chunks**: PDF → Markdown → Clean → Metadata → Heading-Aware
  Chunk → Validate → chunks.json → Build Chunk Graph in Neo4j.
- **Query**: Search (graph + vector hybrid) → Build Context →
  Generate Answer.

CLI
---
::

    python app.py ingest data/scraper.pdf
    python app.py ingest-chunks data/scraper.pdf [--clear-graph]
    python app.py query "What is the main topic?"
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv #type: ignore

load_dotenv()

import warnings
warnings.filterwarnings("ignore", message=".*HF Hub.*")
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", category=FutureWarning)

_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

log_formatter = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.ERROR)
console_handler.setFormatter(log_formatter)

file_handler = logging.FileHandler(
    "graph_rag_pipeline.log", encoding="utf-8"
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(log_formatter)

root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("neo4j").setLevel(logging.WARNING)
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

logger = logging.getLogger("graph_rag")


# ======================================================================
# Ingest flow
# ======================================================================
def ingest(file_path: str) -> None:
    """Run the full ingestion pipeline: Markdown/PDF → Neo4j graph.

    This pipeline runs the complete end-to-end process:
    Load → Clean → Metadata → Chunk → Extract Knowledge → Build Graph

    Supports both:
      - PDF files: converts to markdown via Docling
      - Markdown files: reads directly (skips conversion)

    Parameters
    ----------
    file_path : str
        Path to the source PDF or markdown file.
    """
    import hashlib
    from extraction.extractor import KnowledgeExtractor
    from graph.builder import GraphBuilder
    from graph.neo4j_client import Neo4jClient
    from ingestion.chunker import TokenAwareChunker
    from ingestion.cleaner import TextCleaner
    from ingestion.loader import DocumentLoader
    from ingestion.metadata import MetadataExtractor
    from ingestion.models import DocumentMetadata
    from ingestion.validator import ChunkValidator
    from llm.generator import EmbeddingModel, LLMClient

    output_dir = os.getenv("OUTPUT_DIR", "output")

    logger.info("=" * 60)
    logger.info("  INGESTION PIPELINE START")
    logger.info("=" * 60)

    # 1. Load document (or Markdown)
    logger.info("[1/7] Loading document: %s", file_path)
    file_path_obj = Path(file_path).resolve()
    
    if file_path_obj.suffix.lower() == ".md":
        # Direct markdown input — skip PDF conversion
        logger.info("  → Reading markdown file directly (skipping PDF conversion)")
        if not file_path_obj.exists():
            raise FileNotFoundError(f"Markdown file not found: {file_path_obj}")
        
        raw_text = file_path_obj.read_text(encoding="utf-8")
        
        # Compute file hash for provenance
        file_bytes = file_path_obj.read_bytes()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        
        # Create minimal metadata
        metadata = DocumentMetadata(
            source_file=file_path_obj.name,
            file_hash=file_hash,
            markdown_path=str(file_path_obj),
        )
        logger.info("Markdown loaded (%d chars).", len(raw_text))
    else:
        # PDF input — convert via Docling
        loader = DocumentLoader()
        raw_text, metadata = loader.load(file_path)
        logger.info(
            "Loaded — %d chars, %d pages, hash=%s.",
            len(raw_text),
            metadata.page_count,
            metadata.file_hash[:12],
        )

    # 2. Clean text
    logger.info("[2/7] Cleaning text.")
    cleaner = TextCleaner()
    cleaned_text = cleaner.clean(raw_text)

    # 3. Enrich metadata
    logger.info("[3/7] Enriching metadata.")
    meta_extractor = MetadataExtractor()
    metadata = meta_extractor.extract(cleaned_text, metadata)

    # 4. Chunk
    logger.info("[4/7] Chunking text.")
    chunker = TokenAwareChunker()
    chunks = chunker.chunk(cleaned_text, metadata)
    logger.info("Produced %d chunk(s).", len(chunks))

    # 5. Validate chunks
    logger.info("[5/7] Validating chunks.")
    validator = ChunkValidator()
    chunks = validator.validate(chunks)
    logger.info("%d chunk(s) passed validation.", len(chunks))

    # 6. Extract knowledge
    logger.info("[6/7] Extracting knowledge (LLM + embeddings).")
    llm_client = LLMClient()
    embedding_model = EmbeddingModel()
    extractor = KnowledgeExtractor(
        llm_client, embedding_model, output_dir=output_dir
    )
    graph_data = extractor.extract_from_chunks(chunks)

    # 7. Build graph in Neo4j
    logger.info("[7/7] Building graph in Neo4j.")
    with Neo4jClient() as client:
        client.verify_connection()
        builder = GraphBuilder(client, clear_existing=True)
        builder.build_graph(graph_data)

    logger.info("=" * 60)
    logger.info("  INGESTION COMPLETE")
    logger.info(
        "  %d entities, %d relationships, %d chunks",
        len(graph_data.entities),
        len(graph_data.relationships),
        len(graph_data.chunks),
    )
    logger.info("=" * 60)


# ======================================================================
# Chunk-only ingest flow (new pipeline)
# ======================================================================
def ingest_chunks(file_path: str, clear_graph: bool = False) -> None:
    """Run the chunk-only ingestion pipeline: Markdown → Chunks → Neo4j.

    This pipeline does NOT run LLM extraction or entity resolution.
    It produces the chunk layer of the graph (`:Document` + `:Chunk`
    nodes with `:PART_OF` and `:NEXT_CHUNK` edges) and saves
    ``chunks.json`` to the output directory.

    Supports both:
      - PDF files: converts to markdown via Docling
      - Markdown files: reads directly (skips conversion)

    Parameters
    ----------
    file_path : str
        Path to the source PDF or markdown file.
    clear_graph : bool
        If ``True``, wipe the graph before building.
    """
    import hashlib
    from graph.builder import GraphBuilder
    from graph.neo4j_client import Neo4jClient
    from ingestion.chunker import MarkdownChunker
    from ingestion.cleaner import TextCleaner
    from ingestion.loader import DocumentLoader
    from ingestion.metadata import MetadataExtractor
    from ingestion.models import ChunksOutput, DocumentMetadata
    from ingestion.validator import ChunkValidator

    output_dir = os.getenv("OUTPUT_DIR", "output")
    chunk_size = int(os.getenv("CHUNK_SIZE_TOKENS", "512"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP_TOKENS", "64"))
    min_chunk_tokens = int(os.getenv("MIN_CHUNK_TOKENS", "30"))

    logger.info("=" * 60)
    logger.info("  CHUNK INGESTION PIPELINE START")
    logger.info("=" * 60)

    # ── Step 1: Load Markdown (or PDF → Markdown) ──
    logger.info("[1/7] Loading document: %s", file_path)
    file_path_obj = Path(file_path).resolve()
    
    if file_path_obj.suffix.lower() == ".md":
        # Direct markdown input — skip PDF conversion
        logger.info("  → Reading markdown file directly (skipping PDF conversion)")
        if not file_path_obj.exists():
            raise FileNotFoundError(f"Markdown file not found: {file_path_obj}")
        
        raw_markdown = file_path_obj.read_text(encoding="utf-8")
        
        # Compute file hash for provenance
        file_bytes = file_path_obj.read_bytes()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        
        # Create minimal metadata
        metadata = DocumentMetadata(
            source_file=file_path_obj.name,
            file_hash=file_hash,
            markdown_path=str(file_path_obj),
        )
        logger.info("Markdown loaded from: %s (%d chars)", file_path_obj, len(raw_markdown))
    else:
        # PDF input — convert via Docling
        loader = DocumentLoader(output_dir=output_dir)
        raw_markdown, metadata = loader.load(file_path)
        logger.info("Markdown saved to: %s", metadata.markdown_path)

    # ── Step 2: Clean markdown ──
    logger.info("[2/7] Cleaning text...")
    cleaner = TextCleaner()
    cleaned_text = cleaner.clean(raw_markdown)

    # ── Step 3: Enrich metadata ──
    logger.info("[3/7] Extracting metadata...")
    meta_extractor = MetadataExtractor()
    metadata = meta_extractor.extract(cleaned_text, metadata)
    logger.info(
        "Document: '%s' | %d words | %s",
        metadata.title,
        metadata.word_count,
        metadata.language,
    )

    # ── Step 4: Chunk ──
    logger.info(
        "[4/7] Chunking (size=%d, overlap=%d)...",
        chunk_size,
        chunk_overlap,
    )
    chunker = MarkdownChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = chunker.chunk(cleaned_text, metadata)
    logger.info("Created %d raw chunks.", len(chunks))

    # ── Step 5: Validate ──
    logger.info("[5/7] Validating chunks...")
    validator = ChunkValidator(min_tokens=min_chunk_tokens)
    chunks = validator.validate(chunks)
    logger.info("After validation: %d chunks.", len(chunks))

    # ── Step 6: Serialize to chunks.json ──
    logger.info("[6/7] Serializing to chunks.json...")
    chunks_output = ChunksOutput(
        document_metadata=metadata,
        total_chunks=len(chunks),
        chunks=chunks,
    )
    output_path = Path(output_dir) / "chunks.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        chunks_output.model_dump_json(indent=2), encoding="utf-8"
    )
    logger.info("Chunks saved to: %s", output_path)

    # ── Step 7: Build Neo4j graph ──
    logger.info("[7/7] Building chunk graph in Neo4j...")
    with Neo4jClient() as client:
        client.verify_connection()

        if clear_graph:
            logger.warning("Clearing existing graph data...")
            client.execute_write("MATCH (n) DETACH DELETE n")

        builder = GraphBuilder(client)
        builder.build_chunk_graph(chunks_output)

    logger.info("=" * 60)
    logger.info("  CHUNK INGESTION COMPLETE")
    logger.info(
        "  %d chunks | Document: '%s'",
        chunks_output.total_chunks,
        metadata.title,
    )
    logger.info("=" * 60)


# ======================================================================
# Query flow
# ======================================================================
def query(question: str) -> str:
    """Run the full query pipeline: question → grounded answer.

    Parameters
    ----------
    question : str

    Returns
    -------
    str
        The generated answer.
    """
    from extraction.prompts import (
        GENERATION_SYSTEM_PROMPT,
        build_answer_generation_prompt,
    )
    from graph.neo4j_client import Neo4jClient
    from llm.generator import EmbeddingModel, LLMClient
    from retrieval.context_builder import ContextBuilder
    from retrieval.graph_search import GraphSearchEngine

    llm_client = LLMClient()
    embedding_model = EmbeddingModel()

    with Neo4jClient() as client:
        client.verify_connection()

        # 1. Search the graph
        searcher = GraphSearchEngine(client, llm_client, embedding_model)
        search_results = searcher.search(question)

        # 2. Build context
        context_builder = ContextBuilder(max_context_tokens=2500)
        context = context_builder.build_context(search_results, question)

        # 3. Generate answer
        answer_prompt = build_answer_generation_prompt(question, context)
        answer = llm_client.generate(
            answer_prompt, system_prompt=GENERATION_SYSTEM_PROMPT
        )

    return answer


# ======================================================================
# Interactive Q&A loop
# ======================================================================
def qa_loop() -> None:
    """Start an interactive Q&A terminal session."""
    from extraction.prompts import (
        GENERATION_SYSTEM_PROMPT,
        build_answer_generation_prompt,
    )
    from graph.neo4j_client import Neo4jClient
    from llm.generator import EmbeddingModel, LLMClient
    from retrieval.context_builder import ContextBuilder
    from retrieval.graph_search import GraphSearchEngine

    llm_client = LLMClient()
    embedding_model = EmbeddingModel()

    with Neo4jClient() as client:
        client.verify_connection()

        searcher = GraphSearchEngine(client, llm_client, embedding_model)
        context_builder = ContextBuilder(max_context_tokens=2500)

        print("\n" + "=" * 60)
        print("  GraphRAG Q&A System — Hybrid Graph + Vector Retrieval")
        print("  Type 'exit' or 'quit' to close.")
        print("=" * 60 + "\n")

        while True:
            try:
                question = input("\nQuery > ").strip()
                if not question:
                    continue
                if question.lower() in ("exit", "quit"):
                    print("Goodbye!")
                    break

                # 1. Search
                search_results = searcher.search(question)

                logger.info(
                    "Retrieved: %d entities, %d rels, %d chunks.",
                    len(search_results["entities"]),
                    len(search_results["relationships"]),
                    len(search_results["chunks"]),
                )

                # 2. Build context
                context = context_builder.build_context(
                    search_results, question
                )

                # 3. Generate answer
                prompt = build_answer_generation_prompt(question, context)
                answer = llm_client.generate(
                    prompt, system_prompt=GENERATION_SYSTEM_PROMPT
                )

                print("\nAnswer:\n" + "-" * 60)
                print(answer)
                print("-" * 60 + "\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as exc:
                logger.error("Error during Q&A: %s", exc)


# ======================================================================
# CLI Entry Point
# ======================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python app.py ingest <pdf_path>")
        print("  python app.py ingest-chunks <pdf_path> [--clear-graph]")
        print("  python app.py query <question>")
        print("  python app.py qa")
        sys.exit(1)

    command = sys.argv[1].lower()

    try:
        if command == "ingest":
            if len(sys.argv) < 3:
                print("Error: Please provide a PDF path.")
                print("  python app.py ingest data/scraper.pdf")
                sys.exit(1)
            ingest(sys.argv[2])

        elif command == "ingest-chunks":
            if len(sys.argv) < 3:
                print("Error: Please provide a PDF path.")
                print("  python app.py ingest-chunks data/scraper.pdf [--clear-graph]")
                sys.exit(1)
            clear = "--clear-graph" in sys.argv
            ingest_chunks(sys.argv[2], clear_graph=clear)

        elif command == "query":
            if len(sys.argv) < 3:
                print("Error: Please provide a question.")
                print('  python app.py query "What is web scraping?"')
                sys.exit(1)
            question_text = " ".join(sys.argv[2:])
            answer = query(question_text)
            print("\nAnswer:\n" + "-" * 60)
            print(answer)
            print("-" * 60)

        elif command == "qa":
            qa_loop()

        else:
            print(f"Unknown command: {command}")
            print("Valid commands: ingest, ingest-chunks, query, qa")
            sys.exit(1)

    except Exception as e:
        logger.critical("Pipeline aborted: %s", e, exc_info=True)
        sys.exit(1)
