# GraphRAG Knowledge Assistant – One Page Summary

## Prompting

| Field | Details |
|---------|---------|
| Decision | Used |
| Implementation | System prompts define the assistant behavior, extraction rules, and answer generation. User questions are combined with retrieved context before calling the LLM (`llama-3.3-70b-versatile` via Groq). |
| Justification | Keeps LLM responses focused on the provided document context and prevents made-up answers. |

---

## RAG (Retrieval Augmented Generation)

| Field | Details |
|---------|---------|
| Decision | Used |
| Implementation | Retrieves relevant text chunks, entities, and relationships from the indexed document and feeds them into the LLM prompt within a token limit. |
| Justification | Ensures answers are based on uploaded files rather than general LLM training data. |

---

## GraphRAG

| Field | Details |
|---------|---------|
| Decision | Used |
| Implementation | Combines Neo4j graph search and vector search. Searches entities, traverses connected graph nodes, and retrieves adjacent text chunks (`NEXT_CHUNK`). |
| Justification | Helps answer queries that require connecting related concepts across different sections of a document. |

---

## Chunking

| Field | Details |
|---------|---------|
| Decision | Used (Document-Structure Aware Chunking) |
| Implementation | Uses heading-aware and sentence-boundary chunking (`MarkdownChunker`). Packs max 500-token chunks (80-token overlap) by sentence boundaries without crossing markdown section headings, keeping tables atomic. |
| Justification | Avoids arbitrary cuts across sentences or tables, keeping text chunks small for search while preserving section context and readability. |

---

## Embeddings

| Field | Details |
|---------|---------|
| Decision | Used |
| Implementation | Converts entity descriptions and queries into 384-dimensional vector embeddings using `all-MiniLM-L6-v2`. |
| Justification | Enables search based on meaning and topic similarity instead of exact word matches. |

---

## Vector Database

| Field | Details |
|---------|---------|
| Decision | Used |
| Implementation | Stores vector embeddings directly inside Neo4j using its native vector index (`entity_embedding_index`). |
| Justification | Avoids running a separate vector database and keeps vectors alongside the graph data. |

---

## Knowledge Graph

| Field | Details |
|---------|---------|
| Decision | Used |
| Implementation | Neo4j graph database storing `Document`, `Chunk`, and `Entity` nodes with structural and dynamic relationships. |
| Justification | Links related concepts and source text chunks together for structured traversal and lookup. |

---

## Agentic AI

| Field | Details |
|---------|---------|
| Decision | Not Used |
| Implementation | Not implemented. |
| Justification | The system follows a fixed pipeline (Ingest → Search → Answer) and does not need autonomous planning or tool loops. |

---

## Fine-Tuning

| Field | Details |
|---------|---------|
| Decision | Not Used |
| Implementation | Pretrained Groq LLM and sentence-transformers models are used directly. |
| Justification | Prompting and graph retrieval deliver accurate results without model retraining. |

---

## Distillation

| Field | Details |
|---------|---------|
| Decision | Not Used |
| Implementation | Standard pretrained models are used as-is. |
| Justification | Pretrained cloud LLM speed and local embedding models meet performance needs without model compression. |

---

## LLMOps

| Field | Details |
|---------|---------|
| Decision | Used |
| Implementation | Automated API retries (`tenacity`), strict data schema validation (`pydantic`), typed error handling (`exceptions.py`), and pipeline logging (`graph_rag_pipeline.log`). |
| Justification | Handles API failures automatically, validates data before graph insertion, and provides clear log traces for system monitoring and debugging. |

