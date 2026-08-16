"""
neo4j_client.py — Neo4j Connection Manager
============================================

Manages the Neo4j driver lifecycle, provides query execution helpers
(read, write, batch), and creates the required indexes and constraints
including the **vector index** on ``Entity.embedding``.

Usage
-----
    >>> with Neo4jClient() as client:
    ...     client.execute_query("MATCH (n) RETURN count(n) AS c")
"""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv #type: ignore
from neo4j import Driver, GraphDatabase #type: ignore

from exceptions import GraphConstructionError

load_dotenv()
logger = logging.getLogger(__name__)


class Neo4jClient:
    
    def __init__(
        self,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ) -> None:
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.username = username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")

        if not self.password:
            raise GraphConstructionError("NEO4J_PASSWORD is not set.")

        self.driver: Driver = GraphDatabase.driver(
            self.uri, auth=(self.username, self.password)
        )
        logger.info("Neo4jClient initialised for %s.", self.uri)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------
    def verify_connection(self) -> None:
        """Verify that the driver can reach the Neo4j server.

        Raises
        ------
        GraphConstructionError
        """
        try:
            self.driver.verify_connectivity()
            logger.info("Neo4j connection verified.")
        except Exception as exc:
            raise GraphConstructionError(
                f"Neo4j connection failed: {exc}"
            ) from exc

    def close(self) -> None:
        """Close the Neo4j driver and release resources."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j driver closed.")

    def __enter__(self) -> Neo4jClient:
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Query execution
    # ------------------------------------------------------------------
    def execute_query(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query and return results as dicts.

        Parameters
        ----------
        query : str
        parameters : dict, optional

        Returns
        -------
        list[dict]
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            records = [record.data() for record in result]
            logger.debug(
                "Query returned %d record(s): %.120s", len(records), query
            )
            return records

    def execute_write(
        self, query: str, parameters: dict[str, Any] | None = None
    ) -> None:
        """Execute a write query.

        Parameters
        ----------
        query : str
        parameters : dict, optional
        """
        with self.driver.session(database=self.database) as session:
            session.run(query, parameters or {})
            logger.debug("Write executed: %.120s", query)

    def execute_batch(
        self,
        query: str,
        batch_params: list[dict[str, Any]],
        batch_size: int = 100,
    ) -> None:
        """Batch-write using the ``UNWIND $batch AS row`` pattern.

        Splits *batch_params* into chunks of *batch_size* and executes
        each batch in a separate transaction.

        Parameters
        ----------
        query : str
            Must contain ``$batch`` parameter for UNWIND.
        batch_params : list[dict]
        batch_size : int
        """
        total = len(batch_params)
        for i in range(0, total, batch_size):
            batch = batch_params[i : i + batch_size]
            with self.driver.session(database=self.database) as session:
                session.run(query, {"batch": batch})
            logger.debug(
                "Batch %d-%d / %d executed.",
                i,
                min(i + batch_size, total),
                total,
            )
        logger.info("Batch write complete — %d total rows.", total)

    # ------------------------------------------------------------------
    # Index / constraint creation
    # ------------------------------------------------------------------
    def create_indexes(self, embedding_dimension: int = 384) -> None:
        """Create required indexes, constraints, and the vector index.

        Parameters
        ----------
        embedding_dimension : int
            Dimension of entity embeddings (must match the model).
        """
        statements = [
            # Uniqueness constraint on Entity.name
            (
                "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
            ),
            # Index on Entity.type
            (
                "CREATE INDEX entity_type_index IF NOT EXISTS "
                "FOR (e:Entity) ON (e.type)"
            ),
            # Index on Chunk.chunk_id
            (
                "CREATE INDEX chunk_id_index IF NOT EXISTS "
                "FOR (c:Chunk) ON (c.chunk_id)"
            ),
        ]

        for stmt in statements:
            try:
                self.execute_write(stmt)
                logger.info("Index/constraint created: %.80s", stmt)
            except Exception as exc:
                logger.warning("Index creation skipped: %s", exc)

        # Vector index (separate due to OPTIONS syntax)
        try:
            vector_stmt = (
                "CREATE VECTOR INDEX entity_embedding_index IF NOT EXISTS "
                "FOR (e:Entity) ON (e.embedding) "
                "OPTIONS {indexConfig: {"
                f"`vector.dimensions`: {embedding_dimension}, "
                "`vector.similarity_function`: 'cosine'"
                "}}"
            )
            self.execute_write(vector_stmt)
            logger.info(
                "Vector index created — dim=%d.", embedding_dimension
            )
        except Exception as exc:
            logger.warning(
                "Vector index creation failed (Neo4j 5.11+ required): %s",
                exc,
            )

    def create_chunk_indexes(self) -> None:
        """Create indexes and constraints for the chunk-layer graph.

        Creates:
        - Uniqueness constraint on ``Chunk.chunk_id``.
        - Index on ``Chunk.chunk_index``.
        - Uniqueness constraint on ``Document.file_hash``.
        """
        statements = [
            (
                "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
                "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE"
            ),
            (
                "CREATE INDEX chunk_index_idx IF NOT EXISTS "
                "FOR (c:Chunk) ON (c.chunk_index)"
            ),
            (
                "CREATE CONSTRAINT doc_hash_unique IF NOT EXISTS "
                "FOR (d:Document) REQUIRE d.file_hash IS UNIQUE"
            ),
        ]

        for stmt in statements:
            try:
                self.execute_write(stmt)
                logger.info("Chunk index/constraint created: %.80s", stmt)
            except Exception as exc:
                logger.warning("Chunk index creation skipped: %s", exc)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(uri='{self.uri}')"