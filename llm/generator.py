"""
generator.py — LLM Client & Embedding Model
=============================================

Wraps the Groq OpenAI-compatible API into ``LLMClient`` for text and
JSON generation, and wraps ``sentence-transformers`` into
``EmbeddingModel`` for entity vector embeddings.

Both classes are injected into downstream modules (extraction, retrieval)
rather than instantiated ad-hoc.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from dotenv import load_dotenv #type: ignore
from openai import OpenAI #type: ignore
from tenacity import (
    retry,
    retry_if_exception_type, 
    stop_after_attempt,
    wait_exponential,
)

from exceptions import LLMGenerationError, LLMJsonParseError

load_dotenv()
logger = logging.getLogger(__name__)

_RE_CODE_FENCE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


# ======================================================================
# LLMClient — Groq wrapper
# ======================================================================
class LLMClient:
    """Groq-compatible LLM client for text and JSON generation.

    Parameters
    ----------
    api_key : str | None
        Groq API key. Falls back to ``GROQ_API_KEY`` env var.
    model_name : str | None
        Model identifier. Falls back to ``LLM_MODEL_NAME`` env var.
    temperature : float | None
        Sampling temperature. Falls back to ``LLM_TEMPERATURE`` env var.
    max_output_tokens : int | None
        Max tokens to generate. Falls back to ``LLM_MAX_OUTPUT_TOKENS``.
    """

    BASE_URL: str = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        if not self.api_key:
            raise LLMGenerationError("GROQ_API_KEY is not set.")

        self.model_name = model_name or os.getenv(
            "LLM_MODEL_NAME", "llama-3.3-70b-versatile"
        )
        self.temperature = (
            temperature
            if temperature is not None
            else float(os.getenv("LLM_TEMPERATURE", "0.0"))
        )
        self.max_output_tokens = max_output_tokens or int(
            os.getenv("LLM_MAX_OUTPUT_TOKENS", "8192")
        )

        self.client = OpenAI(api_key=self.api_key, base_url=self.BASE_URL)

        logger.info(
            "LLMClient initialised — model='%s', temp=%.1f.",
            self.model_name,
            self.temperature,
        )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate a text response from the LLM.

        Parameters
        ----------
        prompt : str
            User prompt.
        system_prompt : str
            Optional system prompt.

        Returns
        -------
        str

        Raises
        ------
        LLMGenerationError
            On empty responses or safety blocks.
        """
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                messages=messages,
            )
            content = response.choices[0].message.content
            if not content:
                raise LLMGenerationError("LLM returned an empty response.")
            return content.strip()

        except LLMGenerationError:
            raise
        except Exception as exc:
            raise LLMGenerationError(f"LLM call failed: {exc}") from exc

    def generate_json(
        self, prompt: str, system_prompt: str = ""
    ) -> dict[str, Any]:
        """Generate a JSON response, stripping code fences if present.

        Parameters
        ----------
        prompt : str
        system_prompt : str

        Returns
        -------
        dict

        Raises
        ------
        LLMJsonParseError
            If the response cannot be parsed as JSON.
        """
        raw = self.generate(prompt, system_prompt)

        # Strip markdown code fences
        match = _RE_CODE_FENCE.search(raw)
        if match:
            raw = match.group(1)

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMJsonParseError(
                f"Failed to parse LLM JSON response: {exc}\n"
                f"Raw response (first 500 chars): {raw[:500]}"
            ) from exc

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model='{self.model_name}')"


# ======================================================================
# EmbeddingModel — sentence-transformers wrapper
# ======================================================================
class EmbeddingModel:
    """Sentence-transformer embedding model for entity vectors.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    """

    def __init__(self, model_name: str | None = None) -> None:
        import contextlib
        import io

        from sentence_transformers import SentenceTransformer #type: ignore

        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"
        )
        with (
            contextlib.redirect_stderr(io.StringIO()),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self._model = SentenceTransformer(self.model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()

        logger.info(
            "EmbeddingModel loaded — model='%s', dim=%d.",
            self.model_name,
            self._dimension,
        )

    def embed(self, text: str) -> list[float]:
        """Embed a single text string.

        Parameters
        ----------
        text : str

        Returns
        -------
        list[float]
        """
        vec = self._model.encode(text, convert_to_numpy=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed multiple text strings.

        Parameters
        ----------
        texts : list[str]

        Returns
        -------
        list[list[float]]
        """
        vecs = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return [v.tolist() for v in vecs]

    def get_dimension(self) -> int:
        """Return the embedding vector dimension."""
        return self._dimension

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model='{self.model_name}', dim={self._dimension})"
        )
