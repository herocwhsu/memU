from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class FastEmbedClient:
    """Local embedding client backed by ``fastembed`` (ONNX, CPU-only).

    No network or API key required; suitable for hosts without an embedding
    provider (or without AVX, which rules out most local LLM runtimes). The
    model is loaded lazily on first use and cached for the process lifetime.
    """

    def __init__(self, *, embed_model: str | None = None, batch_size: int = 64):
        # Imported lazily so the optional ``fastembed`` dependency is only
        # required when this client is actually used.
        import fastembed  # noqa: F401

        self.embed_model = embed_model or _DEFAULT_MODEL
        self.batch_size = batch_size
        self._model: Any = None

    def _embed_sync(self, inputs: list[str]) -> list[list[float]]:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self.embed_model)
        return [vec.tolist() for vec in self._model.embed(inputs, batch_size=self.batch_size)]

    async def embed(self, inputs: list[str]) -> tuple[list[list[float]], Any]:
        if not inputs:
            return [], None
        vectors = await asyncio.to_thread(self._embed_sync, inputs)
        logger.debug("fastembed embedded %d inputs (dim %d)", len(vectors), len(vectors[0]))
        return vectors, None
