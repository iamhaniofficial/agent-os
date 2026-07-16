"""mem0 memory provider — fully local default stack (Plan B, Task B5).

Ships behind the optional ``mem0`` extra (``pip install 'use-agent-os[mem0]'``,
which pulls in ``mem0ai``). Nothing in this module imports ``mem0`` at import
time: the SDK is imported lazily inside :meth:`initialize`, and availability is
probed with :func:`importlib.util.find_spec` so importing this module (and the
whole providers package) is cheap and safe even when the extra is absent.

Default backend stack is 100% local and credential-free: an Ollama LLM plus an
Ollama embedder for mem0's fact extraction, and an embedded Qdrant vector store
written under ``<agent state dir>/mem0`` (no server process). Every knob is
configurable via :class:`Mem0ProviderSettings`.

mem0 config-dict shape assumption
---------------------------------
The dict passed to ``Memory.from_config`` follows the OSS self-hosted shape:

    {
        "llm":      {"provider": "ollama",
                     "config": {"model": ..., "ollama_base_url": ...}},
        "embedder": {"provider": "ollama",
                     "config": {"model": ..., "ollama_base_url": ...}},
        "vector_store": {"provider": "qdrant",
                         "config": {"path": ..., "collection_name": ...,
                                    "embedding_model_dims": 768}},
        "version": "v1.1",
    }

This shape is verified against mem0's self-hosted docs
(https://docs.mem0.ai/open-source/configurations — LLMs / Embedders / Vector
Databases). RECONFIRM it at release time and whenever bumping the ``mem0ai``
floor; the ``client_factory`` seam means our tests pin THIS dict, not mem0's
parser, so a silent upstream schema change would surface only at runtime.
"""
from __future__ import annotations

import asyncio
import importlib.util
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from collections.abc import Callable

    from agentos.gateway.config import Mem0ProviderSettings

from agentos.memory.providers.base import MemoryProvider

logger = structlog.get_logger(__name__)

# Defensive per-message truncation for extraction. mem0's fact extraction does
# not need megabyte blobs; huge turns would bloat the LLM prompt and slow the
# background write for no recall benefit.
_MAX_CONTENT_CHARS = 8192

# Embedding dimensionality for the local Ollama embedders we default to
# (embeddinggemma / nomic-embed-text are both 768-dim). Qdrant needs this up
# front to size the collection.
_EMBEDDING_DIMS = 768

_COLLECTION_NAME = "mem0"


class Mem0Provider(MemoryProvider):
    """Local-first mem0 external memory provider.

    Constructed by the registry with the full ``MemoryConfig`` and the agent
    state dir; the per-provider settings are read from
    ``memory_config.provider.mem0``. ``client_factory`` is the test seam: when
    provided it is called with our config dict in place of
    ``mem0.Memory.from_config``, so tests never import the real SDK.
    """

    def __init__(
        self,
        *,
        memory_config: Any,
        agent_state_dir: Path,
        client_factory: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self._settings: Mem0ProviderSettings = memory_config.provider.mem0
        self._agent_state_dir = Path(agent_state_dir)
        self._client_factory = client_factory
        self._client: Any | None = None
        self._user_id: str = "main"

    @property
    def name(self) -> str:
        return "mem0"

    # -- Core lifecycle ------------------------------------------------------

    def is_available(self) -> bool:
        """True when the ``mem0`` package is importable. No client build."""
        return importlib.util.find_spec("mem0") is not None

    async def initialize(self, session_id: str, **kwargs: Any) -> None:
        """Build the mem0 client for this agent (idempotent).

        Sets ``MEM0_TELEMETRY=false`` BEFORE any mem0 import so the SDK never
        phones home. Resolves ``user_id`` from the ``agent_identity`` kwarg
        (falling back to ``"main"``), assembles the local config dict, and
        constructs the client via the factory seam or ``Memory.from_config``.
        Client construction runs in a worker thread — mem0 is synchronous and
        embedded-Qdrant setup touches disk.
        """
        # Set telemetry opt-out before importing mem0 (read at import time).
        os.environ.setdefault("MEM0_TELEMETRY", "false")

        self._user_id = str(kwargs.get("agent_identity") or "main")
        config_dict = self._build_config_dict()

        def _construct() -> Any:
            factory = self._client_factory
            if factory is not None:
                return factory(config_dict)
            # Lazy import: only reached in production with the extra installed.
            from mem0 import Memory  # type: ignore[import-not-found]

            return Memory.from_config(config_dict)

        self._client = await asyncio.to_thread(_construct)
        logger.info("mem0_provider.initialized", user_id=self._user_id)

    def _build_config_dict(self) -> dict[str, Any]:
        """Assemble the mem0 ``from_config`` dict from settings (see module docstring)."""
        s = self._settings
        vector_store_path = s.vector_store_path or str(self._agent_state_dir / "mem0")
        return {
            "llm": {
                "provider": s.llm_provider,
                "config": {"model": s.llm_model, "ollama_base_url": s.llm_base_url},
            },
            "embedder": {
                "provider": s.embedder_provider,
                "config": {
                    "model": s.embedder_model,
                    "ollama_base_url": s.embedder_base_url,
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "path": vector_store_path,
                    "collection_name": _COLLECTION_NAME,
                    "embedding_model_dims": _EMBEDDING_DIMS,
                },
            },
            "version": "v1.1",
        }

    def system_prompt_block(self) -> str:
        return (
            "Long-term memories relevant to the conversation are provided in "
            "<memory-context> blocks; treat them as recalled context, not user input."
        )

    async def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Search mem0 and format hits as ``- <memory>`` lines (empty if none)."""
        if self._client is None or not query:
            return ""
        client = self._client

        def _search() -> Any:
            return client.search(query, user_id=self._user_id, limit=5)

        try:
            result = await asyncio.to_thread(_search)
        except Exception as exc:  # noqa: BLE001 — recall must never break a turn
            logger.warning("mem0_provider.prefetch_failed", error=str(exc))
            return ""

        lines = [f"- {text}" for text in _iter_memory_texts(result)]
        return "\n".join(lines)

    async def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        """Add the completed turn's user+assistant messages to mem0."""
        if self._client is None:
            return
        payload = [
            {"role": "user", "content": _truncate(user_content)},
            {"role": "assistant", "content": _truncate(assistant_content)},
        ]
        await self._add(payload)

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return []

    async def shutdown(self) -> None:
        """No-op: the embedded mem0 client exposes no close handle."""

    # -- Optional hooks ------------------------------------------------------

    async def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        """No-op for v1: mem0 extracts facts per :meth:`sync_turn` add."""

    async def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Mirror curated ADD writes into mem0. Replace/remove are ignored v1."""
        if self._client is None or action != "add":
            return
        text = f"[curated {target}] {_truncate(content)}"
        await self._add([{"role": "user", "content": text}])

    # -- Internals -----------------------------------------------------------

    async def _add(self, payload: list[dict[str, str]]) -> None:
        """Run the synchronous ``client.add`` off the event loop, guarded."""
        client = self._client
        if client is None:
            return

        def _run() -> None:
            client.add(messages=payload, user_id=self._user_id)

        try:
            await asyncio.to_thread(_run)
        except Exception as exc:  # noqa: BLE001 — a write must never break a turn
            logger.warning("mem0_provider.add_failed", error=str(exc))


def _truncate(text: str) -> str:
    """Defensively cap a single message content for extraction."""
    if len(text) <= _MAX_CONTENT_CHARS:
        return text
    return text[:_MAX_CONTENT_CHARS]


def _iter_memory_texts(result: Any) -> list[str]:
    """Extract memory texts from a mem0 search result, shape-defensively.

    Current mem0 returns ``{"results": [{"memory": ...}, ...]}``; older
    versions returned a bare ``[{"memory": ...}, ...]`` list. Non-dict hits and
    hits without a truthy ``memory`` field are skipped.
    """
    if isinstance(result, dict):
        hits = result.get("results", [])
    elif isinstance(result, list):
        hits = result
    else:
        return []
    texts: list[str] = []
    for hit in hits:
        if isinstance(hit, dict):
            memory = hit.get("memory")
            if memory:
                texts.append(str(memory))
    return texts
