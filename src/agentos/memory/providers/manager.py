"""MemoryProviderManager — orchestrates a single external memory provider.

Adapted from hermes-agent ``agent/memory_manager.py`` (MIT). AgentOS runs one
async engine, so this is the async counterpart of hermes' thread-based
manager, with these deliberate adaptations:

* **One provider maximum.** AgentOS's native curated memory lives OUTSIDE this
  layer (``agentos.memory``), so there is no "builtin" provider to register
  first. Only a single external provider is accepted; a second
  :meth:`add_provider` is rejected (log warning, return ``False``).
* **Background = one asyncio task, not threads.** Latency-bound provider work
  (``sync_turn``, ``queue_prefetch``) is serialized through a single
  ``asyncio.Queue`` consumed by one lazily-created task. FIFO ordering means
  turn N lands before turn N+1. A provider exception is caught and logged in
  the consumer; it never propagates to the caller and never wedges the queue.
* **flush_pending** awaits an ``asyncio.Event`` sentinel enqueued behind the
  current work — when it fires, everything submitted before it has run.
* **shutdown** stops accepting work, drains what's queued within a 5s bound
  (mirrors hermes ``_SYNC_DRAIN_TIMEOUT_S``), cancels the consumer, then awaits
  ``provider.shutdown()`` (guarded).
* **Reserved tool names.** AgentOS exposes builtin tool names only at runtime
  via a ``ToolRegistry`` instance — there is no static "all builtin tool
  names" import to pull. Rather than importing engine internals, the manager
  accepts an optional ``reserved_tool_names`` set (the caller — boot wiring,
  Task B3 — supplies the live registry's names). A provider tool whose name
  collides with a reserved name is skipped with a warning, mirroring hermes'
  ``_HERMES_CORE_TOOLS`` shadow check (builtins always win).
* **structlog** logging throughout.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

import structlog

from agentos.memory.providers.base import MemoryProvider
from agentos.memory.providers.fencing import build_memory_context_block

logger = structlog.get_logger(__name__)

# How long shutdown() waits for queued background work to drain before
# abandoning it. A wedged provider must never block engine teardown.
_SYNC_DRAIN_TIMEOUT_S = 5.0

_BackgroundTask = Callable[[], Awaitable[None]]


def _normalize_tool_schema(schema: Any) -> dict[str, Any] | None:
    """Return a bare function-tool dict with a resolvable top-level ``name``.

    Accepts either a bare function schema (``{"name": ...}``) or an entry
    already in OpenAI tool form (``{"type": "function", "function": {...}}``),
    unwrapping the latter. Returns ``None`` for anything without a usable
    string name so callers skip-with-warning rather than routing a nameless
    tool (ported from hermes ``normalize_tool_schema``).
    """
    if not isinstance(schema, dict):
        return None
    if schema.get("type") == "function" and isinstance(schema.get("function"), dict):
        inner = schema["function"]
        if not isinstance(inner, dict):
            return None
        schema = inner
    name = schema.get("name", "")
    if not name or not isinstance(name, str):
        return None
    result: dict[str, Any] = schema
    return result


def _tool_error(message: str) -> str:
    """Return a JSON tool-error string matching AgentOS's tool result shape."""
    return json.dumps({"success": False, "error": message})


class MemoryProviderManager:
    """Orchestrates at most one external memory provider.

    Provider failures never propagate to the caller. Background writes are
    serialized FIFO through a single consumer task so turn N lands before N+1.
    """

    def __init__(self, *, reserved_tool_names: set[str] | None = None) -> None:
        self._provider: MemoryProvider | None = None
        self._tool_to_provider: dict[str, MemoryProvider] = {}
        self._reserved_tool_names: set[str] = set(reserved_tool_names or set())

        # Background dispatch: one queue + one lazily-created consumer task.
        self._queue: asyncio.Queue[_BackgroundTask] | None = None
        self._consumer: asyncio.Task[None] | None = None
        self._accepting: bool = True

    # -- Registration --------------------------------------------------------

    def add_provider(self, provider: MemoryProvider) -> bool:
        """Register the external memory provider.

        At most ONE provider is allowed. A second registration is rejected
        (warning logged, ``False`` returned). Tool names colliding with a
        reserved (builtin) tool name are skipped with a warning — builtins
        always win. Returns ``True`` on success.
        """
        if self._provider is not None:
            logger.warning(
                "memory_provider.rejected_second",
                rejected=provider.name,
                active=self._provider.name,
            )
            return False

        self._provider = provider

        routed = 0
        for raw_schema in provider.get_tool_schemas():
            schema = _normalize_tool_schema(raw_schema)
            if schema is None:
                logger.warning("memory_provider.tool_schema_no_name", provider=provider.name)
                continue
            tool_name = schema["name"]
            if tool_name in self._reserved_tool_names:
                logger.warning(
                    "memory_provider.tool_name_reserved",
                    provider=provider.name,
                    tool=tool_name,
                )
                continue
            if tool_name in self._tool_to_provider:
                logger.warning(
                    "memory_provider.tool_name_conflict",
                    provider=provider.name,
                    tool=tool_name,
                )
                continue
            self._tool_to_provider[tool_name] = provider
            routed += 1

        logger.info("memory_provider.registered", provider=provider.name, tools=routed)
        return True

    @property
    def active_provider(self) -> MemoryProvider | None:
        """The registered provider, or ``None`` if none is active."""
        return self._provider

    # -- System prompt -------------------------------------------------------

    def build_system_prompt(self) -> str:
        """Return the provider's static system-prompt block, or ``""``."""
        if self._provider is None:
            return ""
        try:
            block = self._provider.system_prompt_block()
        except Exception:
            logger.warning("memory_provider.system_prompt_failed", provider=self._provider.name)
            return ""
        return block if block and block.strip() else ""

    # -- Lifecycle -----------------------------------------------------------

    async def initialize_all(self, session_id: str, **kwargs: Any) -> None:
        """Initialize the provider. Failures are logged, not raised."""
        if self._provider is None:
            return
        try:
            await self._provider.initialize(session_id, **kwargs)
        except Exception:
            logger.warning("memory_provider.initialize_failed", provider=self._provider.name)

    async def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        """Forward end-of-session extraction to the provider (guarded)."""
        if self._provider is None:
            return
        try:
            await self._provider.on_session_end(messages)
        except Exception:
            logger.warning("memory_provider.on_session_end_failed", provider=self._provider.name)

    async def on_session_switch(self, new_session_id: str, **kwargs: Any) -> None:
        """Forward a mid-process session_id rotation to the provider (guarded)."""
        if self._provider is None or not new_session_id:
            return
        try:
            await self._provider.on_session_switch(new_session_id, **kwargs)
        except Exception:
            logger.warning(
                "memory_provider.on_session_switch_failed", provider=self._provider.name
            )

    # -- Prefetch / recall ---------------------------------------------------

    async def prefetch_all(self, query: str, *, session_id: str = "") -> str:
        """Recall context from the provider, fenced via the memory-context block.

        Returns the wrapped block, or ``""`` if there's no provider, no
        result, or the provider raised (recall is best-effort).
        """
        if self._provider is None:
            return ""
        try:
            result = await self._provider.prefetch(query, session_id=session_id)
        except Exception:
            logger.debug("memory_provider.prefetch_failed", provider=self._provider.name)
            return ""
        if not result or not result.strip():
            return ""
        return build_memory_context_block(result)

    def queue_prefetch_all(self, query: str, *, session_id: str = "") -> None:
        """Queue the provider's background recall for the next turn (FIFO)."""
        provider = self._provider
        if provider is None:
            return

        async def _run() -> None:
            await provider.queue_prefetch(query, session_id=session_id)

        self._submit_background(_run)

    # -- Sync ----------------------------------------------------------------

    def sync_all(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        """Persist a completed turn to the provider on the background worker.

        Never blocks the turn path; writes are serialized FIFO so turn N lands
        before turn N+1.
        """
        provider = self._provider
        if provider is None:
            return

        async def _run() -> None:
            await provider.sync_turn(
                user_content,
                assistant_content,
                session_id=session_id,
                messages=messages,
            )

        self._submit_background(_run)

    def notify_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Mirror a curated memory write to the provider on the background worker."""
        provider = self._provider
        if provider is None:
            return

        async def _run() -> None:
            await provider.on_memory_write(action, target, content, metadata=metadata)

        self._submit_background(_run)

    # -- Tools ---------------------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Return normalized tool schemas the provider routes (reserved skipped)."""
        schemas: list[dict[str, Any]] = []
        seen: set[str] = set()
        for name, provider in self._tool_to_provider.items():
            if name in seen:
                continue
            for raw_schema in provider.get_tool_schemas():
                schema = _normalize_tool_schema(raw_schema)
                if schema is not None and schema["name"] == name:
                    schemas.append(schema)
                    seen.add(name)
                    break
        return schemas

    def has_tool(self, name: str) -> bool:
        """Return whether the provider handles ``name``."""
        return name in self._tool_to_provider

    async def handle_tool_call(self, name: str, args: dict[str, Any], **kwargs: Any) -> str:
        """Route a tool call to the provider. Returns a JSON string result.

        Unknown tools and provider exceptions return a JSON error string
        rather than raising (a memory tool must never crash the turn).
        """
        provider = self._tool_to_provider.get(name)
        if provider is None:
            return _tool_error(f"No memory provider handles tool '{name}'")
        try:
            return await provider.handle_tool_call(name, args, **kwargs)
        except Exception as exc:
            logger.warning(
                "memory_provider.tool_call_failed", provider=provider.name, tool=name
            )
            return _tool_error(f"Memory tool '{name}' failed: {exc}")

    # -- Background dispatch -------------------------------------------------

    def _submit_background(self, task: _BackgroundTask) -> None:
        """Enqueue ``task`` on the FIFO worker, creating it on first use.

        Silently drops the task once shutdown has stopped accepting work.
        """
        if not self._accepting:
            logger.debug("memory_provider.submit_after_shutdown")
            return
        queue = self._ensure_worker()
        queue.put_nowait(task)

    def _ensure_worker(self) -> asyncio.Queue[_BackgroundTask]:
        """Lazily create the queue + single consumer task."""
        if self._queue is None:
            self._queue = asyncio.Queue()
        if self._consumer is None or self._consumer.done():
            self._consumer = asyncio.create_task(self._consume(self._queue))
        return self._queue

    async def _consume(self, queue: asyncio.Queue[_BackgroundTask]) -> None:
        """Run queued tasks FIFO, catching and logging every exception."""
        while True:
            task = await queue.get()
            try:
                await task()
            except Exception:
                provider = self._provider.name if self._provider else "unknown"
                logger.warning("memory_provider.sync_failed", provider=provider)
            finally:
                queue.task_done()

    async def flush_pending(self, timeout: float | None = None) -> bool:
        """Block until queued background work has drained.

        Enqueues an event-set sentinel behind the current work; when it fires,
        everything submitted before it has run. Returns ``True`` if the barrier
        completed within ``timeout`` (or there was no work), ``False`` on
        timeout.
        """
        if self._queue is None or self._consumer is None:
            return True

        done = asyncio.Event()

        async def _sentinel() -> None:
            done.set()

        self._queue.put_nowait(_sentinel)
        try:
            await asyncio.wait_for(done.wait(), timeout=timeout)
            return True
        except TimeoutError:
            return False

    async def shutdown(self) -> None:
        """Stop accepting work, drain (≤5s), cancel the consumer, shut provider.

        A wedged provider can delay drain by at most ``_SYNC_DRAIN_TIMEOUT_S``;
        anything still queued past that is abandoned so teardown never hangs.
        """
        self._accepting = False

        if self._queue is not None and self._consumer is not None:
            try:
                await asyncio.wait_for(self.flush_pending(), timeout=_SYNC_DRAIN_TIMEOUT_S)
            except TimeoutError:
                logger.warning("memory_provider.shutdown_drain_timeout")

        if self._consumer is not None:
            self._consumer.cancel()
            try:
                await self._consumer
            except asyncio.CancelledError:
                pass
            self._consumer = None
        self._queue = None

        if self._provider is not None:
            try:
                await self._provider.shutdown()
            except Exception:
                logger.warning("memory_provider.shutdown_failed", provider=self._provider.name)
