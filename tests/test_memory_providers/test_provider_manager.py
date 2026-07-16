"""Tests for MemoryProviderManager (Task B2).

Deterministic and offline: fake async providers, no sleeps for ordering —
``flush_pending`` is the barrier that guarantees enqueued background work has
run before assertions.
"""
from __future__ import annotations

import asyncio
from typing import Any

from agentos.memory.providers.base import MemoryProvider
from agentos.memory.providers.manager import MemoryProviderManager

OPEN = "<memory-context>"
CLOSE = "</memory-context>"


class _FakeProvider(MemoryProvider):
    """Recording fake provider with configurable behavior."""

    def __init__(
        self,
        name: str = "fake",
        *,
        prefetch_result: str = "",
        tool_schemas: list[dict[str, Any]] | None = None,
        sync_raises: bool = False,
    ) -> None:
        self._name = name
        self._prefetch_result = prefetch_result
        self._tool_schemas = tool_schemas or []
        self._sync_raises = sync_raises
        self.sync_calls: list[tuple[str, str]] = []
        self.queue_prefetch_calls: list[str] = []
        self.tool_calls: list[tuple[str, dict[str, Any]]] = []
        self.shutdown_called = False
        self.initialized_with: tuple[str, dict[str, Any]] | None = None

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return True

    async def initialize(self, session_id: str, **kwargs: Any) -> None:
        self.initialized_with = (session_id, kwargs)

    async def prefetch(self, query: str, *, session_id: str = "") -> str:
        return self._prefetch_result

    async def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        self.queue_prefetch_calls.append(query)

    async def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        if self._sync_raises:
            raise RuntimeError("boom")
        self.sync_calls.append((user_content, assistant_content))

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return self._tool_schemas

    async def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs: Any) -> str:
        self.tool_calls.append((tool_name, args))
        return f'{{"ok": "{tool_name}"}}'

    async def shutdown(self) -> None:
        self.shutdown_called = True


# -- registration -------------------------------------------------------------


def test_add_first_provider_accepted() -> None:
    mgr = MemoryProviderManager()
    assert mgr.add_provider(_FakeProvider()) is True
    assert mgr.active_provider is not None


def test_second_add_provider_rejected() -> None:
    mgr = MemoryProviderManager()
    first = _FakeProvider("first")
    assert mgr.add_provider(first) is True
    assert mgr.add_provider(_FakeProvider("second")) is False
    assert mgr.active_provider is first


def test_no_provider_active_provider_none() -> None:
    assert MemoryProviderManager().active_provider is None


# -- prefetch fencing ---------------------------------------------------------


async def test_prefetch_all_fences_result() -> None:
    mgr = MemoryProviderManager()
    mgr.add_provider(_FakeProvider(prefetch_result="user prefers dark mode"))
    block = await mgr.prefetch_all("what theme?")
    assert block.startswith(OPEN)
    assert block.endswith(CLOSE)
    assert "user prefers dark mode" in block
    assert "[System note:" in block


async def test_prefetch_all_empty_result_returns_empty() -> None:
    mgr = MemoryProviderManager()
    mgr.add_provider(_FakeProvider(prefetch_result=""))
    assert await mgr.prefetch_all("q") == ""


async def test_prefetch_all_no_provider_returns_empty() -> None:
    assert await MemoryProviderManager().prefetch_all("q") == ""


async def test_prefetch_all_provider_error_returns_empty() -> None:
    class _Boom(_FakeProvider):
        async def prefetch(self, query: str, *, session_id: str = "") -> str:
            raise RuntimeError("recall failed")

    mgr = MemoryProviderManager()
    mgr.add_provider(_Boom())
    assert await mgr.prefetch_all("q") == ""


# -- background FIFO ordering -------------------------------------------------


async def test_sync_all_fifo_ordering() -> None:
    mgr = MemoryProviderManager()
    provider = _FakeProvider()
    mgr.add_provider(provider)

    mgr.sync_all("u1", "a1")
    mgr.sync_all("u2", "a2")
    drained = await mgr.flush_pending(timeout=5.0)

    assert drained is True
    assert provider.sync_calls == [("u1", "a1"), ("u2", "a2")]

    await mgr.shutdown()


async def test_sync_provider_exception_does_not_propagate_and_later_work_runs() -> None:
    mgr = MemoryProviderManager()
    provider = _FakeProvider(sync_raises=True)
    mgr.add_provider(provider)

    # A raising sync_turn must not raise to the caller and must not wedge the
    # consumer: later work still runs.
    mgr.sync_all("u1", "a1")

    ran = asyncio.Event()

    async def _sentinel() -> None:
        ran.set()

    mgr._submit_background(_sentinel)  # type: ignore[attr-defined]
    await mgr.flush_pending(timeout=5.0)

    assert ran.is_set()
    assert provider.sync_calls == []  # the raising sync recorded nothing

    await mgr.shutdown()


async def test_queue_prefetch_all_runs_in_background() -> None:
    mgr = MemoryProviderManager()
    provider = _FakeProvider()
    mgr.add_provider(provider)

    mgr.queue_prefetch_all("next-turn query")
    await mgr.flush_pending(timeout=5.0)

    assert provider.queue_prefetch_calls == ["next-turn query"]

    await mgr.shutdown()


# -- tool routing -------------------------------------------------------------


def test_tool_routing_and_schemas() -> None:
    mgr = MemoryProviderManager()
    mgr.add_provider(
        _FakeProvider(tool_schemas=[{"name": "recall", "description": "d", "parameters": {}}])
    )
    assert mgr.has_tool("recall") is True
    assert mgr.has_tool("nope") is False
    names = {s["name"] for s in mgr.get_tool_schemas()}
    assert names == {"recall"}


async def test_handle_tool_call_routes_to_provider() -> None:
    mgr = MemoryProviderManager()
    provider = _FakeProvider(
        tool_schemas=[{"name": "recall", "description": "d", "parameters": {}}]
    )
    mgr.add_provider(provider)
    result = await mgr.handle_tool_call("recall", {"q": "x"})
    assert "recall" in result
    assert provider.tool_calls == [("recall", {"q": "x"})]


async def test_handle_unknown_tool_returns_error_json() -> None:
    mgr = MemoryProviderManager()
    mgr.add_provider(_FakeProvider())
    result = await mgr.handle_tool_call("ghost", {})
    assert "ghost" in result
    assert "error" in result.lower()


def test_reserved_tool_name_collision_skipped() -> None:
    mgr = MemoryProviderManager(reserved_tool_names={"memory"})
    mgr.add_provider(
        _FakeProvider(
            tool_schemas=[
                {"name": "memory", "description": "collides", "parameters": {}},
                {"name": "recall", "description": "ok", "parameters": {}},
            ]
        )
    )
    # The colliding builtin name is skipped; the unique one is routed.
    assert mgr.has_tool("memory") is False
    assert mgr.has_tool("recall") is True
    names = {s["name"] for s in mgr.get_tool_schemas()}
    assert names == {"recall"}


# -- shutdown -----------------------------------------------------------------


async def test_shutdown_drains_pending_work_and_shuts_provider() -> None:
    mgr = MemoryProviderManager()
    provider = _FakeProvider()
    mgr.add_provider(provider)

    mgr.sync_all("u1", "a1")
    mgr.sync_all("u2", "a2")
    # No flush_pending here — shutdown itself must drain queued work first.
    await mgr.shutdown()

    assert provider.sync_calls == [("u1", "a1"), ("u2", "a2")]
    assert provider.shutdown_called is True


async def test_flush_pending_no_work_returns_true() -> None:
    mgr = MemoryProviderManager()
    assert await mgr.flush_pending(timeout=1.0) is True
