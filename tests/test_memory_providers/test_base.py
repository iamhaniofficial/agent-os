"""Contract tests for the MemoryProvider async ABC (Task B1)."""
from __future__ import annotations

from typing import Any

import pytest

from agentos.memory.providers import MemoryProvider


class _MinimalProvider(MemoryProvider):
    """Concrete subclass implementing only the abstract surface."""

    @property
    def name(self) -> str:
        return "minimal"

    def is_available(self) -> bool:
        return True

    async def initialize(self, session_id: str, **kwargs: Any) -> None:
        self.initialized_with = (session_id, kwargs)


def test_minimal_subclass_instantiates() -> None:
    provider = _MinimalProvider()
    assert provider.name == "minimal"
    assert provider.is_available() is True


async def test_initialize_receives_session_and_kwargs() -> None:
    provider = _MinimalProvider()
    result = await provider.initialize("sess-1", agent_state_dir="/tmp/state", platform="cli")
    assert result is None
    assert provider.initialized_with[0] == "sess-1"
    assert provider.initialized_with[1]["agent_state_dir"] == "/tmp/state"


def test_system_prompt_block_defaults_to_empty() -> None:
    assert _MinimalProvider().system_prompt_block() == ""


def test_get_tool_schemas_defaults_to_empty_list() -> None:
    assert _MinimalProvider().get_tool_schemas() == []


async def test_prefetch_defaults_to_empty_string() -> None:
    assert await _MinimalProvider().prefetch("query", session_id="s") == ""


async def test_queue_prefetch_defaults_to_noop() -> None:
    assert await _MinimalProvider().queue_prefetch("query", session_id="s") is None


async def test_sync_turn_defaults_to_noop() -> None:
    result = await _MinimalProvider().sync_turn(
        "hi", "hello", session_id="s", messages=[{"role": "user", "content": "hi"}]
    )
    assert result is None


async def test_shutdown_defaults_to_noop() -> None:
    assert await _MinimalProvider().shutdown() is None


async def test_on_session_end_defaults_to_noop() -> None:
    assert await _MinimalProvider().on_session_end([{"role": "user", "content": "hi"}]) is None


async def test_on_session_switch_defaults_to_noop() -> None:
    result = await _MinimalProvider().on_session_switch(
        "new-sess", parent_session_id="old", reset=True, rewound=False
    )
    assert result is None


async def test_on_pre_compress_defaults_to_empty_string() -> None:
    assert await _MinimalProvider().on_pre_compress([{"role": "user", "content": "hi"}]) == ""


async def test_on_memory_write_defaults_to_noop() -> None:
    result = await _MinimalProvider().on_memory_write(
        "add", "memory", "content", metadata={"origin": "test"}
    )
    assert result is None


async def test_handle_tool_call_raises_not_implemented_naming_provider() -> None:
    provider = _MinimalProvider()
    with pytest.raises(NotImplementedError) as exc_info:
        await provider.handle_tool_call("some_tool", {})
    assert "minimal" in str(exc_info.value)
    assert "some_tool" in str(exc_info.value)


def test_missing_abstract_member_raises_type_error() -> None:
    class _NoIsAvailable(MemoryProvider):
        @property
        def name(self) -> str:
            return "broken"

        async def initialize(self, session_id: str, **kwargs: Any) -> None:
            return None

    with pytest.raises(TypeError):
        _NoIsAvailable()  # type: ignore[abstract]


def test_missing_name_raises_type_error() -> None:
    class _NoName(MemoryProvider):
        def is_available(self) -> bool:
            return True

        async def initialize(self, session_id: str, **kwargs: Any) -> None:
            return None

    with pytest.raises(TypeError):
        _NoName()  # type: ignore[abstract]
