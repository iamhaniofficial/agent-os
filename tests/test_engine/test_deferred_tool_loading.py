"""A tool loaded with tool_search mid-turn is on the very next provider request."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

import agentos.tools.builtin  # noqa: F401 - registers tool_search
from agentos.engine import Agent, AgentConfig
from agentos.provider import ChatConfig, Message
from agentos.provider import DoneEvent as ProviderDone
from agentos.provider import TextDeltaEvent as ProviderTextDelta
from agentos.provider import ToolUseEndEvent as ProviderToolUseEnd
from agentos.provider import ToolUseStartEvent as ProviderToolUseStart
from agentos.tools import deferred
from agentos.tools.deferred import TOOL_SEARCH_NAME, DeferredToolSurface, bind_agent_tools
from agentos.tools.dispatch import build_tool_handler
from agentos.tools.registry import ToolRegistry, get_default_registry
from agentos.tools.types import ToolContext, ToolSpec

SESSION = "agent:main:deferred-engine-test"


class _ScriptedProvider:
    """Call 1: tool_search(voice_maker). Call 2: voice_maker. Call 3: text."""

    provider_name = "fake"

    def __init__(self) -> None:
        self.tool_names_per_call: list[list[str]] = []

    def chat(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
        config: ChatConfig | None = None,
    ) -> AsyncIterator[Any]:
        self.tool_names_per_call.append([t.name for t in tools or []])
        return self._stream(len(self.tool_names_per_call))

    async def _stream(self, call_number: int) -> AsyncIterator[Any]:
        if call_number == 1:
            name, arguments = TOOL_SEARCH_NAME, {"names": ["voice_maker"]}
        elif call_number == 2:
            name, arguments = "voice_maker", {"text": "hello"}
        else:
            yield ProviderTextDelta(text="done")
            yield ProviderDone(stop_reason="stop", input_tokens=1, output_tokens=1)
            return
        tid = f"call-{call_number}"
        yield ProviderToolUseStart(tool_use_id=tid, tool_name=name)
        yield ProviderToolUseEnd(tool_use_id=tid, tool_name=name, arguments=arguments)
        yield ProviderDone(stop_reason="tool_use", input_tokens=1, output_tokens=1)

    async def list_models(self) -> list[Any]:
        return []


@pytest.fixture(autouse=True)
def _clean() -> Any:
    deferred.forget_session(SESSION)
    yield
    deferred.forget_session(SESSION)


async def test_tool_search_mid_turn_reaches_the_next_provider_call() -> None:
    ran: list[dict[str, Any]] = []

    async def voice_maker(text: str) -> str:
        ran.append({"text": text})
        return "audio.mp3"

    async def read_file() -> str:
        return ""

    registry = ToolRegistry()
    registry.register(ToolSpec(name="read_file", description="Read.", parameters={}), read_file)
    registry.register(
        ToolSpec(
            name="voice_maker",
            description="Make speech.",
            parameters={"text": {"type": "string"}},
            required=["text"],
            defer_loading=True,
        ),
        voice_maker,
    )
    search = get_default_registry().get(TOOL_SEARCH_NAME)
    assert search is not None
    registry.register(search.spec, search.handler)

    ctx = ToolContext(session_key=SESSION)
    surface = DeferredToolSurface.build(registry, ctx)
    assert surface is not None
    handler = surface.guard(build_tool_handler(registry, ctx))
    provider = _ScriptedProvider()
    agent = Agent(
        provider=provider,
        config=AgentConfig(max_iterations=5),
        tool_definitions=surface.tool_definitions(),
        tool_handler=handler,
    )
    bind_agent_tools(handler, agent)

    events = [event async for event in agent.run_turn("say hello out loud")]

    first, second, *_ = provider.tool_names_per_call
    assert "voice_maker" not in first and TOOL_SEARCH_NAME in first
    assert second[-1] == "voice_maker"
    assert ran == [{"text": "hello"}]
    tool_results = [
        e for e in events if type(e).__name__ == "ToolResultEvent" and e.tool_name == "voice_maker"
    ]
    assert tool_results and not getattr(tool_results[0], "is_error", False)
    assert json.dumps(surface.loaded) == '["voice_maker"]'


# ---------------------------------------------------------------------------
# Through the real turn runner: mid-turn load, then a restart
# ---------------------------------------------------------------------------


class _SequenceProvider:
    provider_name = "fake"

    def __init__(self, streams: list[list[Any]]) -> None:
        self.streams = streams
        self.tool_names_per_call: list[list[str]] = []

    def chat(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
        config: ChatConfig | None = None,
    ) -> AsyncIterator[Any]:
        index = len(self.tool_names_per_call)
        self.tool_names_per_call.append([t.name for t in tools or []])
        events = self.streams[index] if index < len(self.streams) else self.streams[-1]
        return self._stream(events)

    async def _stream(self, events: list[Any]) -> AsyncIterator[Any]:
        for event in events:
            yield event

    async def list_models(self) -> list[Any]:
        return []


class _SelectorClone:
    def __init__(self, provider: _SequenceProvider) -> None:
        self.provider = provider
        self.current_config = SimpleNamespace(model="fake-model")

    def resolve(self) -> _SequenceProvider:
        return self.provider

    def override_model(self, model: str) -> None:
        self.current_config.model = model

    def next_fallback_after_failure(self, primary_failure: Exception) -> _SequenceProvider:
        raise IndexError("No fallback configured")


class _ProviderSelector:
    def __init__(self, provider: _SequenceProvider) -> None:
        self.provider = provider

    def clone(self) -> _SelectorClone:
        return _SelectorClone(self.provider)


def _tool_call(tid: str, name: str, arguments: dict[str, Any]) -> list[Any]:
    return [
        ProviderToolUseStart(tool_use_id=tid, tool_name=name),
        ProviderToolUseEnd(tool_use_id=tid, tool_name=name, arguments=arguments),
        ProviderDone(stop_reason="tool_use", input_tokens=1, output_tokens=1),
    ]


_TEXT = [
    ProviderTextDelta(text="done"),
    ProviderDone(stop_reason="stop", input_tokens=1, output_tokens=1),
]


async def test_turn_runner_loads_mid_turn_and_a_restarted_runner_keeps_it() -> None:
    from agentos.engine.runtime import TurnRunner
    from agentos.session.manager import SessionManager
    from agentos.session.storage import SessionStorage
    from agentos.tools.types import CallerKind

    async def voice_maker(text: str) -> str:
        return f"spoke {text}"

    async def read_file() -> str:
        return ""

    registry = ToolRegistry()
    registry.register(ToolSpec(name="read_file", description="Read.", parameters={}), read_file)
    registry.register(
        ToolSpec(
            name="voice_maker",
            description="Make speech.",
            parameters={"text": {"type": "string"}},
            required=["text"],
            defer_loading=True,
        ),
        voice_maker,
    )
    search = get_default_registry().get(TOOL_SEARCH_NAME)
    assert search is not None
    registry.register(search.spec, search.handler)

    storage = SessionStorage(":memory:")
    await storage.connect()
    manager = SessionManager(storage)
    session_key = "agent:main:deferred-runner"
    await manager.create(session_key)
    deferred.forget_session(session_key)

    async def run(provider: _SequenceProvider) -> list[Any]:
        runner = TurnRunner(
            provider_selector=_ProviderSelector(provider),
            session_manager=manager,
            tool_registry=registry,
        )
        return [
            event
            async for event in runner.run(
                "say hello",
                session_key,
                ToolContext(caller_kind=CallerKind.CLI, session_key=session_key),
                history_has_persisted_user=False,
                no_memory_capture=True,
            )
        ]

    try:
        first = _SequenceProvider(
            [
                _tool_call("c1", TOOL_SEARCH_NAME, {"names": ["voice_maker"]}),
                _tool_call("c2", "voice_maker", {"text": "hello"}),
                _TEXT,
            ]
        )
        turn1_events = await run(first)

        # Gateway restart: in-memory state gone, only the session store remains.
        deferred.forget_session(session_key)
        second = _SequenceProvider([_TEXT])
        await run(second)
    finally:
        deferred.forget_session(session_key)
        await storage.close()

    turn1_call1, turn1_call2, *_ = first.tool_names_per_call
    assert "voice_maker" not in turn1_call1 and TOOL_SEARCH_NAME in turn1_call1
    assert turn1_call2[-1] == "voice_maker"
    voice_results = [
        e
        for e in turn1_events
        if type(e).__name__ == "ToolResultEvent" and e.tool_name == "voice_maker"
    ]
    assert voice_results and "spoke hello" in voice_results[0].result
    # Same order on the next turn, so the request prefix does not reshuffle.
    assert second.tool_names_per_call[0] == turn1_call2


async def test_a_deferred_tool_batched_after_tool_search_sees_the_load() -> None:
    """[tool_search, voice_maker] in one response: the search runs alone first."""

    ran: list[str] = []

    async def voice_maker(text: str) -> str:
        ran.append(text)
        return "audio.mp3"

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            name="voice_maker",
            description="Make speech.",
            parameters={"text": {"type": "string"}},
            required=["text"],
            defer_loading=True,
            concurrency="parallel",
        ),
        voice_maker,
    )
    search = get_default_registry().get(TOOL_SEARCH_NAME)
    assert search is not None
    real_search = search.handler

    async def slow_search(**kwargs: Any) -> str:
        # Yield first, as a real search can (hooks, budget, persistence): a
        # call running beside it would then reach the guard before the load.
        await asyncio.sleep(0.02)
        return await real_search(**kwargs)

    registry.register(search.spec, slow_search)
    ctx = ToolContext(session_key=SESSION)
    surface = DeferredToolSurface.build(registry, ctx)
    assert surface is not None
    handler = surface.guard(build_tool_handler(registry, ctx))
    batch = [
        ProviderToolUseStart(tool_use_id="a", tool_name=TOOL_SEARCH_NAME),
        ProviderToolUseEnd(
            tool_use_id="a", tool_name=TOOL_SEARCH_NAME, arguments={"names": ["voice_maker"]}
        ),
        ProviderToolUseStart(tool_use_id="b", tool_name="voice_maker"),
        ProviderToolUseEnd(tool_use_id="b", tool_name="voice_maker", arguments={"text": "hi"}),
        ProviderDone(stop_reason="tool_use", input_tokens=1, output_tokens=1),
    ]
    provider = _SequenceProvider([batch, _TEXT])
    agent = Agent(
        provider=provider,
        config=AgentConfig(max_iterations=3),
        tool_definitions=surface.tool_definitions(),
        tool_handler=handler,
        tool_registry=registry,
    )
    bind_agent_tools(handler, agent)

    events = [event async for event in agent.run_turn("say hi")]

    assert ran == ["hi"]
    voice = [
        e for e in events if type(e).__name__ == "ToolResultEvent" and e.tool_name == "voice_maker"
    ]
    assert voice and not voice[0].is_error
