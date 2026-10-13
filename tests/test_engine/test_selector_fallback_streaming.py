"""``_SelectorFallbackProvider`` forwards events as they arrive (issue #3643).

The wrapper used to hold back every non-text event until the first text delta
or ``done``, so tool-only and reasoning-heavy responses did not stream at all
and provider heartbeats never reached idle watchdogs. These tests drive a
scripted provider that logs each event as it produces it; interleaving that
log with what the consumer receives shows whether an event was forwarded
immediately or buffered.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from agentos.engine.runtime import _SelectorFallbackProvider
from agentos.provider import DoneEvent as ProviderDone
from agentos.provider import ErrorEvent as ProviderError
from agentos.provider import ProviderHeartbeatEvent
from agentos.provider import TextDeltaEvent as ProviderText
from agentos.provider import ThinkingDeltaEvent as ProviderThinking
from agentos.provider import ToolUseEndEvent as ProviderToolUseEnd
from agentos.provider import ToolUseStartEvent as ProviderToolUseStart
from agentos.provider.circuit_breaker import (
    BreakerSettings,
    BreakerState,
    ProviderCircuitBreaker,
)

pytestmark = pytest.mark.anyio


def _label(event: Any) -> str:
    return str(getattr(event, "kind", type(event).__name__))


class _ScriptedProvider:
    """Yields a fixed script, logging ``produced:<kind>`` before each event."""

    def __init__(self, name: str, script: list[Any], log: list[str]) -> None:
        self.provider_name = name
        self._script = script
        self._log = log
        self.calls = 0

    def chat(self, messages: list[Any], tools: Any = None, config: Any = None) -> AsyncIterator:
        self.calls += 1
        return self._stream()

    async def _stream(self) -> AsyncIterator[Any]:
        for event in self._script:
            self._log.append(f"{self.provider_name}:produced:{_label(event)}")
            yield event

    async def list_models(self) -> list[Any]:
        return []


class _Selector:
    def __init__(self, breaker: ProviderCircuitBreaker, chain: list[str]) -> None:
        self._breaker = breaker
        self._chain = chain
        self._index = 0
        self.fallbacks: list[str] = []
        self.providers: dict[str, _ScriptedProvider] = {}

    @property
    def active_provider_id(self) -> str:
        return self._chain[self._index]

    def has_fallback(self) -> bool:
        return self._index < len(self._chain) - 1

    def next_fallback_after_failure(self, failure: Exception) -> _ScriptedProvider:
        if not self.has_fallback():
            raise IndexError("No fallback chain available")
        self._index += 1
        self.fallbacks.append(str(failure))
        return self.providers[self.active_provider_id]

    def record_provider_failure(self, kind: Any = None, reason: str = "") -> BreakerState:
        return self._breaker.record_failure(self.active_provider_id, kind=kind, reason=reason)

    def record_provider_success(self) -> None:
        self._breaker.record_success(self.active_provider_id)


def _breaker() -> ProviderCircuitBreaker:
    return ProviderCircuitBreaker(BreakerSettings(failure_threshold=1, cooldown_seconds=60.0))


def _setup(
    primary_script: list[Any],
    fallback_script: list[Any] | None = None,
) -> tuple[_SelectorFallbackProvider, _Selector, _ScriptedProvider, list[str]]:
    log: list[str] = []
    breaker = _breaker()
    selector = _Selector(breaker, ["primary", "backup"])
    selector.providers["backup"] = _ScriptedProvider(
        "backup",
        fallback_script or [ProviderText(text="from backup"), ProviderDone(stop_reason="stop")],
        log,
    )
    primary = _ScriptedProvider("primary", primary_script, log)
    return _SelectorFallbackProvider(primary, selector), selector, primary, log


async def _consume(wrapper: _SelectorFallbackProvider, log: list[str]) -> list[Any]:
    received: list[Any] = []
    async for event in wrapper.chat([], tools=None, config=None):
        log.append(f"received:{_label(event)}")
        received.append(event)
    return received


def _overload() -> ProviderError:
    return ProviderError(code="503", message="upstream 503")


async def test_tool_only_response_streams_before_done_and_never_fails_over() -> None:
    wrapper, selector, _, log = _setup(
        [
            ProviderToolUseStart(tool_use_id="t1", tool_name="read_file"),
            ProviderToolUseEnd(tool_use_id="t1", tool_name="read_file", arguments={"p": "x"}),
            _overload(),
            ProviderDone(stop_reason="tool_use"),
        ]
    )

    received = await _consume(wrapper, log)

    # tool_use_start reaches the consumer before the provider produces anything else.
    assert log[:2] == ["primary:produced:tool_use_start", "received:tool_use_start"]
    assert log.index("received:tool_use_start") < log.index("primary:produced:done")
    # The error after tool use passes through instead of triggering a failover.
    assert selector.fallbacks == []
    assert selector.providers["backup"].calls == 0
    assert [_label(e) for e in received] == ["tool_use_start", "tool_use_end", "error", "done"]
    # Forwarded tool use counted as success before the error arrived.
    assert selector._breaker.state("primary") is BreakerState.CLOSED


async def test_thinking_deltas_stream_immediately() -> None:
    wrapper, selector, _, log = _setup(
        [
            ProviderThinking(text="step one"),
            ProviderThinking(text="step two"),
            ProviderText(text="answer"),
            ProviderDone(stop_reason="stop"),
        ]
    )

    await _consume(wrapper, log)

    assert log == [
        "primary:produced:thinking_delta",
        "received:thinking_delta",
        "primary:produced:thinking_delta",
        "received:thinking_delta",
        "primary:produced:text_delta",
        "received:text_delta",
        "primary:produced:done",
        "received:done",
    ]
    assert selector.fallbacks == []


async def test_error_after_thinking_does_not_fail_over() -> None:
    wrapper, selector, _, log = _setup([ProviderThinking(text="hmm"), _overload()])

    received = await _consume(wrapper, log)

    assert selector.fallbacks == []
    assert [_label(e) for e in received] == ["thinking_delta", "error"]


async def test_heartbeats_stream_immediately() -> None:
    wrapper, _, _, log = _setup(
        [
            ProviderHeartbeatEvent(phase="provider", message="waiting"),
            ProviderHeartbeatEvent(phase="provider", message="still waiting"),
            ProviderText(text="hi"),
            ProviderDone(stop_reason="stop"),
        ]
    )

    await _consume(wrapper, log)

    assert log[:4] == [
        "primary:produced:provider_heartbeat",
        "received:provider_heartbeat",
        "primary:produced:provider_heartbeat",
        "received:provider_heartbeat",
    ]


async def test_error_after_heartbeats_only_still_fails_over() -> None:
    wrapper, selector, primary, log = _setup(
        [
            ProviderHeartbeatEvent(phase="provider", message="waiting"),
            _overload(),
        ]
    )

    received = await _consume(wrapper, log)

    assert selector.fallbacks == ["upstream 503"]
    assert selector.providers["backup"].calls == 1
    assert log[:2] == ["primary:produced:provider_heartbeat", "received:provider_heartbeat"]
    texts = [e.text for e in received if _label(e) == "text_delta"]
    assert texts == ["from backup"]
    assert not any(_label(e) == "error" for e in received)
    # A heartbeat is not success: the primary's failure opened its breaker.
    assert selector._breaker.state("primary") is BreakerState.OPEN
    assert selector._breaker.state("backup") is BreakerState.CLOSED


async def test_error_before_any_event_fails_over() -> None:
    wrapper, selector, _, log = _setup([_overload()])

    received = await _consume(wrapper, log)

    assert selector.fallbacks == ["upstream 503"]
    assert [_label(e) for e in received] == ["provider_heartbeat", "text_delta", "done"]
    assert received[0].phase == "llm_fallback"


async def test_error_after_text_does_not_fail_over() -> None:
    wrapper, selector, _, log = _setup([ProviderText(text="partial"), _overload()])

    received = await _consume(wrapper, log)

    assert selector.fallbacks == []
    assert selector.providers["backup"].calls == 0
    assert [_label(e) for e in received] == ["text_delta", "error"]


async def test_fallback_stream_does_not_fail_over_again_after_content() -> None:
    """Content forwarded from a fallback provider pins the stream to it too."""
    log: list[str] = []
    selector = _Selector(_breaker(), ["primary", "backup", "third"])
    selector.providers["backup"] = _ScriptedProvider(
        "backup", [ProviderToolUseStart(tool_use_id="t1", tool_name="x"), _overload()], log
    )
    selector.providers["third"] = _ScriptedProvider(
        "third", [ProviderText(text="third"), ProviderDone(stop_reason="stop")], log
    )
    wrapper = _SelectorFallbackProvider(_ScriptedProvider("primary", [_overload()], log), selector)

    received = await _consume(wrapper, log)

    assert selector.fallbacks == ["upstream 503"]
    assert selector.providers["third"].calls == 0
    assert [_label(e) for e in received] == ["provider_heartbeat", "tool_use_start", "error"]


async def test_fallback_classifies_each_error_independently() -> None:
    """A later error is judged by its own kind, not the first error's.

    The backup's overload asks for another failover, which fails (say the
    third provider could not be built). The auth error that follows must not
    inherit the overload's "fall back" verdict and retry the failover.
    """

    class _FlakySelector(_Selector):
        def __init__(self, *args: Any) -> None:
            super().__init__(*args)
            self.attempts = 0

        def next_fallback_after_failure(self, failure: Exception) -> _ScriptedProvider:
            self.attempts += 1
            if self.attempts == 2:
                raise RuntimeError("could not build the next provider")
            return super().next_fallback_after_failure(failure)

    log: list[str] = []
    selector = _FlakySelector(_breaker(), ["primary", "backup", "third"])
    selector.providers["backup"] = _ScriptedProvider(
        "backup",
        [_overload(), ProviderError(code="401", message="invalid api key")],
        log,
    )
    selector.providers["third"] = _ScriptedProvider(
        "third", [ProviderText(text="third"), ProviderDone(stop_reason="stop")], log
    )
    wrapper = _SelectorFallbackProvider(_ScriptedProvider("primary", [_overload()], log), selector)

    received = await _consume(wrapper, log)

    assert selector.attempts == 2
    assert selector.providers["third"].calls == 0
    assert [e.code for e in received if _label(e) == "error"] == ["503", "401"]
