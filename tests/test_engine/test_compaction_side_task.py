"""The in-turn compaction summary is a bounded side task that never blocks
a turn for nothing (live QA: a 90 s summary that saved 0.3% of the request).

- the summary runs on the auxiliary client (task ``compaction``: the cheap
  side-task model, no thinking) under its own timeout;
- a summary that does not take >= 10% off the request is not applied: no
  durable rewrite, one log line, the request goes out as is, and the same
  attempt is not repeated on every request of the turn;
- an overflow caused only by the current turn's own tool result never calls
  the summarizer;
- a failed summarizer logs its error type and message (a bare TimeoutError
  used to log ``error=``).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
import structlog

from agentos.engine import Agent, AgentConfig, ToolResult
from agentos.engine.compaction_service import auxiliary_chunk_summarizer
from agentos.engine.types import CompactionEvent, DoneEvent, ThinkingEvent
from agentos.provider import ChatConfig, Message, auxiliary
from agentos.provider import DoneEvent as ProviderDone
from agentos.provider import ErrorEvent as ProviderError
from agentos.provider import TextDeltaEvent as ProviderText
from agentos.provider import ThinkingDeltaEvent as ProviderThinking
from agentos.provider import ToolUseEndEvent as ProviderToolUseEnd
from agentos.provider import ToolUseStartEvent as ProviderToolUseStart


class _HangingProvider:
    """A summarizer model that never answers (the live 90 s timeout)."""

    provider_name = "fake"

    def chat(self, messages=None, tools=None, config=None) -> AsyncIterator[Any]:
        return self._stream()

    async def _stream(self) -> AsyncIterator[Any]:
        await asyncio.sleep(30)
        yield ProviderText(text="never")


class _ToolLoopProvider:
    """Each turn: ``rounds`` tool calls, then a final answer."""

    provider_name = "fake"

    def __init__(self, rounds: int = 2) -> None:
        self.rounds = rounds
        self.calls: list[list[Message]] = []

    def chat(self, messages, tools=None, config=None) -> AsyncIterator[Any]:
        self.calls.append(list(messages))
        return self._stream(len(self.calls))

    async def _stream(self, call: int) -> AsyncIterator[Any]:
        if call <= self.rounds:
            yield ProviderToolUseStart(tool_use_id=f"t{call}", tool_name="read_file")
            yield ProviderToolUseEnd(
                tool_use_id=f"t{call}", tool_name="read_file", arguments={"path": f"f{call}"}
            )
            yield ProviderDone(stop_reason="tool_use", input_tokens=1, output_tokens=1)
            return
        yield ProviderText(text="done")
        yield ProviderDone(stop_reason="end_turn", input_tokens=1, output_tokens=1)


def _install_aux(monkeypatch: pytest.MonkeyPatch, provider: Any) -> list[dict[str, Any]]:
    built: list[dict[str, Any]] = []

    def _factory(**kwargs: Any) -> Any:
        built.append(kwargs)
        return provider

    monkeypatch.setattr(
        auxiliary,
        "_client",
        auxiliary.AuxiliaryClient(
            config={"provider": "opencap", "model": "cheap-model"},
            provider_factory=_factory,
        ),
    )
    return built


@pytest.mark.asyncio
async def test_summary_runs_on_the_auxiliary_cheap_model_without_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    class _Client:
        async def complete(self, **kwargs: Any) -> Any:
            seen.update(kwargs)
            return auxiliary.AuxResult(text="short summary", provider="opencap", model="c0")

    monkeypatch.setattr(auxiliary, "_client", _Client())
    summarize = auxiliary_chunk_summarizer(hint=("opencap", "c0-model"), session_key="s1")

    assert await summarize("conversation text", 12.5) == "short summary"
    assert seen["task"] == "compaction"
    assert (seen["preferred_provider"], seen["preferred_model"]) == ("opencap", "c0-model")
    assert seen["timeout"] == 12.5
    assert seen["session_key"] == "s1"
    config = seen["chat_config"]
    assert isinstance(config, ChatConfig)
    assert config.thinking is False
    assert "conversation compactor" in str(config.system)


@pytest.mark.asyncio
async def test_timed_out_summary_that_saves_nothing_is_not_applied_and_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = _install_aux(monkeypatch, _HangingProvider())
    provider = _ToolLoopProvider(rounds=2)

    async def _tool(call: Any) -> ToolResult:
        return ToolResult(tool_use_id=call.tool_use_id, tool_name=call.tool_name, content="ok")

    agent = Agent(
        provider=provider,
        config=AgentConfig(
            context_window_tokens=3000,
            context_overflow_threshold=0.5,
            max_provider_retries=0,
            max_iterations=10,
            compaction_summary_timeout_seconds=0.2,
        ),
        tool_handler=_tool,
        session_key="agent:main:main",
    )
    # Many short lines: the fallback digest is as long as what it replaces.
    history = [
        Message(
            role="user" if index % 2 == 0 else "assistant", content=f"note {index} " + "n" * 140
        )
        for index in range(60)
    ]
    agent.set_history(list(history))

    with structlog.testing.capture_logs() as logs:
        events = [event async for event in agent.run_turn("go on")]

    assert any(isinstance(event, DoneEvent) for event in events)
    # One summary attempt on the cheap auxiliary model, bounded, then given up.
    assert [kwargs["model"] for kwargs in built] == ["cheap-model"]
    failures = [log for log in logs if log["event"] == "compaction.summarizer_failed"]
    assert len(failures) == 1
    assert failures[0]["error_type"] == "AuxiliaryError"
    assert "timed out after" in failures[0]["error"]
    assert failures[0]["timeout_s"] == 0.2
    declined = [log for log in logs if log["event"] == "compaction.declined"]
    assert [log["reason"] for log in declined] == ["compaction_not_smaller"]
    # Nothing persisted or rewritten; every request carries the full history.
    assert not any(isinstance(event, CompactionEvent) for event in events)
    assert len(provider.calls) == 3
    assert all(call[: len(history)] == history for call in provider.calls)
    assert agent.transcript_record()["mode"] == "append"


@pytest.mark.asyncio
async def test_overflow_from_the_current_turns_own_tool_result_never_summarizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []

    async def _summarize(*args: Any, **kwargs: Any) -> tuple[str, str]:
        calls.append(args)
        return "s", "llm"

    monkeypatch.setattr("agentos.engine.compaction_service.summarize_messages", _summarize)
    provider = _ToolLoopProvider(rounds=1)

    async def _big(call: Any) -> ToolResult:
        return ToolResult(
            tool_use_id=call.tool_use_id, tool_name=call.tool_name, content="r" * 30_000
        )

    agent = Agent(
        provider=provider,
        config=AgentConfig(
            context_window_tokens=8000,
            context_overflow_threshold=0.5,
            max_provider_retries=0,
            tool_result_projection_max_inline_chars=200_000,
            # Keep the result whole when it is appended, so the request is
            # over the limit because of this turn's own tool result.
            tool_result_provider_request_max_chars=200_000,
        ),
        tool_handler=_big,
    )

    with structlog.testing.capture_logs() as logs:
        events = [event async for event in agent.run_turn("read the big file")]

    assert calls == []
    assert any(isinstance(event, DoneEvent) for event in events)
    assert not any(isinstance(event, CompactionEvent) for event in events)
    reasons = [log.get("reason") for log in logs if log["event"] == "compaction.declined"]
    assert reasons == ["current_turn_too_large"]
    # The big result reached the model in full.
    assert any("r" * 30_000 in str(message.content) for message in provider.calls[-1])


class _ThinkThenFailProvider:
    """Attempt 1 streams reasoning, then a transient error; attempt 2 succeeds."""

    provider_name = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, tools=None, config=None) -> AsyncIterator[Any]:
        self.calls += 1
        return self._stream(self.calls)

    async def _stream(self, call: int) -> AsyncIterator[Any]:
        yield ProviderThinking(text="plan: ")
        yield ProviderThinking(text="read the file")
        if call == 1:
            yield ProviderError(message="temporary overload", code="503")
            return
        yield ProviderText(text="answer")
        yield ProviderDone(stop_reason="end_turn", input_tokens=1, output_tokens=1)


@pytest.mark.asyncio
async def test_retry_after_a_thinking_only_attempt_shows_each_thinking_delta_once() -> None:
    provider = _ThinkThenFailProvider()
    agent = Agent(
        provider=provider,
        config=AgentConfig(max_provider_retries=2, retry_base_backoff_ms=1, retry_max_backoff_ms=1),
    )

    events = [event async for event in agent.run_turn("hi")]

    assert provider.calls == 2
    assert [event.text for event in events if isinstance(event, ThinkingEvent)] == [
        "plan: ",
        "read the file",
    ]
    assert any(isinstance(event, DoneEvent) and event.text == "answer" for event in events)


@pytest.mark.asyncio
async def test_fixed_context_over_the_limit_is_its_own_decline_reason() -> None:
    from agentos.engine.compaction_service import CompactionService

    calls: list[Any] = []

    async def _summarizer(messages: list[Message], previous: str | None) -> tuple[str, str]:
        calls.append(messages)
        return "s", "llm"

    messages = [
        Message(role="user", content="old " + "x" * 8000),
        Message(role="assistant", content="answer"),
        Message(role="user", content="now"),
    ]

    def _tokens(items: Any) -> int:
        return sum(len(str(m.content)) for m in items) // 4

    with structlog.testing.capture_logs() as logs:
        result = await CompactionService(provider=None, summarizer=_summarizer).compact(
            messages,
            turn_input_index=2,
            # 30k tokens of system prompt + tools against a 1,000 x 0.85 limit.
            estimated_tokens=_tokens(messages) + 30_000,
            window_tokens=1_000,
            message_tokens=_tokens,
        )

    assert result.status == "skipped"
    assert result.reason == "fixed_context_too_large"
    assert calls == []
    (declined,) = [log for log in logs if log["event"] == "compaction.declined"]
    assert declined["reason"] == "fixed_context_too_large"
    assert "preflight_compact_ratio" in declined["hint"]
    assert declined["fixed_tokens"] >= 30_000


@pytest.mark.asyncio
async def test_each_response_logs_estimate_against_provider_input_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agentos.engine.agent as agent_module

    debug_events: list[dict[str, Any]] = []
    real_logger = agent_module.logger

    class _Logger:
        def debug(self, event: str, **kwargs: Any) -> None:
            debug_events.append({"event": event, **kwargs})

        def __getattr__(self, name: str) -> Any:
            return getattr(real_logger, name)

    # Independent of the process-wide structlog level filter.
    monkeypatch.setattr(agent_module, "logger", _Logger())

    class _Provider:
        provider_name = "fake"

        def chat(self, messages, tools=None, config=None) -> AsyncIterator[Any]:
            async def _stream() -> AsyncIterator[Any]:
                yield ProviderText(text="ok")
                yield ProviderDone(
                    stop_reason="end_turn", input_tokens=1234, output_tokens=1, cached_tokens=7
                )

            return _stream()

    agent = Agent(provider=_Provider(), config=AgentConfig())

    _ = [event async for event in agent.run_turn("hello")]

    (drift,) = [log for log in debug_events if log["event"] == "agent.context_estimate_vs_provider"]
    assert drift["provider_input_tokens"] == 1234
    assert drift["provider_cached_tokens"] == 7
    assert drift["estimated_tokens"] > 0
