"""The agent loop acts on normalized stop reasons (#3641).

Providers report why they stopped in their own words; the engine used to
recognize only OpenAI's ``"length"``. These tests drive ``Agent.run_turn`` with
scripted provider streams that stop the way Anthropic does: ``max_tokens`` must
continue like ``length``, ``refusal`` must end the turn without a retry or a
failover, and ``pause_turn`` must be resumed (a bounded number of times).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

from agentos.engine import Agent, AgentConfig, ToolResult
from agentos.engine.runtime import TurnRunner
from agentos.engine.types import DoneEvent as EngineDone
from agentos.provider import (
    ChatConfig,
    Message,
    StopReason,
    ToolDefinition,
    ToolInputSchema,
)
from agentos.provider import DoneEvent as ProviderDone
from agentos.provider import TextDeltaEvent as ProviderText
from agentos.provider import ToolUseEndEvent as ProviderToolUseEnd
from agentos.provider import ToolUseStartEvent as ProviderToolUseStart


class _SequenceProvider:
    provider_name = "fake"

    def __init__(self, streams: list[list[Any]]) -> None:
        self.streams = streams
        self.calls: list[dict[str, Any]] = []
        self.fallback_reasons: list[str] = []

    def chat(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
        config: ChatConfig | None = None,
    ) -> AsyncIterator[Any]:
        index = len(self.calls)
        self.calls.append({"messages": list(messages), "tools": tools, "config": config})
        events = self.streams[index] if index < len(self.streams) else self.streams[-1]
        return self._stream(events)

    async def _stream(self, events: list[Any]) -> AsyncIterator[Any]:
        for event in events:
            yield event

    def fallback_after_invalid_response(self, reason: str) -> bool:
        # Offered on purpose: a refusal must never take it.
        self.fallback_reasons.append(reason)
        return True

    async def list_models(self) -> list[Any]:
        return []


class _CallLog:
    def __init__(self) -> None:
        self.records: list[tuple[str, dict[str, Any]]] = []

    def write(self, kind: str, payload: dict[str, Any]) -> None:
        self.records.append((kind, payload))

    def decisions(self) -> list[dict[str, Any]]:
        return [payload for kind, payload in self.records if kind == "turn_policy_decision"]


def _agent(provider: _SequenceProvider, **config: Any) -> Agent:
    return Agent(
        provider=provider,
        config=AgentConfig(retry_base_backoff_ms=0, retry_max_backoff_ms=0, **config),
    )


def _anthropic_done(stop: str, **kwargs: Any) -> ProviderDone:
    """A DoneEvent the way the Anthropic provider builds it."""

    return ProviderDone(stop_reason=StopReason(stop), raw_stop_reason=stop, **kwargs)


def _texts_in_messages(messages: list[Message]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for message in messages:
        if isinstance(message.content, str):
            out.append((message.role, message.content))
            continue
        for block in message.content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                out.append((message.role, text))
    return out


# -------------------------------------------------------------------- max_tokens


@pytest.mark.asyncio
async def test_anthropic_max_tokens_triggers_length_capped_continuation() -> None:
    provider = _SequenceProvider(
        [
            [
                ProviderText(text="partial answer"),
                _anthropic_done("max_tokens", input_tokens=7, output_tokens=9),
            ],
            [
                ProviderText(text=" finished"),
                _anthropic_done("end_turn", input_tokens=8, output_tokens=1),
            ],
        ]
    )
    agent = _agent(provider, max_provider_retries=1)

    events = [event async for event in agent.run_turn("hello")]

    assert len(provider.calls) == 2
    assert any(
        event.kind == "warning" and event.code == "provider_output_continue" for event in events
    )
    assert not any(event.kind == "error" for event in events)
    done = next(event for event in events if event.kind == "done")
    assert done.text == "partial answer finished"
    assert done.stop_reason == "end_turn"


# ----------------------------------------------------------------------- refusal


@pytest.mark.asyncio
async def test_refusal_ends_turn_without_retry_or_failover_and_keeps_text() -> None:
    details = {"type": "refusal", "category": "cyber", "explanation": "Declined."}
    provider = _SequenceProvider(
        [
            [
                ProviderText(text="Here is the start"),
                _anthropic_done("refusal", stop_details=details, input_tokens=5, output_tokens=4),
            ],
            [ProviderText(text="should never be requested"), _anthropic_done("end_turn")],
        ]
    )
    agent = _agent(provider, max_provider_retries=2)
    call_log = _CallLog()
    agent._turn_call_logger = call_log  # type: ignore[assignment]

    events = [event async for event in agent.run_turn("hello")]

    assert len(provider.calls) == 1
    assert provider.fallback_reasons == []
    assert not any(event.kind == "error" for event in events)
    warnings = [event for event in events if event.kind == "warning"]
    assert [w.code for w in warnings] == ["provider_refusal"]
    assert "cyber" in warnings[0].message
    done = next(event for event in events if event.kind == "done")
    assert done.text == "Here is the start"
    assert done.stop_reason == "refusal"
    assert done.stop_details == details
    decisions = call_log.decisions()
    assert any(d["action"] == "finish_refused_turn" and d["category"] == "cyber" for d in decisions)
    # The text the model produced before declining stays in the transcript.
    assert any(
        role == "assistant" and text == "Here is the start"
        for role, text in _texts_in_messages(agent._history)
    )


@pytest.mark.asyncio
async def test_refusal_without_text_finishes_with_notice_instead_of_empty_retry() -> None:
    provider = _SequenceProvider(
        [
            [_anthropic_done("refusal", input_tokens=5, output_tokens=0)],
            [ProviderText(text="should never be requested"), _anthropic_done("end_turn")],
        ]
    )
    agent = _agent(provider, max_provider_retries=2)

    events = [event async for event in agent.run_turn("hello")]

    # Without the refusal path an empty response is retried / failed over.
    assert len(provider.calls) == 1
    assert provider.fallback_reasons == []
    assert not any(event.kind == "error" for event in events)
    warning = next(event for event in events if event.kind == "warning")
    assert warning.code == "provider_refusal"
    assert "safety category" not in warning.message
    done = next(event for event in events if event.kind == "done")
    assert done.text == "The model declined to respond to this request."


@pytest.mark.asyncio
async def test_openai_content_filter_is_treated_as_refusal() -> None:
    provider = _SequenceProvider(
        [
            [ProviderText(text="Partial"), ProviderDone(stop_reason="content_filter")],  # type: ignore[arg-type]
            [ProviderText(text="should never be requested"), ProviderDone()],
        ]
    )
    agent = _agent(provider, max_provider_retries=2)

    events = [event async for event in agent.run_turn("hello")]

    assert len(provider.calls) == 1
    assert any(event.kind == "warning" and event.code == "provider_refusal" for event in events)
    done = next(event for event in events if event.kind == "done")
    assert done.text == "Partial"


@pytest.mark.asyncio
async def test_refusal_tool_call_is_not_executed() -> None:
    provider = _SequenceProvider(
        [
            [
                ProviderToolUseStart(tool_use_id="tool-1", tool_name="echo"),
                ProviderToolUseEnd(tool_use_id="tool-1", tool_name="echo", arguments={}),
                _anthropic_done("refusal"),
            ]
        ]
    )
    called = False

    async def tool_handler(call: Any) -> ToolResult:
        nonlocal called
        called = True
        return ToolResult(tool_use_id=call.tool_use_id, tool_name=call.tool_name, content="ok")

    agent = Agent(
        provider=provider,
        config=AgentConfig(retry_base_backoff_ms=0, retry_max_backoff_ms=0),
        tool_definitions=[
            ToolDefinition(name="echo", description="Echo.", input_schema=ToolInputSchema())
        ],
        tool_handler=tool_handler,
    )

    events = [event async for event in agent.run_turn("hello")]

    assert called is False
    assert len(provider.calls) == 1
    assert not any(event.kind == "tool_result" for event in events)
    assert any(event.kind == "warning" and event.code == "provider_refusal" for event in events)


def test_decision_log_records_refusal_category(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "agentos.engine.runtime.write_decision_entry",
        lambda entry: captured.setdefault("entry", entry),
    )
    runner = TurnRunner(provider_selector=None)

    runner._emit_decision_entry(
        turn_id="turn-1",
        session_key="agent:main:session",
        session_id="session-id",
        message="hello",
        final_prompt="system",
        tool_defs=[],
        turn_obj=SimpleNamespace(metadata={}),
        provider=SimpleNamespace(),
        resolved_model="claude-opus-4-7",
        turn_started_at=0.0,
        done_event=EngineDone(
            stop_reason="refusal",
            stop_details={"type": "refusal", "category": "bio", "explanation": "x"},
        ),
    )

    entry = captured["entry"]
    assert entry.stop_reason == "refusal"
    assert entry.refusal_category == "bio"


# -------------------------------------------------------------------- pause_turn


@pytest.mark.asyncio
async def test_pause_turn_resumes_once_and_keeps_paused_text() -> None:
    provider = _SequenceProvider(
        [
            [ProviderText(text="part one"), _anthropic_done("pause_turn", input_tokens=3)],
            [ProviderText(text=" part two"), _anthropic_done("end_turn", input_tokens=4)],
        ]
    )
    # Resuming a pause does not spend the provider retry budget.
    agent = _agent(provider, max_provider_retries=0)
    call_log = _CallLog()
    agent._turn_call_logger = call_log  # type: ignore[assignment]

    events = [event async for event in agent.run_turn("hello")]

    assert len(provider.calls) == 2
    assert not any(event.kind == "error" for event in events)
    assert not any(event.kind == "warning" for event in events)
    done = next(event for event in events if event.kind == "done")
    assert done.text == "part one part two"
    assert done.input_tokens == 7
    assert done.stop_reason == "end_turn"
    resumed = _texts_in_messages(provider.calls[1]["messages"])
    assert ("assistant", "part one") in resumed
    assert resumed[-1][0] == "user"
    assert "paused" in resumed[-1][1]
    assert [d["action"] for d in call_log.decisions()] == ["resume_paused_turn"]


@pytest.mark.asyncio
async def test_pause_turn_resumes_are_bounded_then_end_the_turn() -> None:
    provider = _SequenceProvider(
        [[ProviderText(text="chunk "), _anthropic_done("pause_turn", output_tokens=1)]]
    )
    agent = _agent(provider, max_provider_retries=1)
    call_log = _CallLog()
    agent._turn_call_logger = call_log  # type: ignore[assignment]

    events = [event async for event in agent.run_turn("hello")]

    # One original call plus three resumes, then the pause counts as end_turn.
    assert len(provider.calls) == 4
    assert not any(event.kind == "error" for event in events)
    done = next(event for event in events if event.kind == "done")
    assert done.text == "chunk chunk chunk chunk "
    assert done.stop_reason == "end_turn"
    actions = [d["action"] for d in call_log.decisions()]
    assert actions == ["resume_paused_turn"] * 3 + ["end_paused_turn"]


@pytest.mark.asyncio
async def test_unknown_stop_reason_is_a_normal_end_of_turn() -> None:
    provider = _SequenceProvider(
        [[ProviderText(text="ok"), ProviderDone(stop_reason="brand_new_reason")]]  # type: ignore[arg-type]
    )
    agent = _agent(provider, max_provider_retries=1)

    events = [event async for event in agent.run_turn("hello")]

    assert len(provider.calls) == 1
    assert not any(event.kind in {"error", "warning"} for event in events)
    done = next(event for event in events if event.kind == "done")
    assert done.text == "ok"
    assert done.stop_reason == "unknown"
