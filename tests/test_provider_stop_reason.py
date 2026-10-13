"""Every provider maps its wire stop value onto the shared ``StopReason`` set.

Anthropic says ``max_tokens``, OpenAI-compatible gateways say ``length``,
Ollama says ``done_reason: "length"`` and the Responses API says
``status: "incomplete"``. Before #3641 those strings reached the engine raw and
it only recognized ``"length"``, so an Anthropic ``max_tokens`` never triggered
the continuation path and ``refusal`` / ``pause_turn`` looked like a normal end
of turn. The wire value is kept on ``raw_stop_reason``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from agentos.provider import (
    ChatConfig,
    DoneEvent,
    Message,
    StopReason,
    TextDeltaEvent,
    normalize_stop_reason,
)
from agentos.provider.anthropic import AnthropicProvider
from agentos.provider.ollama import OllamaProvider
from agentos.provider.openai import OpenAIProvider
from agentos.provider.openai_responses import OpenAIResponsesProvider


def _patch_transport(
    monkeypatch: pytest.MonkeyPatch,
    module: str,
    body: bytes,
    *,
    content_type: str = "text/event-stream",
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": content_type}, content=body)

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def patched_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(f"agentos.provider.{module}.httpx.AsyncClient", patched_async_client)


def _collect(provider: Any) -> list[Any]:
    async def _run() -> list[Any]:
        return [
            event
            async for event in provider.chat(
                [Message(role="user", content="hi")],
                config=ChatConfig(),
            )
        ]

    return asyncio.run(_run())


def _done(events: list[Any]) -> DoneEvent:
    done = [ev for ev in events if isinstance(ev, DoneEvent)]
    assert len(done) == 1
    return done[0]


# ------------------------------------------------------------------ vocabulary


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("end_turn", StopReason.END_TURN),
        ("max_tokens", StopReason.MAX_TOKENS),
        ("tool_use", StopReason.TOOL_USE),
        ("stop_sequence", StopReason.STOP_SEQUENCE),
        ("refusal", StopReason.REFUSAL),
        ("pause_turn", StopReason.PAUSE_TURN),
        ("stop", StopReason.END_TURN),
        ("length", StopReason.MAX_TOKENS),
        ("LENGTH", StopReason.MAX_TOKENS),
        ("tool_calls", StopReason.TOOL_USE),
        ("function_call", StopReason.TOOL_USE),
        ("content_filter", StopReason.REFUSAL),
        ("max_output_tokens", StopReason.MAX_TOKENS),
        ("something_new", StopReason.UNKNOWN),
        (None, StopReason.END_TURN),
        ("", StopReason.END_TURN),
    ],
)
def test_normalize_stop_reason(raw: str | None, expected: StopReason) -> None:
    assert normalize_stop_reason(raw) is expected


def test_normalize_stop_reason_upgrades_end_turn_with_tool_calls() -> None:
    assert normalize_stop_reason("stop", has_tool_calls=True) is StopReason.TOOL_USE
    # A truncation stays a truncation even if a tool call slipped out first.
    assert normalize_stop_reason("length", has_tool_calls=True) is StopReason.MAX_TOKENS


def test_done_event_normalizes_raw_values_and_keeps_them() -> None:
    done = DoneEvent(stop_reason="length")  # type: ignore[arg-type]
    assert done.stop_reason is StopReason.MAX_TOKENS
    assert done.raw_stop_reason == "length"

    canonical = DoneEvent(stop_reason=StopReason.REFUSAL, raw_stop_reason="refusal")
    assert canonical.stop_reason is StopReason.REFUSAL
    assert canonical.raw_stop_reason == "refusal"

    default = DoneEvent()
    assert default.stop_reason is StopReason.END_TURN
    assert default.raw_stop_reason is None
    # Backward compatible with callers comparing against plain strings.
    assert default.stop_reason == "end_turn"


# ------------------------------------------------------------------- Anthropic


def _anthropic_body(
    stop_reason: str | None,
    *,
    stop_details: dict[str, Any] | None = None,
    text: str = "partial",
) -> bytes:
    delta: dict[str, Any] = {}
    if stop_reason is not None:
        delta["stop_reason"] = stop_reason
    if stop_details is not None:
        delta["stop_details"] = stop_details
    events: list[dict[str, Any]] = [
        {
            "type": "message_start",
            "message": {"id": "msg_1", "model": "claude-opus-4-7", "usage": {}},
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": delta, "usage": {"output_tokens": 3}},
        {"type": "message_stop"},
    ]
    parts = []
    for ev in events:
        parts.append(f"event: {ev['type']}\n".encode())
        parts.append(f"data: {json.dumps(ev)}\n\n".encode())
    return b"".join(parts)


@pytest.mark.parametrize(
    ("wire", "expected"),
    [
        ("end_turn", StopReason.END_TURN),
        ("max_tokens", StopReason.MAX_TOKENS),
        ("tool_use", StopReason.TOOL_USE),
        ("stop_sequence", StopReason.STOP_SEQUENCE),
        ("pause_turn", StopReason.PAUSE_TURN),
        ("model_context_window_exceeded", StopReason.UNKNOWN),
    ],
)
def test_anthropic_stop_reason_mapping(
    monkeypatch: pytest.MonkeyPatch, wire: str, expected: StopReason
) -> None:
    _patch_transport(monkeypatch, "anthropic", _anthropic_body(wire))
    provider = AnthropicProvider(api_key="test", model="claude-opus-4-7")

    done = _done(_collect(provider))

    assert done.stop_reason is expected
    assert done.raw_stop_reason == wire
    assert done.stop_details is None


def test_anthropic_refusal_carries_stop_details(monkeypatch: pytest.MonkeyPatch) -> None:
    details = {
        "type": "refusal",
        "category": "cyber",
        "explanation": "The request was declined by a safety classifier.",
    }
    _patch_transport(
        monkeypatch,
        "anthropic",
        _anthropic_body("refusal", stop_details=details, text="I started to"),
    )
    provider = AnthropicProvider(api_key="test", model="claude-opus-4-7")

    events = _collect(provider)
    done = _done(events)

    assert [ev.text for ev in events if isinstance(ev, TextDeltaEvent)] == ["I started to"]
    assert done.stop_reason is StopReason.REFUSAL
    assert done.raw_stop_reason == "refusal"
    assert done.stop_details == details


def test_anthropic_refusal_without_stop_details(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_transport(monkeypatch, "anthropic", _anthropic_body("refusal"))
    provider = AnthropicProvider(api_key="test", model="claude-opus-4-7")

    done = _done(_collect(provider))

    assert done.stop_reason is StopReason.REFUSAL
    assert done.stop_details is None


def test_anthropic_missing_stop_reason_defaults_to_end_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_transport(monkeypatch, "anthropic", _anthropic_body(None))
    provider = AnthropicProvider(api_key="test", model="claude-opus-4-7")

    done = _done(_collect(provider))

    assert done.stop_reason is StopReason.END_TURN


# ------------------------------------------------------------- OpenAI-compatible


def _openai_body(finish_reason: str | None, *, tool_call: bool = False) -> bytes:
    delta: dict[str, Any] = {"content": "hello"}
    if tool_call:
        delta = {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_1",
                    "function": {"name": "read_file", "arguments": '{"path": "a"}'},
                }
            ]
        }
    chunks = [
        {"id": "c1", "model": "test-model", "choices": [{"index": 0, "delta": delta}]},
        {
            "id": "c1",
            "model": "test-model",
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        },
    ]
    parts = [f"data: {json.dumps(chunk)}\n\n".encode() for chunk in chunks]
    parts.append(b"data: [DONE]\n\n")
    return b"".join(parts)


@pytest.mark.parametrize(
    ("wire", "expected"),
    [
        ("stop", StopReason.END_TURN),
        ("length", StopReason.MAX_TOKENS),
        ("content_filter", StopReason.REFUSAL),
        ("error", StopReason.UNKNOWN),
    ],
)
def test_openai_finish_reason_mapping(
    monkeypatch: pytest.MonkeyPatch, wire: str, expected: StopReason
) -> None:
    _patch_transport(monkeypatch, "openai", _openai_body(wire))
    provider = OpenAIProvider(api_key="test", model="test-model")

    done = _done(_collect(provider))

    assert done.stop_reason is expected
    assert done.raw_stop_reason == wire


@pytest.mark.parametrize("wire", ["tool_calls", "function_call", "stop"])
def test_openai_tool_calls_map_to_tool_use(monkeypatch: pytest.MonkeyPatch, wire: str) -> None:
    _patch_transport(monkeypatch, "openai", _openai_body(wire, tool_call=True))
    provider = OpenAIProvider(api_key="test", model="test-model")

    done = _done(_collect(provider))

    assert done.stop_reason is StopReason.TOOL_USE
    assert done.raw_stop_reason == wire


def test_openai_missing_finish_reason_is_end_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_transport(monkeypatch, "openai", _openai_body(None))
    provider = OpenAIProvider(api_key="test", model="test-model")

    done = _done(_collect(provider))

    assert done.stop_reason is StopReason.END_TURN
    assert done.raw_stop_reason is None


# ----------------------------------------------------------------------- Ollama


def _ollama_body(done_reason: str | None, *, tool_call: bool = False) -> bytes:
    message: dict[str, Any] = {"role": "assistant", "content": "hello"}
    if tool_call:
        message["tool_calls"] = [{"function": {"name": "read_file", "arguments": {"path": "a"}}}]
    final: dict[str, Any] = {
        "model": "llama3",
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "prompt_eval_count": 4,
        "eval_count": 2,
    }
    if done_reason is not None:
        final["done_reason"] = done_reason
    lines = [
        json.dumps({"model": "llama3", "message": message, "done": False}),
        json.dumps(final),
    ]
    return ("\n".join(lines) + "\n").encode()


@pytest.mark.parametrize(
    ("wire", "expected"),
    [
        ("stop", StopReason.END_TURN),
        ("length", StopReason.MAX_TOKENS),
        ("unload", StopReason.UNKNOWN),
        (None, StopReason.END_TURN),
    ],
)
def test_ollama_done_reason_mapping(
    monkeypatch: pytest.MonkeyPatch, wire: str | None, expected: StopReason
) -> None:
    _patch_transport(monkeypatch, "ollama", _ollama_body(wire), content_type="application/x-ndjson")
    provider = OllamaProvider(model="llama3", base_url="http://localhost:11434")

    done = _done(_collect(provider))

    assert done.stop_reason is expected
    assert done.raw_stop_reason == wire


def test_ollama_tool_calls_map_to_tool_use(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_transport(
        monkeypatch,
        "ollama",
        _ollama_body("stop", tool_call=True),
        content_type="application/x-ndjson",
    )
    provider = OllamaProvider(model="llama3", base_url="http://localhost:11434")

    done = _done(_collect(provider))

    assert done.stop_reason is StopReason.TOOL_USE
    assert done.raw_stop_reason == "stop"


# ------------------------------------------------------------- Responses API


def _responses_body(
    status: str | None,
    *,
    incomplete_reason: str | None = None,
    function_call: bool = False,
    refusal: str | None = None,
) -> bytes:
    content: list[dict[str, Any]] = [{"type": "output_text", "text": "hello"}]
    if refusal is not None:
        content = [{"type": "refusal", "refusal": refusal}]
    output: list[dict[str, Any]] = [{"type": "message", "content": content}]
    if function_call:
        output.append(
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "read_file",
                "arguments": '{"path": "a"}',
            }
        )
    data: dict[str, Any] = {"model": "gpt-5.4", "output": output, "usage": {}}
    if status is not None:
        data["status"] = status
    if incomplete_reason is not None:
        data["incomplete_details"] = {"reason": incomplete_reason}
    return json.dumps(data).encode()


def _responses_done(monkeypatch: pytest.MonkeyPatch, body: bytes) -> DoneEvent:
    _patch_transport(monkeypatch, "openai_responses", body, content_type="application/json")
    provider = OpenAIResponsesProvider(api_key="test", model="gpt-5.4")
    return _done(_collect(provider))


def test_responses_completed_is_end_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    done = _responses_done(monkeypatch, _responses_body("completed"))
    assert done.stop_reason is StopReason.END_TURN
    assert done.raw_stop_reason == "completed"


def test_responses_completed_with_function_call_is_tool_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    done = _responses_done(monkeypatch, _responses_body("completed", function_call=True))
    assert done.stop_reason is StopReason.TOOL_USE


def test_responses_incomplete_max_output_tokens_is_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    done = _responses_done(
        monkeypatch, _responses_body("incomplete", incomplete_reason="max_output_tokens")
    )
    assert done.stop_reason is StopReason.MAX_TOKENS
    assert done.raw_stop_reason == "incomplete:max_output_tokens"


def test_responses_incomplete_content_filter_is_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    done = _responses_done(
        monkeypatch, _responses_body("incomplete", incomplete_reason="content_filter")
    )
    assert done.stop_reason is StopReason.REFUSAL
    assert done.stop_details == {"type": "refusal", "category": "content_filter"}


def test_responses_refusal_content_part_is_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    done = _responses_done(
        monkeypatch, _responses_body("completed", refusal="I can't help with that.")
    )
    assert done.stop_reason is StopReason.REFUSAL
    assert done.stop_details == {"type": "refusal", "explanation": "I can't help with that."}


def test_responses_unrecognized_status_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    done = _responses_done(monkeypatch, _responses_body("cancelled"))
    assert done.stop_reason is StopReason.UNKNOWN
    assert done.raw_stop_reason == "cancelled"
