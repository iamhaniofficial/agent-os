"""A provider error that arrives *after* the 200 must surface as an ErrorEvent.

Anthropic emits ``event: error`` (``overloaded_error``, ``api_error``, ...),
OpenAI-compatible gateways emit a chunk carrying ``error``, and Ollama emits an
NDJSON line ``{"error": "..."}``; all three then close the stream without a
terminal marker. Before #2118 the Anthropic dispatch chain skipped the event
and the generator simply ended, while the OpenAI and Ollama loops fell through
to a DoneEvent, so the runtime recorded a *success* against the circuit
breaker for a turn that had actually failed.
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
    ErrorEvent,
    Message,
    TextDeltaEvent,
)
from agentos.provider.anthropic import AnthropicProvider
from agentos.provider.failures import ProviderFailureKind, classify_provider_error
from agentos.provider.ollama import OllamaProvider
from agentos.provider.openai import OpenAIProvider


def _patch_transport(monkeypatch: pytest.MonkeyPatch, module: str, body: bytes) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

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


def _sse(events: list[dict[str, Any]]) -> bytes:
    parts = []
    for ev in events:
        parts.append(f"event: {ev.get('type', 'message')}\n".encode())
        parts.append(f"data: {json.dumps(ev)}\n\n".encode())
    return b"".join(parts)


def _classify(provider_name: str, error: ErrorEvent) -> ProviderFailureKind:
    """Mirror ``engine.runtime._classify_provider_event``: a numeric code is the status."""
    return classify_provider_error(
        provider_name,
        int(error.code) if error.code.isdigit() else None,
        raw_code=error.code,
        message=error.message,
    )


def _assert_error_terminal(events: list[Any]) -> ErrorEvent:
    """The stream ends on exactly one ErrorEvent and never yields a DoneEvent."""
    assert not any(isinstance(ev, DoneEvent) for ev in events)
    errors = [ev for ev in events if isinstance(ev, ErrorEvent)]
    assert len(errors) == 1
    assert events[-1] is errors[0]
    return errors[0]


# --------------------------------------------------------------------------- Anthropic


def test_anthropic_mid_stream_error_event_yields_error_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = _sse(
        [
            {
                "type": "message_start",
                "message": {"id": "msg_1", "model": "claude-opus-4-7", "usage": {}},
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "partial"},
            },
            {
                "type": "error",
                "error": {"type": "overloaded_error", "message": "Overloaded"},
            },
            # Anthropic closes the connection here: no message_stop, no [DONE].
        ]
    )
    _patch_transport(monkeypatch, "anthropic", body)
    provider = AnthropicProvider(api_key="test", model="claude-opus-4-7")

    events = _collect(provider)

    assert [ev.text for ev in events if isinstance(ev, TextDeltaEvent)] == ["partial"]
    error = _assert_error_terminal(events)
    assert error.code == "overloaded_error"
    assert error.message == "overloaded_error: Overloaded"
    assert _classify("anthropic", error) is ProviderFailureKind.PROVIDER_OVERLOADED


def test_anthropic_error_event_without_detail_still_surfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_transport(monkeypatch, "anthropic", _sse([{"type": "error"}]))
    provider = AnthropicProvider(api_key="test", model="claude-opus-4-7")

    error = _assert_error_terminal(_collect(provider))

    assert error.code == "stream_error"
    assert error.message == "stream_error"


# --------------------------------------------------------------------- OpenAI-compat


def _openai_sse(chunks: list[dict[str, Any]], *, done: bool) -> bytes:
    parts = [f"data: {json.dumps(chunk)}\n\n".encode() for chunk in chunks]
    if done:
        parts.append(b"data: [DONE]\n\n")
    return b"".join(parts)


@pytest.mark.parametrize("done_marker", [False, True])
def test_openai_compat_mid_stream_error_chunk_yields_error_event(
    monkeypatch: pytest.MonkeyPatch,
    done_marker: bool,
) -> None:
    body = _openai_sse(
        [
            {
                "id": "c1",
                "model": "test-model",
                "choices": [{"index": 0, "delta": {"content": "partial"}}],
            },
            # OpenRouter's mid-stream shape: no choices, a bare error object.
            {
                "id": "c1",
                "error": {"code": 502, "message": "Provider returned error"},
            },
        ],
        done=done_marker,
    )
    _patch_transport(monkeypatch, "openai", body)
    provider = OpenAIProvider(
        api_key="test",
        model="test-model",
        base_url="https://openrouter.ai/api",
        provider_kind="openrouter",
    )

    events = _collect(provider)

    assert [ev.text for ev in events if isinstance(ev, TextDeltaEvent)] == ["partial"]
    error = _assert_error_terminal(events)
    assert error.code == "502"
    assert error.message == "502: Provider returned error"
    assert _classify("openrouter", error) is ProviderFailureKind.PROVIDER_OVERLOADED


def test_openai_compat_error_chunk_falls_back_to_type(monkeypatch: pytest.MonkeyPatch) -> None:
    body = _openai_sse(
        [{"error": {"type": "rate_limit_exceeded", "message": "Rate limit reached"}}],
        done=False,
    )
    _patch_transport(monkeypatch, "openai", body)
    provider = OpenAIProvider(api_key="test", model="test-model")

    error = _assert_error_terminal(_collect(provider))

    assert error.code == "rate_limit_exceeded"
    assert error.message == "rate_limit_exceeded: Rate limit reached"
    assert _classify("openai", error) is ProviderFailureKind.RATE_LIMITED


def test_openai_compat_error_chunk_without_code_or_message_keeps_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_transport(
        monkeypatch, "openai", _openai_sse([{"error": {"param": "messages"}}], done=False)
    )
    provider = OpenAIProvider(api_key="test", model="test-model")

    error = _assert_error_terminal(_collect(provider))

    assert error.code == "stream_error"
    assert error.message == 'stream_error: {"param": "messages"}'


def test_openai_compat_string_error_chunk_yields_error_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_transport(monkeypatch, "openai", _openai_sse([{"error": "upstream failed"}], done=False))
    provider = OpenAIProvider(api_key="test", model="test-model")

    error = _assert_error_terminal(_collect(provider))

    assert error.code == "stream_error"
    assert error.message == "stream_error: upstream failed"


def test_openai_compat_null_error_key_is_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gateways that send ``"error": null`` on ordinary chunks keep streaming."""
    body = _openai_sse(
        [
            {
                "error": None,
                "model": "test-model",
                "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}],
            }
        ],
        done=True,
    )
    _patch_transport(monkeypatch, "openai", body)
    provider = OpenAIProvider(api_key="test", model="test-model")

    events = _collect(provider)

    assert not any(isinstance(ev, ErrorEvent) for ev in events)
    assert isinstance(events[-1], DoneEvent)
    assert [ev.text for ev in events if isinstance(ev, TextDeltaEvent)] == ["ok"]


# --------------------------------------------------------------------------- Ollama


def test_ollama_mid_stream_error_line_yields_error_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = (
        json.dumps({"model": "llama3", "message": {"role": "assistant", "content": "partial"}})
        + "\n"
        + json.dumps({"error": "model runner has unexpectedly stopped"})
        + "\n"
    ).encode()
    _patch_transport(monkeypatch, "ollama", body)
    provider = OllamaProvider(model="llama3", base_url="http://localhost:11434")

    events = _collect(provider)

    assert [ev.text for ev in events if isinstance(ev, TextDeltaEvent)] == ["partial"]
    error = _assert_error_terminal(events)
    assert error.code == "stream_error"
    assert error.message == "model runner has unexpectedly stopped"


def test_ollama_mid_stream_missing_model_error_classifies_as_model_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = (json.dumps({"error": 'model "nope" not found, try pulling it first'}) + "\n").encode()
    _patch_transport(monkeypatch, "ollama", body)
    provider = OllamaProvider(model="nope", base_url="http://localhost:11434")

    error = _assert_error_terminal(_collect(provider))

    assert _classify("ollama", error) is ProviderFailureKind.MODEL_NOT_FOUND
