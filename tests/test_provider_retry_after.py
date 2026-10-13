"""Providers keep the ``Retry-After`` hint a non-200 response carried (#3646).

Before this the hint was thrown away with the rest of the response headers, so
a 429 that said "come back in two seconds" could only be answered with a blind
backoff or a failover to another model.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from typing import Any

import httpx
import pytest

from agentos.provider import ChatConfig, ErrorEvent, Message
from agentos.provider.anthropic import AnthropicProvider
from agentos.provider.ollama import OllamaProvider
from agentos.provider.openai import OpenAIProvider
from agentos.provider.openai_responses import OpenAIResponsesProvider
from agentos.provider.retry_after import (
    MAX_RETRY_AFTER_SECONDS,
    parse_duration,
    parse_retry_after,
    parse_retry_after_value,
    retry_after_from_response,
)

_NOW = datetime(2026, 10, 13, 12, 0, 0, tzinfo=UTC)

# ----------------------------------------------------------------- parsing


def test_delta_seconds() -> None:
    assert parse_retry_after({"Retry-After": "2"}) == 2.0
    assert parse_retry_after({"retry-after": " 1.5 "}) == 1.5


def test_http_date_is_seconds_from_now() -> None:
    when = format_datetime(_NOW + timedelta(seconds=7), usegmt=True)
    assert parse_retry_after({"Retry-After": when}, now=_NOW) == pytest.approx(7.0)


def test_http_date_in_the_past_clamps_to_zero() -> None:
    when = format_datetime(_NOW - timedelta(seconds=30), usegmt=True)
    assert parse_retry_after({"Retry-After": when}, now=_NOW) == 0.0


def test_values_clamp_to_the_ceiling() -> None:
    assert parse_retry_after({"Retry-After": "86400"}) == MAX_RETRY_AFTER_SECONDS
    assert parse_retry_after({"Retry-After": "-5"}) == 0.0


@pytest.mark.parametrize("value", ["", "soon", "nan", "Tue, 99 Foo 2026"])
def test_unreadable_values_are_no_hint(value: str) -> None:
    assert parse_retry_after_value(value) is None
    assert parse_retry_after({"Retry-After": value}) is None


def test_no_headers_is_no_hint() -> None:
    assert parse_retry_after(None) is None
    assert parse_retry_after({}) is None
    assert parse_retry_after({"content-type": "application/json"}) is None


def test_retry_after_ms_wins() -> None:
    assert parse_retry_after({"retry-after-ms": "250", "Retry-After": "9"}) == 0.25


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1s", 1.0),
        ("20ms", 0.02),
        ("2m0s", 120.0),
        ("1m2.5s", 62.5),
        # 6 minutes is past the ceiling and clamps to it.
        ("6m0s", MAX_RETRY_AFTER_SECONDS),
    ],
)
def test_openai_durations(value: str, expected: float) -> None:
    assert parse_duration(value) == pytest.approx(expected)


@pytest.mark.parametrize("value", ["", "abc", "5x", "1s junk"])
def test_bad_openai_durations(value: str) -> None:
    assert parse_duration(value) is None


def test_openai_reset_headers_use_the_spent_limit() -> None:
    headers = {
        "x-ratelimit-remaining-requests": "0",
        "x-ratelimit-reset-requests": "2s",
        "x-ratelimit-remaining-tokens": "5000",
        "x-ratelimit-reset-tokens": "45s",
    }
    assert parse_retry_after(headers) == 2.0


def test_retry_after_beats_vendor_reset_headers() -> None:
    headers = {"Retry-After": "3", "x-ratelimit-reset-requests": "40s"}
    assert parse_retry_after(headers) == 3.0


def test_anthropic_reset_timestamp() -> None:
    reset = (_NOW + timedelta(seconds=4)).isoformat().replace("+00:00", "Z")
    headers = {
        "anthropic-ratelimit-requests-remaining": "0",
        "anthropic-ratelimit-requests-reset": reset,
    }
    assert parse_retry_after(headers, now=_NOW) == pytest.approx(4.0)


def test_response_without_headers_is_no_hint() -> None:
    class _Bare:
        status_code = 429

    assert retry_after_from_response(_Bare()) is None


# ----------------------------------------------------------------- providers


def _patch_transport(
    monkeypatch: pytest.MonkeyPatch,
    module: str,
    *,
    status: int,
    headers: dict[str, str],
    body: bytes = b'{"error": {"type": "rate_limit_error", "message": "slow down"}}',
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            headers={"content-type": "application/json", **headers},
            content=body,
        )

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def patched_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(f"agentos.provider.{module}.httpx.AsyncClient", patched_async_client)


def _only_error(provider: Any) -> ErrorEvent:
    async def _run() -> list[Any]:
        return [
            event
            async for event in provider.chat(
                [Message(role="user", content="hi")],
                config=ChatConfig(),
            )
        ]

    events = asyncio.run(_run())
    errors = [event for event in events if isinstance(event, ErrorEvent)]
    assert len(errors) == 1, events
    return errors[0]


def _providers() -> list[tuple[str, Any]]:
    return [
        ("anthropic", lambda: AnthropicProvider(api_key="test", model="claude-opus-4-7")),
        ("openai", lambda: OpenAIProvider(api_key="test", model="test-model")),
        ("ollama", lambda: OllamaProvider(model="llama3", base_url="http://localhost:11434")),
        ("openai_responses", lambda: OpenAIResponsesProvider(api_key="test", model="gpt-5.5")),
    ]


@pytest.mark.parametrize(("module", "factory"), _providers())
def test_429_carries_delta_seconds_hint(
    monkeypatch: pytest.MonkeyPatch, module: str, factory: Any
) -> None:
    _patch_transport(monkeypatch, module, status=429, headers={"Retry-After": "2"})

    error = _only_error(factory())

    assert error.code == "429"
    assert error.retry_after_seconds == 2.0


@pytest.mark.parametrize(("module", "factory"), _providers())
def test_429_carries_http_date_hint(
    monkeypatch: pytest.MonkeyPatch, module: str, factory: Any
) -> None:
    when = format_datetime(datetime.now(UTC) + timedelta(seconds=20), usegmt=True)
    _patch_transport(monkeypatch, module, status=429, headers={"Retry-After": when})

    error = _only_error(factory())

    assert error.code == "429"
    assert error.retry_after_seconds is not None
    # HTTP-dates have one-second resolution and the clock moves while we run.
    assert 17.0 <= error.retry_after_seconds <= 20.0


@pytest.mark.parametrize(("module", "factory"), _providers())
def test_error_without_hint_has_none(
    monkeypatch: pytest.MonkeyPatch, module: str, factory: Any
) -> None:
    _patch_transport(monkeypatch, module, status=503, headers={})

    error = _only_error(factory())

    assert error.code == "503"
    assert error.retry_after_seconds is None
