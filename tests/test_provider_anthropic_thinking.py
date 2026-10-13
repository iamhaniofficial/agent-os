"""The Anthropic adapter must send the thinking payload each Claude family accepts.

Claude Opus 4.6 / Sonnet 4.6 and later take ``thinking: {"type": "adaptive"}``
with depth in ``output_config.effort``; ``budget_tokens`` is a 400 on 4.7+ and
on every 5.x model. Haiku 4.5 and older still need ``{"type": "enabled",
"budget_tokens": N}``. Before #3640 only ids containing ``claude-opus-4-6`` /
``claude-sonnet-4-6`` were treated as adaptive, so Opus 5.5 got a budget and
was rejected.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from agentos.engine.types import ThinkingLevel
from agentos.provider import ChatConfig, DoneEvent, Message
from agentos.provider.anthropic import AnthropicProvider, claude_capabilities

_SSE_BODY = b"".join(
    f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n".encode()
    for ev in (
        {"type": "message_start", "message": {"usage": {"input_tokens": 3}}},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 1},
        },
        {"type": "message_stop"},
    )
)


def _sent_payload(monkeypatch: pytest.MonkeyPatch, model: str, cfg: ChatConfig) -> dict[str, Any]:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_SSE_BODY)

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def patched_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr("agentos.provider.anthropic.httpx.AsyncClient", patched_async_client)
    provider = AnthropicProvider(api_key="test-key", model=model)

    async def _run() -> list[Any]:
        return [ev async for ev in provider.chat([Message(role="user", content="hi")], config=cfg)]

    events = asyncio.run(_run())
    assert any(isinstance(ev, DoneEvent) for ev in events)
    assert len(captured) == 1
    return captured[0]


def test_opus_5_5_thinking_high_sends_adaptive_and_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = ChatConfig(
        max_tokens=16000,
        thinking=True,
        thinking_budget_tokens=20000,
        thinking_level=ThinkingLevel.HIGH,
        temperature=0.3,
    )
    payload = _sent_payload(monkeypatch, "claude-opus-5-5", cfg)

    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert payload["output_config"] == {"effort": "high"}
    assert payload["max_tokens"] == 16000
    assert "budget_tokens" not in json.dumps(payload)
    assert "temperature" not in payload


def test_dotted_opus_5_5_id_is_adaptive(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = ChatConfig(thinking=True, thinking_level=ThinkingLevel.XHIGH)
    payload = _sent_payload(monkeypatch, "claude-opus-5.5", cfg)

    assert payload["model"] == "claude-opus-5.5"
    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert payload["output_config"] == {"effort": "xhigh"}


def test_haiku_4_5_thinking_keeps_budget_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = ChatConfig(
        max_tokens=8000,
        thinking=True,
        thinking_budget_tokens=4096,
        thinking_level=ThinkingLevel.LOW,
    )
    payload = _sent_payload(monkeypatch, "claude-haiku-4-5-20251001", cfg)

    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    assert payload["max_tokens"] == 8000
    assert "output_config" not in payload


def test_haiku_4_5_budget_is_raised_to_the_api_minimum(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = ChatConfig(max_tokens=8000, thinking=True, thinking_budget_tokens=200)
    payload = _sent_payload(monkeypatch, "claude-haiku-4-5", cfg)

    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 1024}


def test_sonnet_4_6_thinking_is_adaptive_without_xhigh(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = ChatConfig(thinking=True, thinking_level=ThinkingLevel.XHIGH)
    payload = _sent_payload(monkeypatch, "claude-sonnet-4-6", cfg)

    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    # xhigh arrived with Opus 4.7; 4.6 tops out at high below max.
    assert payload["output_config"] == {"effort": "high"}
    assert "budget_tokens" not in json.dumps(payload)


@pytest.mark.parametrize("level", [None, ThinkingLevel.ADAPTIVE])
def test_adaptive_level_leaves_effort_to_the_model(
    monkeypatch: pytest.MonkeyPatch, level: ThinkingLevel | None
) -> None:
    cfg = ChatConfig(thinking=True, thinking_level=level)
    payload = _sent_payload(monkeypatch, "claude-sonnet-5-5", cfg)

    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert "output_config" not in payload


def test_minimal_level_maps_to_low_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = ChatConfig(thinking=True, thinking_level=ThinkingLevel.MINIMAL)
    payload = _sent_payload(monkeypatch, "claude-fable-5-1", cfg)

    assert payload["output_config"] == {"effort": "low"}


def test_opus_5_5_thinking_off_drops_thinking_and_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = ChatConfig(thinking=False, temperature=0.2)
    payload = _sent_payload(monkeypatch, "claude-opus-5-5", cfg)

    assert "thinking" not in payload
    assert "output_config" not in payload
    assert "temperature" not in payload


def test_sonnet_4_6_thinking_off_still_sends_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = ChatConfig(thinking=False, temperature=0.2)
    payload = _sent_payload(monkeypatch, "claude-sonnet-4-6", cfg)

    assert "thinking" not in payload
    assert payload["temperature"] == 0.2


@pytest.mark.parametrize(
    ("model", "mode", "accepts_effort", "accepts_sampling"),
    [
        ("claude-opus-5-5", "adaptive", True, False),
        ("claude-opus-5.5", "adaptive", True, False),
        ("anthropic/claude-sonnet-5.5", "adaptive", True, False),
        ("claude-sonnet-5", "adaptive", True, False),
        ("claude-opus-5", "adaptive", True, False),
        ("claude-fable-5-1", "adaptive", True, False),
        ("claude-opus-4-8", "adaptive", True, False),
        ("claude-opus-4-7", "adaptive", True, False),
        ("claude-opus-4-6", "adaptive", True, True),
        ("us.anthropic.claude-sonnet-4-6-v1:0", "adaptive", True, True),
        ("claude-sonnet-4-5-20250929", "budget", False, True),
        ("claude-opus-4-5", "budget", False, True),
        ("claude-haiku-4-5-20251001", "budget", False, True),
        ("claude-sonnet-4-20250514", "budget", False, True),
        ("claude-3-7-sonnet-20250219", "budget", False, True),
        ("claude-3-5-sonnet-20241022", "none", False, True),
        ("claude-some-future-model", "adaptive", True, False),
        ("MiniMax-M2", "budget", False, True),
    ],
)
def test_claude_capabilities_table(
    model: str, mode: str, accepts_effort: bool, accepts_sampling: bool
) -> None:
    caps = claude_capabilities(model)
    assert caps.thinking_mode == mode
    assert caps.accepts_effort is accepts_effort
    assert caps.accepts_sampling is accepts_sampling


def test_known_models_list_current_claude_ids() -> None:
    provider = AnthropicProvider(api_key="test-key")
    models = {m.model_id: m for m in asyncio.run(provider.list_models())}

    for model_id in (
        "claude-opus-5-5",
        "claude-sonnet-5-5",
        "claude-fable-5-1",
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-opus-4-8",
        "claude-opus-4-6",
        "claude-sonnet-4-6",
    ):
        assert models[model_id].context_window == 1_000_000
    assert models["claude-haiku-4-5"].context_window == 200_000
    assert "claude-haiku-4-5-20251001" in models
    assert provider.model == "claude-sonnet-4-6"
