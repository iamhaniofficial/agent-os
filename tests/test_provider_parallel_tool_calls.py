"""``ChatConfig.parallel_tool_calls=True`` reaches only APIs known to accept it."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from agentos.provider import (
    ChatConfig,
    Message,
    ToolChoice,
    ToolDefinition,
    ToolInputSchema,
)
from agentos.provider.anthropic import _build_tool_choice_payload
from agentos.provider.openai import OpenAIProvider, _accepts_parallel_tool_calls_opt_in

_STOP = (
    b'data: {"choices": [{"delta": {"content": "ok"}, "finish_reason": null}]}\n\n'
    b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}], '
    b'"usage": {"prompt_tokens": 1}}\n\n'
    b"data: [DONE]\n\n"
)

_TOOL = ToolDefinition(
    name="read_file",
    description="Read a file.",
    input_schema=ToolInputSchema(properties={"path": {"type": "string"}}, required=["path"]),
)

_REAL_ASYNC_CLIENT = httpx.AsyncClient


def _payload(
    monkeypatch: pytest.MonkeyPatch,
    provider: OpenAIProvider,
    cfg: ChatConfig,
    tools: list[ToolDefinition] | None = None,
) -> dict[str, Any]:
    payloads: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_STOP)

    transport = httpx.MockTransport(handler)

    def patched(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return _REAL_ASYNC_CLIENT(*args, **kwargs)

    monkeypatch.setattr("agentos.provider.openai.httpx.AsyncClient", patched)

    async def run() -> None:
        async for _ in provider.chat(
            [Message(role="user", content="a")],
            tools=[_TOOL] if tools is None else tools,
            config=cfg,
        ):
            pass

    asyncio.run(run())
    return payloads[0]


def _provider(base_url: str, kind: str, model: str = "gpt-6") -> OpenAIProvider:
    return OpenAIProvider(api_key="k", model=model, base_url=base_url, provider_kind=kind)


_ACCEPTING = [
    ("https://openrouter.ai/api/v1", "openrouter", "anthropic/claude-opus-5.5"),
    ("https://openrouter.ai/api/v1", "openrouter", "deepseek/deepseek-v4.1-flash"),
    ("https://api.openai.com/v1", "openai", "gpt-6"),
    ("https://api.openai.com/v1", "openai", "gpt-5.6-luna"),
]

_REJECTING_OR_UNKNOWN = [
    ("https://api.openai.com/v1", "openai", "o3-mini"),
    ("https://api.openai.com/v1", "openai", "o4-mini"),
    ("https://api.opencap.ai/v1", "opencap", "gpt-6-luna"),
    ("https://llm.bankr.bot/v1", "bankr", "gpt-6-luna"),
    ("https://api.deepseek.com/v1", "deepseek", "deepseek-v4.1-flash"),
    ("https://open.bigmodel.cn/api/paas/v4", "zhipu", "glm-5.3"),
    ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini", "gemini-3-pro"),
    ("http://localhost:11434/v1", "ollama", "qwen3"),
    ("http://localhost:8000/v1", "openai", "local-model"),
]


@pytest.mark.parametrize(("base_url", "kind", "model"), _ACCEPTING)
def test_parallel_opt_in_sent_where_accepted(
    monkeypatch: pytest.MonkeyPatch, base_url: str, kind: str, model: str
) -> None:
    payload = _payload(
        monkeypatch, _provider(base_url, kind, model), ChatConfig(parallel_tool_calls=True)
    )
    assert payload["parallel_tool_calls"] is True


@pytest.mark.parametrize(("base_url", "kind", "model"), _REJECTING_OR_UNKNOWN)
def test_parallel_opt_in_dropped_where_rejected_or_unknown(
    monkeypatch: pytest.MonkeyPatch, base_url: str, kind: str, model: str
) -> None:
    assert not _accepts_parallel_tool_calls_opt_in(kind, base_url, model)
    payload = _payload(
        monkeypatch, _provider(base_url, kind, model), ChatConfig(parallel_tool_calls=True)
    )
    assert "parallel_tool_calls" not in payload


def test_explicit_serial_request_is_not_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    # False (one call per response) is a correctness request: unchanged from
    # before, it still goes to every OpenAI-compatible endpoint.
    local = _provider("http://localhost:8000/v1", "openai", "local-model")
    assert (
        _payload(monkeypatch, local, ChatConfig(parallel_tool_calls=False))["parallel_tool_calls"]
        is False
    )
    cfg = ChatConfig(
        parallel_tool_calls=True,
        tool_choice=ToolChoice(type="auto", disable_parallel_tool_use=True),
    )
    # A gated opt-in falls back to what the tool choice asks for.
    assert _payload(monkeypatch, local, cfg)["parallel_tool_calls"] is False


def test_parallel_opt_in_needs_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider("https://openrouter.ai/api/v1", "openrouter", "openai/gpt-6")
    payload = _payload(monkeypatch, provider, ChatConfig(parallel_tool_calls=True), tools=[])
    assert "parallel_tool_calls" not in payload


def test_unset_sends_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider("https://openrouter.ai/api/v1", "openrouter", "openai/gpt-6")
    assert "parallel_tool_calls" not in _payload(monkeypatch, provider, ChatConfig())


def test_anthropic_parallel_opt_in_sends_nothing() -> None:
    # Parallel tool use is the Anthropic default: True adds no tool_choice and
    # never a disable_parallel_tool_use flag.
    assert _build_tool_choice_payload(None, parallel_tool_calls=True) is None
    assert _build_tool_choice_payload("auto", parallel_tool_calls=True) == {"type": "auto"}
    assert _build_tool_choice_payload({"type": "any"}, parallel_tool_calls=True) == {"type": "any"}
