"""Anthropic payloads: cache breakpoints, per-block thinking replay, tool contract.

Offline: a MockTransport records every request body and answers with a canned
SSE stream, so the assertions are on the exact bytes the provider would send.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from agentos.provider import (
    ChatConfig,
    ContentBlockRedactedThinking,
    ContentBlockText,
    ContentBlockThinking,
    ContentBlockToolResult,
    ContentBlockToolUse,
    DoneEvent,
    Message,
    ToolChoice,
    ToolDefinition,
    ToolInputSchema,
)
from agentos.provider.anthropic import AnthropicProvider, _build_tool_choice_payload


def _sse(*events: dict[str, Any]) -> bytes:
    return b"".join(f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n".encode() for ev in events)


_END_TURN = _sse(
    {"type": "message_start", "message": {"usage": {"input_tokens": 3}}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
    {"type": "content_block_stop", "index": 0},
    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
    {"type": "message_stop"},
)


class _Recorder:
    def __init__(self, bodies: list[bytes] | None = None) -> None:
        self.payloads: list[dict[str, Any]] = []
        self.headers: list[httpx.Headers] = []
        self._bodies = list(bodies or [])

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.payloads.append(json.loads(request.content))
        self.headers.append(request.headers)
        body = self._bodies.pop(0) if self._bodies else _END_TURN
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)


def _install(monkeypatch: pytest.MonkeyPatch, recorder: _Recorder) -> None:
    transport = httpx.MockTransport(recorder.handler)
    real_async_client = httpx.AsyncClient

    def patched(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr("agentos.provider.anthropic.httpx.AsyncClient", patched)


def _run(coro_factory: Callable[[], Any]) -> Any:
    return asyncio.run(coro_factory())


async def _collect(
    provider: AnthropicProvider,
    messages: list[Message],
    cfg: ChatConfig,
    tools: list[ToolDefinition] | None = None,
) -> list[Any]:
    return [ev async for ev in provider.chat(messages, tools=tools, config=cfg)]


_TOOLS = [
    ToolDefinition(
        name="read_file",
        description="Read a file.",
        input_schema=ToolInputSchema(properties={"path": {"type": "string"}}, required=["path"]),
    ),
    ToolDefinition(
        name="write_file",
        description="Write a file.",
        input_schema=ToolInputSchema(
            properties={"path": {"type": "string"}, "text": {"type": "string"}},
            required=["path", "text"],
        ),
    ),
]


def _strip_cache_control(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_cache_control(v) for k, v in value.items() if k != "cache_control"}
    if isinstance(value, list):
        return [_strip_cache_control(v) for v in value]
    return value


def _cache_markers(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []

    def walk(where: str, value: Any) -> None:
        if isinstance(value, dict):
            if "cache_control" in value:
                found.append((where, value["cache_control"]))
            for k, v in value.items():
                walk(f"{where}.{k}", v)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                walk(f"{where}[{i}]", v)

    for key in ("tools", "system", "messages"):
        walk(key, payload.get(key))
    return found


_CACHED_CFG = ChatConfig(
    system="You are AgentOS.",
    cache_breakpoints=[{"text": "You are AgentOS.", "cache": "true"}],
    cache_mode="auto",
)


def _tool_loop_messages() -> tuple[list[Message], list[Message]]:
    """Request N ends on a tool result; request N+1 appends the next exchange."""
    first = [
        Message(role="user", content="Read README.md"),
        Message(
            role="assistant",
            content=[
                ContentBlockThinking(thinking="Need the file.", signature="sig-1"),
                ContentBlockToolUse(id="toolu_1", name="read_file", input={"path": "README.md"}),
            ],
        ),
        Message(
            role="user",
            content=[ContentBlockToolResult(tool_use_id="toolu_1", content="# AgentOS")],
        ),
    ]
    second = [
        *first,
        Message(
            role="assistant",
            content=[
                ContentBlockToolUse(
                    id="toolu_2", name="write_file", input={"path": "a", "text": "b"}
                )
            ],
        ),
        Message(
            role="user",
            content=[ContentBlockToolResult(tool_use_id="toolu_2", content="written")],
        ),
    ]
    return first, second


def test_breakpoints_on_system_last_tool_and_last_user_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder()
    _install(monkeypatch, recorder)
    provider = AnthropicProvider(api_key="k", model="claude-opus-5-5")
    first, _ = _tool_loop_messages()

    _run(lambda: _collect(provider, first, _CACHED_CFG, _TOOLS))

    payload = recorder.payloads[0]
    markers = _cache_markers(payload)
    assert [where for where, _ in markers] == [
        "tools[1]",
        "system[0]",
        "messages[2].content[0]",
    ]
    assert all(cc == {"type": "ephemeral"} for _, cc in markers)
    assert "cache_control" not in payload["tools"][0]
    assert payload["messages"][2]["content"][0]["type"] == "tool_result"


def test_consecutive_requests_share_a_byte_identical_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request N+1 repeats request N up to the new turn; only the rolling
    marker moves. This is what lets iteration N+1 read N's prefix from cache."""
    recorder = _Recorder()
    _install(monkeypatch, recorder)
    provider = AnthropicProvider(api_key="k", model="claude-opus-5-5")
    first, second = _tool_loop_messages()

    _run(lambda: _collect(provider, first, _CACHED_CFG, _TOOLS))
    _run(lambda: _collect(provider, second, _CACHED_CFG, _TOOLS))

    n, n1 = recorder.payloads
    assert json.dumps(n["tools"]) == json.dumps(n1["tools"])
    assert json.dumps(n["system"]) == json.dumps(n1["system"])
    prefix = json.dumps(_strip_cache_control(n["messages"]))
    assert json.dumps(_strip_cache_control(n1["messages"][: len(n["messages"])])) == prefix
    # The rolling marker moved to the new last user turn.
    assert "cache_control" in n1["messages"][-1]["content"][-1]
    assert "cache_control" not in n1["messages"][2]["content"][0]


def test_string_user_content_is_sent_as_a_block_so_the_shape_never_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder()
    _install(monkeypatch, recorder)
    provider = AnthropicProvider(api_key="k", model="claude-opus-5-5")
    history = [Message(role="user", content="hello")]

    _run(lambda: _collect(provider, history, _CACHED_CFG))
    _run(
        lambda: _collect(
            provider,
            [*history, Message(role="assistant", content="hi"), Message(role="user", content="x")],
            _CACHED_CFG,
        )
    )

    n, n1 = recorder.payloads
    assert n["messages"][0]["content"] == [
        {"type": "text", "text": "hello", "cache_control": {"type": "ephemeral"}}
    ]
    assert n1["messages"][0]["content"] == [{"type": "text", "text": "hello"}]


def test_cache_mode_off_sends_no_cache_control(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder()
    _install(monkeypatch, recorder)
    provider = AnthropicProvider(api_key="k", model="claude-opus-5-5")
    first, _ = _tool_loop_messages()

    _run(lambda: _collect(provider, first, ChatConfig(system="sys", cache_mode="off"), _TOOLS))

    payload = recorder.payloads[0]
    assert _cache_markers(payload) == []
    assert payload["system"] == "sys"
    assert payload["messages"][0]["content"] == "Read README.md"


def test_cache_mode_auto_without_breakpoints_still_caches_the_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder()
    _install(monkeypatch, recorder)
    provider = AnthropicProvider(api_key="k", model="claude-opus-5-5")

    _run(
        lambda: _collect(
            provider,
            [Message(role="user", content="hi")],
            ChatConfig(system="sys", cache_mode="on"),
        )
    )

    assert recorder.payloads[0]["system"] == [
        {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}
    ]


def test_one_hour_ttl_on_the_stable_prefix_only(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder()
    _install(monkeypatch, recorder)
    provider = AnthropicProvider(api_key="k", model="claude-opus-5-5")
    first, _ = _tool_loop_messages()
    cfg = _CACHED_CFG.model_copy(update={"cache_ttl": "1h"})

    _run(lambda: _collect(provider, first, cfg, _TOOLS))

    markers = dict(_cache_markers(recorder.payloads[0]))
    one_hour = {"type": "ephemeral", "ttl": "1h"}
    assert markers["tools[1]"] == one_hour
    assert markers["system[0]"] == one_hour
    # The rolling tail moves every request: 5 minute default, and longer TTLs
    # come first as Anthropic requires.
    assert markers["messages[2].content[0]"] == {"type": "ephemeral"}
    # 1h TTL is GA: no beta header needed.
    assert "anthropic-beta" not in recorder.headers[0]


def test_breakpoints_respect_the_limit_of_four(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder()
    _install(monkeypatch, recorder)
    provider = AnthropicProvider(api_key="k", model="claude-opus-5-5")
    cfg = ChatConfig(
        system="a\nb\nc",
        cache_mode="on",
        cache_breakpoints=[
            {"text": "a", "cache": "true"},
            {"text": "b", "cache": "true"},
            {"text": "c", "cache": "true"},
        ],
    )

    _run(lambda: _collect(provider, [Message(role="user", content="hi")], cfg, _TOOLS))

    markers = _cache_markers(recorder.payloads[0])
    assert len(markers) == 4
    # The rolling marker covers the tools too, so it wins the last slot.
    assert markers[-1][0] == "messages[0].content[0]"
    assert "cache_control" not in recorder.payloads[0]["tools"][-1]


_THINKING_STREAM = _sse(
    {"type": "message_start", "message": {"usage": {"input_tokens": 3}}},
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "thinking", "thinking": "", "signature": ""},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "thinking_delta", "thinking": "A1"},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "thinking_delta", "thinking": "A2"},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "signature_delta", "signature": "sigA"},
    },
    {"type": "content_block_stop", "index": 0},
    {
        "type": "content_block_start",
        "index": 1,
        "content_block": {"type": "redacted_thinking", "data": "ENCRYPTED=="},
    },
    {"type": "content_block_stop", "index": 1},
    {
        "type": "content_block_start",
        "index": 2,
        "content_block": {"type": "thinking", "thinking": "", "signature": ""},
    },
    {
        "type": "content_block_delta",
        "index": 2,
        "delta": {"type": "thinking_delta", "thinking": "B"},
    },
    {
        "type": "content_block_delta",
        "index": 2,
        "delta": {"type": "signature_delta", "signature": "sigB"},
    },
    {"type": "content_block_stop", "index": 2},
    {
        "type": "content_block_start",
        "index": 3,
        "content_block": {"type": "tool_use", "id": "toolu_9", "name": "read_file", "input": {}},
    },
    {
        "type": "content_block_delta",
        "index": 3,
        "delta": {"type": "input_json_delta", "partial_json": '{"path": "x"}'},
    },
    {"type": "content_block_stop", "index": 3},
    {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}},
    {"type": "message_stop"},
)


def test_thinking_blocks_stream_separately_with_their_own_signatures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder([_THINKING_STREAM])
    _install(monkeypatch, recorder)
    provider = AnthropicProvider(api_key="k", model="claude-opus-5-5")

    events = _run(
        lambda: _collect(
            provider, [Message(role="user", content="go")], ChatConfig(thinking=True), _TOOLS
        )
    )

    done = next(ev for ev in events if isinstance(ev, DoneEvent))
    assert done.thinking_blocks == [
        ContentBlockThinking(thinking="A1A2", signature="sigA"),
        ContentBlockRedactedThinking(data="ENCRYPTED=="),
        ContentBlockThinking(thinking="B", signature="sigB"),
    ]
    # Older callers keep the joined text and the last signature.
    assert done.reasoning_content == "A1A2B"
    assert done.thinking_signature == "sigB"


def test_per_block_thinking_round_trips_unmodified_and_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder([_THINKING_STREAM])
    _install(monkeypatch, recorder)
    provider = AnthropicProvider(api_key="k", model="claude-opus-5-5")
    cfg = ChatConfig(thinking=True)
    user = Message(role="user", content="go")

    events = _run(lambda: _collect(provider, [user], cfg, _TOOLS))
    done = next(ev for ev in events if isinstance(ev, DoneEvent))
    # What the engine stores: the thinking blocks first, then the tool call.
    assistant = Message(
        role="assistant",
        content=[
            *done.thinking_blocks,
            ContentBlockToolUse(id="toolu_9", name="read_file", input={"path": "x"}),
        ],
    )
    result = Message(
        role="user", content=[ContentBlockToolResult(tool_use_id="toolu_9", content="data")]
    )
    _run(lambda: _collect(provider, [user, assistant, result], cfg, _TOOLS))

    replayed = recorder.payloads[1]["messages"][1]["content"]
    assert replayed == [
        {"type": "thinking", "thinking": "A1A2", "signature": "sigA"},
        {"type": "redacted_thinking", "data": "ENCRYPTED=="},
        {"type": "thinking", "thinking": "B", "signature": "sigB"},
        {"type": "tool_use", "id": "toolu_9", "name": "read_file", "input": {"path": "x"}},
    ]


def test_legacy_single_string_thinking_still_loads_and_replays() -> None:
    from agentos.engine.session_sanitize import sanitize_session_messages
    from agentos.provider.anthropic import _build_message_payload

    stored = Message.model_validate(
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "old joined text", "signature": "s"},
                {"type": "text", "text": "answer"},
            ],
            "reasoning_content": "old joined text",
        }
    )
    redacted = Message.model_validate(
        {"role": "assistant", "content": [{"type": "redacted_thinking", "data": "Zm9v"}]}
    )
    # As persisted history can carry it: bookkeeping keys on a raw block dict.
    persisted = Message.model_construct(
        role="assistant",
        content=[{"type": "redacted_thinking", "data": "Zm9v", "_meta": {"ts": 1}}],
    )

    assert _build_message_payload(stored)["content"][0] == {
        "type": "thinking",
        "thinking": "old joined text",
        "signature": "s",
    }
    assert _build_message_payload(redacted)["content"] == [
        {"type": "redacted_thinking", "data": "Zm9v"}
    ]
    sanitized, _ = sanitize_session_messages([stored, redacted, persisted])
    assert sanitized[2].content == [ContentBlockRedactedThinking(data="Zm9v")]


def test_drop_reasoning_also_drops_redacted_thinking() -> None:
    from agentos.engine.thinking import drop_reasoning

    msg = Message(
        role="assistant",
        content=[ContentBlockRedactedThinking(data="x"), ContentBlockText(text="hi")],
    )

    out = drop_reasoning([msg])

    assert [b.type for b in out[0].content if not isinstance(out[0].content, str)] == ["text"]


def test_strict_tool_and_additional_properties_pass_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder()
    _install(monkeypatch, recorder)
    provider = AnthropicProvider(api_key="k", model="claude-opus-5-5")
    strict_tool = ToolDefinition(
        name="emit",
        description="Emit.",
        strict=True,
        input_schema=ToolInputSchema(
            properties={"x": {"type": "string"}}, required=["x"], additionalProperties=False
        ),
    )

    _run(
        lambda: _collect(
            provider, [Message(role="user", content="hi")], ChatConfig(), [strict_tool, *_TOOLS]
        )
    )

    sent = recorder.payloads[0]["tools"]
    assert sent[0]["strict"] is True
    assert sent[0]["input_schema"]["additionalProperties"] is False
    # Tools that do not opt in are unchanged.
    assert "strict" not in sent[1]
    assert "additionalProperties" not in sent[1]["input_schema"]


def test_typed_tool_choice_and_disable_parallel_tool_use() -> None:
    assert _build_tool_choice_payload(ToolChoice(type="tool", name="emit")) == {
        "type": "tool",
        "name": "emit",
    }
    assert _build_tool_choice_payload(ToolChoice(type="any")) == {"type": "any"}
    assert _build_tool_choice_payload("required") == {"type": "any"}
    assert _build_tool_choice_payload(ToolChoice(type="auto", disable_parallel_tool_use=True)) == {
        "type": "auto",
        "disable_parallel_tool_use": True,
    }
    assert _build_tool_choice_payload(None, parallel_tool_calls=False) == {
        "type": "auto",
        "disable_parallel_tool_use": True,
    }
    assert _build_tool_choice_payload({"type": "any"}, parallel_tool_calls=False) == {
        "type": "any",
        "disable_parallel_tool_use": True,
    }
    assert _build_tool_choice_payload("none", parallel_tool_calls=False) == {"type": "none"}
    assert _build_tool_choice_payload(None) is None


def test_beta_headers_merge_provider_and_request_values(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder()
    _install(monkeypatch, recorder)
    provider = AnthropicProvider(api_key="k", model="claude-opus-5-5", betas=["beta-a"])

    _run(
        lambda: _collect(
            provider,
            [Message(role="user", content="hi")],
            ChatConfig(anthropic_betas=["beta-b", "beta-a", " "]),
        )
    )
    _run(
        lambda: _collect(
            AnthropicProvider(api_key="k"), [Message(role="user", content="hi")], ChatConfig()
        )
    )

    assert recorder.headers[0]["anthropic-beta"] == "beta-a,beta-b"
    assert "anthropic-beta" not in recorder.headers[1]
