"""OpenAI-compatible payloads: reasoning_details replay, prefix stability, tool contract."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from agentos.provider import (
    ChatConfig,
    ContentBlockToolResult,
    ContentBlockToolUse,
    DoneEvent,
    Message,
    ModelCapabilities,
    ToolChoice,
    ToolDefinition,
    ToolInputSchema,
)
from agentos.provider.openai import (
    OpenAIProvider,
    _build_openai_tool_choice,
    _merge_reasoning_detail,
)


def _sse(chunks: list[dict[str, Any]], *, done: bool = True) -> bytes:
    body = b"".join(f"data: {json.dumps(c)}\n\n".encode() for c in chunks)
    return body + (b"data: [DONE]\n\n" if done else b"")


_STOP = _sse(
    [
        {"choices": [{"delta": {"content": "ok"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 1}},
    ]
)


class _Recorder:
    def __init__(self, bodies: list[bytes] | None = None) -> None:
        self.payloads: list[dict[str, Any]] = []
        self._bodies = list(bodies or [])

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.payloads.append(json.loads(request.content))
        body = self._bodies.pop(0) if self._bodies else _STOP
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)


_REAL_ASYNC_CLIENT = httpx.AsyncClient


def _install(monkeypatch: pytest.MonkeyPatch, recorder: _Recorder) -> None:
    transport = httpx.MockTransport(recorder.handler)

    def patched(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return _REAL_ASYNC_CLIENT(*args, **kwargs)

    monkeypatch.setattr("agentos.provider.openai.httpx.AsyncClient", patched)


def _collect(
    provider: OpenAIProvider,
    messages: list[Message],
    cfg: ChatConfig,
    tools: list[ToolDefinition] | None = None,
) -> list[Any]:
    async def run() -> list[Any]:
        return [ev async for ev in provider.chat(messages, tools=tools, config=cfg)]

    return asyncio.run(run())


_TOOL = ToolDefinition(
    name="read_file",
    description="Read a file.",
    input_schema=ToolInputSchema(properties={"path": {"type": "string"}}, required=["path"]),
)

_OPENROUTER_CAPS = ModelCapabilities(
    supports_reasoning=True, supports_tools=True, reasoning_format="openrouter"
)


def _openrouter(model: str = "anthropic/claude-opus-5.5") -> OpenAIProvider:
    return OpenAIProvider(
        api_key="k",
        model=model,
        base_url="https://openrouter.ai/api/v1",
        provider_kind="openrouter",
    )


_REASONING_TOOL_STREAM = _sse(
    [
        {
            "choices": [
                {
                    "delta": {
                        "reasoning_details": [
                            {
                                "type": "reasoning.text",
                                "text": "Need ",
                                "index": 0,
                                "format": "anthropic-claude-v1",
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "reasoning_details": [
                            {"type": "reasoning.text", "text": "the file.", "index": 0},
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "reasoning_details": [
                            {"type": "reasoning.text", "signature": "sig-xyz", "index": 0},
                            {
                                "type": "reasoning.encrypted",
                                "data": "BLOB",
                                "index": 1,
                                "format": "anthropic-claude-v1",
                            },
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "function": {"name": "read_file", "arguments": '{"path": "a"}'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        },
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]
)


def test_streamed_reasoning_details_are_merged_per_block() -> None:
    details: list[dict[str, Any]] = []
    for fragment in (
        {"type": "reasoning.text", "text": "a", "index": 0, "format": "f"},
        {"type": "reasoning.text", "text": "b", "index": 0},
        {"type": "reasoning.text", "signature": "s", "index": 0},
        {"type": "reasoning.summary", "summary": "sum", "index": 1},
    ):
        _merge_reasoning_detail(details, fragment)

    assert details == [
        {"type": "reasoning.text", "text": "ab", "index": 0, "format": "f", "signature": "s"},
        {"type": "reasoning.summary", "summary": "sum", "index": 1},
    ]


def test_openrouter_reasoning_details_round_trip_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder([_REASONING_TOOL_STREAM])
    _install(monkeypatch, recorder)
    provider = _openrouter()
    cfg = ChatConfig(thinking=True, model_capabilities=_OPENROUTER_CAPS)
    user = Message(role="user", content="read a")

    events = _collect(provider, [user], cfg, [_TOOL])
    done = next(ev for ev in events if isinstance(ev, DoneEvent))
    expected = [
        {
            "type": "reasoning.text",
            "text": "Need the file.",
            "index": 0,
            "format": "anthropic-claude-v1",
            "signature": "sig-xyz",
        },
        {
            "type": "reasoning.encrypted",
            "data": "BLOB",
            "index": 1,
            "format": "anthropic-claude-v1",
        },
    ]
    assert done.reasoning_details == expected

    # What the engine stores, then the next request of the tool loop.
    assistant = Message(
        role="assistant",
        content=[ContentBlockToolUse(id="call_1", name="read_file", input={"path": "a"})],
        reasoning_content=done.reasoning_content,
        reasoning_details=done.reasoning_details,
    )
    result = Message(
        role="user", content=[ContentBlockToolResult(tool_use_id="call_1", content="data")]
    )
    _collect(provider, [user, assistant, result], cfg, [_TOOL])

    replayed = recorder.payloads[1]["messages"][1]
    assert replayed["reasoning_details"] == expected
    # The structured form supersedes the plain-text copy.
    assert "reasoning_content" not in replayed


_CLAUDE_DETAILS = [
    {
        "type": "reasoning.text",
        "text": "t",
        "signature": "claude-sig",
        "index": 0,
        "format": "anthropic-claude-v1",
    }
]


def _replayed_assistant(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    details: list[dict[str, Any]],
    caps: ModelCapabilities | None = _OPENROUTER_CAPS,
) -> dict[str, Any]:
    recorder = _Recorder()
    _install(monkeypatch, recorder)
    assistant = Message(
        role="assistant",
        content=[ContentBlockToolUse(id="call_1", name="read_file", input={"path": "a"})],
        reasoning_content="t",
        reasoning_details=details,
    )
    result = Message(
        role="user", content=[ContentBlockToolResult(tool_use_id="call_1", content="A")]
    )
    _collect(
        _openrouter(model),
        [Message(role="user", content="a"), assistant, result],
        ChatConfig(model_capabilities=caps),
        [_TOOL],
    )
    return recorder.payloads[0]["messages"][1]


def test_reasoning_details_replay_to_the_same_family(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _replayed_assistant(monkeypatch, "anthropic/claude-sonnet-5.5", _CLAUDE_DETAILS)

    assert sent["reasoning_details"] == _CLAUDE_DETAILS


def test_reasoning_details_are_dropped_after_a_model_family_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A router or /model switch must not hand Claude's signed blocks to GPT."""
    to_gpt = _replayed_assistant(monkeypatch, "openai/gpt-6-luna", _CLAUDE_DETAILS)
    to_glm = _replayed_assistant(monkeypatch, "z-ai/glm-5.3", _CLAUDE_DETAILS)
    glm_details = [{"type": "reasoning.text", "text": "t", "index": 0, "format": "unknown"}]
    glm_to_claude = _replayed_assistant(monkeypatch, "anthropic/claude-opus-5.5", glm_details)
    glm_to_glm = _replayed_assistant(monkeypatch, "z-ai/glm-5.3", glm_details)
    mixed = _replayed_assistant(
        monkeypatch,
        "anthropic/claude-opus-5.5",
        [*_CLAUDE_DETAILS, {"type": "reasoning.encrypted", "data": "X", "format": "openai-v1"}],
    )

    for sent in (to_gpt, to_glm, glm_to_claude, mixed):
        assert "reasoning_details" not in sent
        assert "claude-sig" not in json.dumps(sent)
    assert glm_to_glm["reasoning_details"] == glm_details


def test_reasoning_details_respect_the_reasoning_replay_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent = _replayed_assistant(
        monkeypatch, "anthropic/claude-opus-5.5", _CLAUDE_DETAILS, caps=ModelCapabilities()
    )

    assert "reasoning_details" not in sent
    assert "reasoning_content" not in sent


def test_reasoning_details_are_not_sent_to_other_openai_compatible_upstreams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder()
    _install(monkeypatch, recorder)
    provider = OpenAIProvider(
        api_key="k",
        model="some-model",
        base_url="https://api.opencap.ai/v1",
        provider_kind="opencap",
    )
    assistant = Message(
        role="assistant", content="hi", reasoning_details=[{"type": "reasoning.text", "text": "t"}]
    )

    _collect(provider, [Message(role="user", content="a"), assistant], ChatConfig())

    assert "reasoning_details" not in recorder.payloads[0]["messages"][1]


def test_consecutive_requests_share_a_byte_identical_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder()
    _install(monkeypatch, recorder)
    provider = _openrouter("openai/gpt-6-luna")
    cfg = ChatConfig(system="You are AgentOS.", model_capabilities=_OPENROUTER_CAPS)
    first = [
        Message(role="user", content="read a"),
        Message(
            role="assistant",
            content=[ContentBlockToolUse(id="call_1", name="read_file", input={"path": "a"})],
            reasoning_details=[
                {
                    "type": "reasoning.encrypted",
                    "data": "X",
                    "index": 0,
                    "format": "openai-responses-v1",
                }
            ],
        ),
        Message(role="user", content=[ContentBlockToolResult(tool_use_id="call_1", content="A")]),
    ]
    second = [
        *first,
        Message(role="assistant", content="done"),
        Message(role="user", content="thanks"),
    ]

    _collect(provider, first, cfg, [_TOOL])
    _collect(provider, second, cfg, [_TOOL])

    n, n1 = recorder.payloads
    assert json.dumps(n1["messages"][: len(n["messages"])]) == json.dumps(n["messages"])
    assert json.dumps(n1["tools"]) == json.dumps(n["tools"])


def test_parallel_tool_calls_strict_and_typed_tool_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder()
    _install(monkeypatch, recorder)
    provider = _openrouter("openai/gpt-6-luna")
    strict_tool = ToolDefinition(
        name="emit",
        description="Emit.",
        strict=True,
        input_schema=ToolInputSchema(
            properties={"x": {"type": "string"}}, required=["x"], additionalProperties=False
        ),
    )
    cfg = ChatConfig(tool_choice=ToolChoice(type="tool", name="emit"), parallel_tool_calls=False)

    _collect(provider, [Message(role="user", content="a")], cfg, [strict_tool, _TOOL])
    _collect(provider, [Message(role="user", content="a")], ChatConfig(), [_TOOL])

    forced, plain = recorder.payloads
    assert forced["tool_choice"] == {"type": "function", "function": {"name": "emit"}}
    assert forced["parallel_tool_calls"] is False
    assert forced["tools"][0]["function"]["strict"] is True
    assert forced["tools"][0]["function"]["parameters"]["additionalProperties"] is False
    assert "strict" not in forced["tools"][1]["function"]
    # Nothing opted in: the payload is unchanged.
    assert "parallel_tool_calls" not in plain
    assert "tool_choice" not in plain


def test_tool_choice_translation_table() -> None:
    forced = {"type": "function", "function": {"name": "emit_route"}}
    assert _build_openai_tool_choice(forced) == forced
    assert _build_openai_tool_choice({"type": "tool", "name": "x"}) == {
        "type": "function",
        "function": {"name": "x"},
    }
    assert _build_openai_tool_choice({"type": "any"}) == "required"
    assert _build_openai_tool_choice("any") == "required"
    assert _build_openai_tool_choice(ToolChoice(type="none")) == "none"
    assert _build_openai_tool_choice(ToolChoice()) == "auto"
    assert _build_openai_tool_choice(None) is None


def test_disable_parallel_on_tool_choice_maps_to_parallel_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder()
    _install(monkeypatch, recorder)
    provider = _openrouter("openai/gpt-6-luna")
    cfg = ChatConfig(tool_choice=ToolChoice(type="auto", disable_parallel_tool_use=True))

    _collect(provider, [Message(role="user", content="a")], cfg, [_TOOL])

    assert recorder.payloads[0]["parallel_tool_calls"] is False
    assert recorder.payloads[0]["tool_choice"] == "auto"


def test_prompt_cache_key_only_for_api_openai_com(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder()
    _install(monkeypatch, recorder)
    direct = OpenAIProvider(api_key="k", model="gpt-6", base_url="https://api.openai.com")
    cfg = ChatConfig(system="sys", cache_mode="auto")

    _collect(direct, [Message(role="user", content="a")], cfg, [_TOOL])
    _collect(direct, [Message(role="user", content="b")], cfg, [_TOOL])
    _collect(
        direct,
        [Message(role="user", content="a")],
        cfg.model_copy(update={"prompt_cache_key": "sess-1"}),
    )
    _collect(
        direct, [Message(role="user", content="a")], ChatConfig(system="sys", cache_mode="off")
    )
    for provider in (
        _openrouter("openai/gpt-6-luna"),
        OpenAIProvider(
            api_key="k", model="m", base_url="https://api.opencap.ai/v1", provider_kind="opencap"
        ),
        OpenAIProvider(api_key="k", model="m", base_url="http://localhost:8000/v1"),
    ):
        _collect(provider, [Message(role="user", content="a")], cfg, [_TOOL])

    derived_a, derived_b, explicit, off, *others = recorder.payloads
    assert derived_a["prompt_cache_key"].startswith("agentos-")
    # Same system prompt and tools -> same routing key across turns.
    assert derived_a["prompt_cache_key"] == derived_b["prompt_cache_key"]
    assert explicit["prompt_cache_key"] == "sess-1"
    assert "prompt_cache_key" not in off
    assert all("prompt_cache_key" not in payload for payload in others)
