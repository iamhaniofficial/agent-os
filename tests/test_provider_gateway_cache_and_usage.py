"""OpenAI-compatible gateways: Anthropic cache hints per upstream, split usage chunks.

Recorded shapes come from live OpenCAP (gw.capminal.ai) captures of
claude-opus-5 and gpt-5.6-luna turns.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from agentos.engine.cache_break_monitor import CacheBreakMonitor
from agentos.provider import (
    ChatConfig,
    ContentBlockToolResult,
    ContentBlockToolUse,
    DoneEvent,
    Message,
    ToolDefinition,
    ToolInputSchema,
)
from agentos.provider.openai import OpenAIProvider

_REAL_ASYNC_CLIENT = httpx.AsyncClient
_OPENCAP = "https://gw.capminal.ai/api/inference/v1"


def _sse(chunks: list[dict[str, Any]]) -> bytes:
    body = b"".join(f"data: {json.dumps(c)}\n\n".encode() for c in chunks)
    return body + b"data: [DONE]\n\n"


_PLAIN_STOP = _sse(
    [
        {"choices": [{"delta": {"content": "ok"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
)


class _Recorder:
    def __init__(self, body: bytes = _PLAIN_STOP) -> None:
        self.payloads: list[dict[str, Any]] = []
        self.body = body

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            self.payloads.append(json.loads(request.content))
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=self.body
            )

        transport = httpx.MockTransport(handler)

        def patched(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
            kwargs["transport"] = transport
            return _REAL_ASYNC_CLIENT(*args, **kwargs)

        monkeypatch.setattr("agentos.provider.openai.httpx.AsyncClient", patched)


def _run(provider: OpenAIProvider, messages: list[Message], cfg: ChatConfig) -> list[Any]:
    async def run() -> list[Any]:
        return [ev async for ev in provider.chat(messages, tools=[_TOOL], config=cfg)]

    return asyncio.run(run())


_TOOL = ToolDefinition(
    name="read_file",
    description="Read.",
    input_schema=ToolInputSchema(properties={"path": {"type": "string"}}),
)
_CFG = ChatConfig(
    system="You are AgentOS.",
    cache_mode="auto",
    cache_breakpoints=[{"text": "You are AgentOS.", "cache": "true"}],
)


def _markers(payload: dict[str, Any]) -> list[str]:
    found: list[str] = []

    def walk(where: str, value: Any) -> None:
        if isinstance(value, dict):
            if "cache_control" in value:
                found.append(where)
            for key, item in value.items():
                if key != "cache_control":
                    walk(f"{where}.{key}", item)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(f"{where}[{index}]", item)

    walk("$", payload)
    return found


def _strip_markers(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _strip_markers(v) for k, v in value.items() if k != "cache_control"}
    if isinstance(value, list):
        return [_strip_markers(v) for v in value]
    return value


def _text_of(content: Any) -> Any:
    """Content as the upstream sees it: a lone text part equals its string."""
    if isinstance(content, list) and len(content) == 1 and content[0].get("type") == "text":
        return content[0]["text"]
    return content


def _tool_loop() -> tuple[list[Message], list[Message]]:
    first = [
        Message(role="user", content="Read README.md"),
        Message(
            role="assistant",
            content=[ContentBlockToolUse(id="call_1", name="read_file", input={"path": "R"})],
        ),
        Message(role="user", content=[ContentBlockToolResult(tool_use_id="call_1", content="# A")]),
    ]
    second = [
        *first,
        Message(role="assistant", content="Done."),
        Message(role="user", content="thanks"),
    ]
    return first, second


# --- cache hints per upstream ---------------------------------------------


def test_opencap_claude_gets_a_rolling_marker_on_the_last_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder()
    recorder.install(monkeypatch)
    provider = OpenAIProvider(
        api_key="k", model="claude-opus-5", base_url=_OPENCAP, provider_kind="opencap"
    )
    first, second = _tool_loop()

    _run(provider, first, _CFG)
    _run(provider, second, _CFG)

    n, n1 = recorder.payloads
    # Tool loop: the newest input is the tool result message.
    assert _markers(n) == ["$.messages[3].content[0]"]
    assert n["messages"][3]["role"] == "tool"
    assert n["messages"][3]["content"] == [
        {"type": "text", "text": "# A", "cache_control": {"type": "ephemeral"}}
    ]
    # Next request: the marker moved to the new last user message; nothing on
    # the system prompt, tools or the top level (OpenCAP caches those itself).
    assert _markers(n1) == ["$.messages[5].content[0]"]
    assert "cache_control" not in n1
    # Same prompt up to the new turn once the marker is set aside.
    prefix = [{**m, "content": _text_of(m.get("content"))} for m in _strip_markers(n["messages"])]
    replay = [
        {**m, "content": _text_of(m.get("content"))}
        for m in _strip_markers(n1["messages"][: len(n["messages"])])
    ]
    assert replay == prefix


def test_opencap_matched_by_host_for_a_custom_provider_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _Recorder()
    recorder.install(monkeypatch)
    provider = OpenAIProvider(api_key="k", model="claude-sonnet-5-5", base_url=_OPENCAP)

    _run(provider, [Message(role="user", content="hi")], _CFG)

    assert _markers(recorder.payloads[0]) == ["$.messages[1].content[0]"]


@pytest.mark.parametrize(
    ("provider_kind", "base_url", "model"),
    [
        ("opencap", _OPENCAP, "gpt-5.6-luna"),  # not a Claude model
        ("openai", "https://api.openai.com", "claude-opus-5"),
        ("deepseek", "https://api.deepseek.com", "deepseek-v4-flash"),
        ("gemini", "https://generativelanguage.googleapis.com/v1beta/openai", "gemini-3-pro"),
        ("zhipu", "https://open.bigmodel.cn/api/paas/v4", "glm-5"),
        ("openai", "http://localhost:8000/v1", "claude-local"),
    ],
)
def test_no_anthropic_cache_hints_for_other_upstreams(
    monkeypatch: pytest.MonkeyPatch, provider_kind: str, base_url: str, model: str
) -> None:
    recorder = _Recorder()
    recorder.install(monkeypatch)
    provider = OpenAIProvider(
        api_key="k", model=model, base_url=base_url, provider_kind=provider_kind
    )
    first, _ = _tool_loop()

    _run(provider, first, _CFG)

    assert _markers(recorder.payloads[0]) == []
    assert recorder.payloads[0]["messages"][3]["content"] == "# A"


def test_opencap_cache_mode_off_sends_no_hints(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _Recorder()
    recorder.install(monkeypatch)
    provider = OpenAIProvider(
        api_key="k", model="claude-opus-5", base_url=_OPENCAP, provider_kind="opencap"
    )

    _run(provider, [Message(role="user", content="hi")], ChatConfig(cache_mode="off"))

    assert _markers(recorder.payloads[0]) == []


def test_openrouter_claude_keeps_the_top_level_automatic_breakpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenRouter turns a top-level cache_control into Anthropic's automatic
    caching, which already advances with the history: no extra block markers."""
    recorder = _Recorder()
    recorder.install(monkeypatch)
    provider = OpenAIProvider(
        api_key="k",
        model="anthropic/claude-opus-5.5",
        base_url="https://openrouter.ai/api/v1",
        provider_kind="openrouter",
    )
    first, _ = _tool_loop()

    _run(provider, first, _CFG)

    payload = recorder.payloads[0]
    assert payload["cache_control"] == {"type": "ephemeral"}
    assert not any(m.startswith("$.messages[3]") for m in _markers(payload))


def test_rolling_marker_respects_the_limit_of_four() -> None:
    from agentos.provider.openai import _mark_rolling_cache_breakpoint

    full = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": str(i), "cache_control": {"type": "ephemeral"}}
                for i in range(4)
            ],
        },
        {"role": "user", "content": "hi"},
    ]

    assert _mark_rolling_cache_breakpoint(full) is False
    assert full[1]["content"] == "hi"


# --- split usage chunks ---------------------------------------------------

_CLAUDE_FINISH = {
    "choices": [{"delta": {"role": "assistant"}, "finish_reason": "stop", "index": 0}],
    "model": "claude-opus-5",
    "usage": {
        "prompt_tokens": 36456,
        "completion_tokens": 111,
        "total_tokens": 36567,
        "prompt_tokens_details": {"cached_tokens": 32809, "cache_creation_tokens": 3646},
    },
}
_TEXT = {"choices": [{"delta": {"content": "ok"}, "finish_reason": None, "index": 0}]}


def _usage_done(monkeypatch: pytest.MonkeyPatch, chunks: list[dict[str, Any]]) -> DoneEvent:
    recorder = _Recorder(_sse(chunks))
    recorder.install(monkeypatch)
    provider = OpenAIProvider(
        api_key="k", model="claude-opus-5", base_url=_OPENCAP, provider_kind="opencap"
    )
    events = _run(provider, [Message(role="user", content="hi")], ChatConfig())
    return next(ev for ev in events if isinstance(ev, DoneEvent))


def test_trailing_all_zero_usage_chunk_does_not_erase_the_real_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trailing_zero = {
        "choices": [],
        "object": "chat.completion.chunk",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }

    done = _usage_done(monkeypatch, [_TEXT, _CLAUDE_FINISH, trailing_zero])

    assert (done.input_tokens, done.output_tokens) == (36456, 111)
    assert (done.cached_tokens, done.cache_write_tokens) == (32809, 3646)
    assert done.cache_usage_reported is True


def test_trailing_billing_chunk_without_details_keeps_cached_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    billing = {
        "choices": [],
        "object": "chat.completion.chunk",
        "usage": {
            "completion_tokens": 111,
            "cost": 0.000044,
            "prompt_tokens": 36456,
            "total_tokens": 36567,
            "buyer_cost_micro": 50,
        },
    }

    done = _usage_done(monkeypatch, [_TEXT, _CLAUDE_FINISH, billing])

    assert done.input_tokens == 36456
    assert done.cached_tokens == 32809
    assert done.cache_write_tokens == 3646


def test_estimated_usage_never_overrides_the_real_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    estimated = {
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 25746, "completion_tokens": 0, "estimated": True},
    }
    real = {
        "choices": [],
        "usage": {
            "prompt_tokens": 20750,
            "completion_tokens": 176,
            "prompt_tokens_details": {"cached_tokens": 16805},
        },
    }

    done = _usage_done(monkeypatch, [_TEXT, estimated, real])

    assert (done.input_tokens, done.output_tokens, done.cached_tokens) == (20750, 176, 16805)


def test_usage_without_any_cache_figure_is_reported_as_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bare = {"choices": [], "usage": {"prompt_tokens": 106, "completion_tokens": 158}}

    done = _usage_done(
        monkeypatch, [_TEXT, {"choices": [{"delta": {}, "finish_reason": "stop"}]}, bare]
    )

    assert done.cached_tokens == 0
    assert done.cache_usage_reported is False


# --- cache-break monitor --------------------------------------------------


def _snapshot(monitor: CacheBreakMonitor, messages: list[Message]) -> Any:
    return monitor.record_prompt_state(
        messages=messages, tools=None, config=ChatConfig(system="s"), model="m"
    )


def test_cache_break_monitor_skips_unreported_usage_and_keeps_its_baseline() -> None:
    monitor = CacheBreakMonitor()
    history = [Message(role="user", content=f"m{i}") for i in range(4)]
    monitor.check_response_for_cache_break("s", _snapshot(monitor, history[:2]), 20000)

    unknown = monitor.check_response_for_cache_break(
        "s", _snapshot(monitor, history[:3]), 0, usage_reported=False
    )
    after = monitor.check_response_for_cache_break("s", _snapshot(monitor, history), 21000)

    assert unknown.break_detected is False
    assert unknown.reason == "cache_usage_unreported"
    # The next real response is compared with the last real one (20000).
    assert after.previous_cache_read_tokens == 20000
    assert after.break_detected is False


def test_cache_break_monitor_still_reports_a_real_drop() -> None:
    monitor = CacheBreakMonitor()
    monitor.check_response_for_cache_break(
        "s", _snapshot(monitor, [Message(role="user", content="a")] * 3), 20000
    )

    report = monitor.check_response_for_cache_break(
        "s", _snapshot(monitor, [Message(role="user", content="b")] * 3), 0
    )

    assert report.break_detected is True
