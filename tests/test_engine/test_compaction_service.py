"""The one CompactionService: trigger, shape, summarizer, safety rules.

Coverage moved here from the deleted before-turn paths (T3-upgrade,
DB preflight, emergency request-scoped trim): the threshold rules, the
cron/subagent session handling and the summarizer now live in one service
the agent calls before every provider request.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest

from agentos.engine.compaction_service import (
    CONTEXT_SUMMARY_HEADER,
    SUMMARY_ACK,
    CompactionService,
    ends_mid_tool_round,
    is_ephemeral_session,
    render_messages_for_summary,
    should_compact,
    summarize_messages,
)
from agentos.provider import (
    ChatConfig,
    ContentBlockRedactedThinking,
    ContentBlockText,
    ContentBlockThinking,
    ContentBlockToolResult,
    ContentBlockToolUse,
    Message,
)
from agentos.provider import DoneEvent as ProviderDone
from agentos.provider import ErrorEvent as ProviderError
from agentos.provider import TextDeltaEvent as ProviderText
from agentos.provider.types import ContentBlockImage


def _tokens(messages: Sequence[Message]) -> int:
    return sum(len(render_messages_for_summary([message])) for message in messages) // 4


def _round(tool_id: str, result: str) -> list[Message]:
    return [
        Message(
            role="assistant",
            content=[
                ContentBlockThinking(thinking=f"think {tool_id}", signature=f"sig-{tool_id}"),
                ContentBlockToolUse(id=tool_id, name="read_file", input={"path": tool_id}),
            ],
        ),
        Message(
            role="user",
            content=[ContentBlockToolResult(tool_use_id=tool_id, content=result)],
        ),
    ]


class _SummarizingProvider:
    """A minimal LLMProvider: the summarizer must call it, never raw HTTP."""

    provider_name = "fake"

    def __init__(self, *, text: str = "provider summary", error: bool = False) -> None:
        self.text = text
        self.error = error
        self.calls: list[tuple[list[Message], ChatConfig | None]] = []

    def chat(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
        config: ChatConfig | None = None,
    ) -> AsyncIterator[Any]:
        self.calls.append((messages, config))
        return self._stream()

    async def _stream(self) -> AsyncIterator[Any]:
        if self.error:
            yield ProviderError(message="upstream down", code="503")
            return
        yield ProviderText(text=self.text)
        yield ProviderDone(stop_reason="end_turn")


# ---------------------------------------------------------------------------
# Trigger
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("estimated", "window", "ratio", "expected"),
    [
        (849, 1000, 0.85, False),
        (850, 1000, 0.85, False),  # exactly at the threshold does not compact
        (851, 1000, 0.85, True),
        (601, 1000, 0.6, True),  # configured ratio is honoured
        (10_000, 0, 0.85, False),  # unknown window never triggers
    ],
)
def test_single_trigger_is_the_live_estimate_against_window_times_ratio(
    estimated: int, window: int, ratio: float, expected: bool
) -> None:
    assert should_compact(estimated, window, ratio) is expected
    assert CompactionService(provider=None, ratio=ratio).should_compact(estimated, window) is (
        expected
    )


@pytest.mark.asyncio
async def test_under_threshold_is_a_noop_without_summarizing() -> None:
    calls: list[Any] = []

    async def _summarizer(messages: list[Message], previous: str | None) -> tuple[str, str]:
        calls.append(messages)
        return "s", "llm"

    messages = [Message(role="user", content="q"), Message(role="assistant", content="a")]
    result = await CompactionService(provider=None, summarizer=_summarizer).compact(
        [*messages, Message(role="user", content="now")],
        turn_input_index=2,
        estimated_tokens=100,
        window_tokens=1000,
        message_tokens=_tokens,
    )

    assert result.status == "skipped"
    assert result.reason == "under_threshold"
    assert calls == []


@pytest.mark.parametrize(
    ("key", "ephemeral"),
    [
        ("subagent:agent:main:main", True),
        ("agent:main:subagent:run-7", True),  # canonical key a prefix test missed
        ("AGENT:Main:SUBAGENT:Run-7", True),
        ("cron:daily-digest", True),
        ("agent:main:main", False),
        ("agent:main:webchat:abc", False),
        ("", False),
        (None, False),
    ],
)
def test_ephemeral_sessions_use_is_subagent_key(key: str | None, ephemeral: bool) -> None:
    assert is_ephemeral_session(key) is ephemeral


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shape_is_one_summary_exchange_plus_the_current_turn_verbatim() -> None:
    seen: list[list[Message]] = []

    async def _summarizer(messages: list[Message], previous: str | None) -> tuple[str, str]:
        seen.append(messages)
        return "condensed", "llm"

    image = ContentBlockImage(media_type="image/png", data="aGVsbG8=")
    history = [
        Message(role="user", content="old question " + "h" * 4000),
        Message(role="assistant", content="old answer " + "a" * 4000),
    ]
    current = [
        Message(role="user", content=[ContentBlockText(text="look at this"), image]),
        *_round("t1", "result one"),
    ]
    messages = [*history, *current]

    result = await CompactionService(provider=None, summarizer=_summarizer).compact(
        messages,
        turn_input_index=2,
        estimated_tokens=50_000,
        window_tokens=10_000,
        message_tokens=_tokens,
        input_entry={"role": "user", "content": "look at this"},
    )

    assert result.compacted
    assert result.messages[0] == Message(
        role="user", content=f"{CONTEXT_SUMMARY_HEADER}\ncondensed"
    )
    assert result.messages[1] == Message(role="assistant", content=SUMMARY_ACK)
    # The current turn is kept object-for-object: image, thinking signature,
    # tool pair -- nothing flattened or re-projected.
    assert result.messages[2:] == current
    assert all(a is b for a, b in zip(result.messages[2:], current, strict=True))
    assert result.turn_input_index == 2
    assert result.removed_count == 2
    assert seen == [history]
    assert result.kept_entries == [{"role": "user", "content": "look at this"}]


@pytest.mark.asyncio
async def test_previous_summary_is_folded_into_the_next_one() -> None:
    previous: list[str | None] = []

    async def _summarizer(messages: list[Message], prev: str | None) -> tuple[str, str]:
        previous.append(prev)
        return "merged summary", "llm"

    messages = [
        Message(role="user", content=f"{CONTEXT_SUMMARY_HEADER}\nfirst summary"),
        Message(role="assistant", content=SUMMARY_ACK),
        Message(role="user", content="q " + "x" * 4000),
        Message(role="assistant", content="a " + "y" * 4000),
        Message(role="user", content="now"),
    ]

    result = await CompactionService(provider=None, summarizer=_summarizer).compact(
        messages,
        turn_input_index=4,
        estimated_tokens=50_000,
        window_tokens=10_000,
        message_tokens=_tokens,
    )

    assert previous == ["first summary"]
    assert result.messages[0].content == f"{CONTEXT_SUMMARY_HEADER}\nmerged summary"
    assert [m.content for m in result.messages[2:]] == ["now"]


@pytest.mark.asyncio
async def test_oversized_current_turn_folds_oldest_complete_rounds_only() -> None:
    folded: list[list[Message]] = []

    async def _summarizer(messages: list[Message], previous: str | None) -> tuple[str, str]:
        folded.append(messages)
        return "s", "llm"

    current = [
        Message(role="user", content="do the big task"),
        *_round("r1", "a" * 20_000),
        *_round("r2", "b" * 20_000),
        *_round("r3", "c" * 2_000),
    ]
    messages = [Message(role="user", content="old"), Message(role="assistant", content="old a")]

    result = await CompactionService(provider=None, summarizer=_summarizer).compact(
        [*messages, *current],
        turn_input_index=2,
        estimated_tokens=40_000,
        window_tokens=10_000,
        message_tokens=_tokens,
    )

    assert result.compacted
    kept = result.messages[2:]
    # Input message and the latest round survive; r1/r2 were summarized.
    assert kept[0] == current[0]
    assert kept[1:] == current[-2:]
    assert result.folded_rounds == 2
    assert folded[0][2:] == current[1:5]
    # Never a tool_result without its tool_use.
    kept_use_ids = {
        block.id
        for message in kept
        if isinstance(message.content, list)
        for block in message.content
        if isinstance(block, ContentBlockToolUse)
    }
    kept_result_ids = {
        block.tool_use_id
        for message in kept
        if isinstance(message.content, list)
        for block in message.content
        if isinstance(block, ContentBlockToolResult)
    }
    assert kept_result_ids == kept_use_ids == {"r3"}


@pytest.mark.asyncio
async def test_never_compacts_mid_tool_round() -> None:
    calls: list[Any] = []

    async def _summarizer(messages: list[Message], previous: str | None) -> tuple[str, str]:
        calls.append(messages)
        return "s", "llm"

    messages = [
        Message(role="user", content="old " + "x" * 8000),
        Message(role="assistant", content="answer"),
        Message(role="user", content="now"),
        _round("pending", "unused")[0],  # tool_use whose result is not appended yet
    ]
    assert ends_mid_tool_round(messages)

    result = await CompactionService(provider=None, summarizer=_summarizer).compact(
        messages,
        turn_input_index=2,
        estimated_tokens=50_000,
        window_tokens=1_000,
        message_tokens=_tokens,
        force=True,
    )

    assert result.status == "skipped"
    assert result.reason == "mid_tool_round"
    assert result.messages is messages
    assert calls == []


@pytest.mark.asyncio
async def test_nothing_before_the_current_turn_is_a_noop() -> None:
    result = await CompactionService(provider=None).compact(
        [Message(role="user", content="x" * 40_000)],
        turn_input_index=0,
        estimated_tokens=50_000,
        window_tokens=1_000,
        message_tokens=_tokens,
        force=True,
    )

    assert result.status == "skipped"
    assert result.reason == "nothing_to_compact"


# ---------------------------------------------------------------------------
# Summarizer goes through LLMProvider
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summarizer_streams_through_the_provider_abstraction() -> None:
    provider = _SummarizingProvider(text="what happened so far")
    history = [
        Message(role="user", content="deploy to host-17 at 10.0.0.7:8443"),
        Message(
            role="assistant",
            content=[
                ContentBlockThinking(thinking="secret reasoning", signature="s1"),
                ContentBlockRedactedThinking(data="opaque"),
                ContentBlockText(text="deploying"),
            ],
        ),
    ]

    result = await CompactionService(provider=provider).compact(
        [*history, Message(role="user", content="status?")],
        turn_input_index=2,
        estimated_tokens=50_000,
        window_tokens=1_000,
        message_tokens=_tokens,
    )

    assert result.compacted
    assert result.summary == "what happened so far"
    assert result.summary_source == "llm"
    assert len(provider.calls) == 1
    sent, config = provider.calls[0]
    assert "10.0.0.7:8443" in str(sent[0].content)
    # Reasoning is not summarized; thinking is off for the summarizer call.
    assert "secret reasoning" not in str(sent[0].content)
    assert config is not None and config.thinking is False
    assert "identifiers" in str(config.system)


@pytest.mark.asyncio
async def test_provider_failure_falls_back_to_a_deterministic_digest() -> None:
    provider = _SummarizingProvider(error=True)
    history = [
        Message(role="user", content="please remember order 4711"),
        Message(role="assistant", content="noted order 4711"),
    ]

    summary, source = await summarize_messages(provider, history, None, window_tokens=1_000)

    assert source == "fallback"
    assert "order 4711" in summary
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_summarizer_exception_is_a_failed_compaction_not_a_crash() -> None:
    async def _boom(messages: list[Message], previous: str | None) -> tuple[str, str]:
        raise RuntimeError("summarizer unavailable")

    messages = [
        Message(role="user", content="old " + "x" * 8000),
        Message(role="assistant", content="answer"),
        Message(role="user", content="now"),
    ]
    result = await CompactionService(provider=None, summarizer=_boom).compact(
        messages,
        turn_input_index=2,
        estimated_tokens=50_000,
        window_tokens=1_000,
        message_tokens=_tokens,
    )

    assert result.status == "failed"
    assert result.reason == "compaction_failed"
    assert result.messages is messages


@pytest.mark.asyncio
async def test_compaction_lifecycle_events_keep_their_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        "agentos.engine.compaction_service.notify_compaction",
        lambda session_key, **payload: events.append((session_key, payload)),
    )

    async def _summarizer(messages: list[Message], previous: str | None) -> tuple[str, str]:
        return "s", "llm"

    await CompactionService(
        provider=None, summarizer=_summarizer, session_key="agent:main:main"
    ).compact(
        [
            Message(role="user", content="old " + "x" * 8000),
            Message(role="assistant", content="answer"),
            Message(role="user", content="now"),
        ],
        turn_input_index=2,
        estimated_tokens=50_000,
        window_tokens=1_000,
        message_tokens=_tokens,
    )

    assert [payload["status"] for _, payload in events] == ["started", "observed", "observed"]
    assert {payload["phase"] for _, payload in events} == {"agent_inline_overflow"}
    assert [payload["event"] for _, payload in events] == [
        "compaction.triggered",
        "compaction.chunk_summarized",
        "compaction.summary_verified",
    ]
    assert len({payload["compaction_id"] for _, payload in events}) == 1
