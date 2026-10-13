"""Append-only structured transcript, end to end and at the codec level.

The real guard for prompt caching: two consecutive turns driven through the
``TurnRunner`` with a real ``SessionManager`` (persist + reload between the
turns), asserting every provider request starts with the previous request
byte for byte (system + tools + messages). The cache-break monitor keeps
logging; this test is what fails when a later request rewrites an earlier
message.

Fixtures are synthetic and anonymised (no real transcripts).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import agentos.engine.agent as agent_module
import agentos.engine.transcript as transcript_module
from agentos.engine.runtime import TurnRunner
from agentos.engine.transcript import (
    PROVIDER_MESSAGES_KIND,
    TranscriptRecord,
    build_history_from_transcript,
    decode_transcript_record,
    encode_transcript_record,
    persistable_messages,
)
from agentos.engine.types import DoneEvent
from agentos.execution_status import normalize_legacy_execution_status
from agentos.gateway.config import AgentOSRouterConfig, AttachmentsConfig, GatewayConfig
from agentos.provider import (
    ContentBlockRedactedThinking,
    ContentBlockText,
    ContentBlockThinking,
    ContentBlockToolResult,
    ContentBlockToolUse,
    Message,
)
from agentos.provider import DoneEvent as ProviderDone
from agentos.provider import TextDeltaEvent as ProviderText
from agentos.provider import ToolUseEndEvent as ProviderToolUseEnd
from agentos.provider import ToolUseStartEvent as ProviderToolUseStart
from agentos.provider.types import ContentBlockImage
from agentos.session.manager import SessionManager
from agentos.session.models import TranscriptEntry
from agentos.session.storage import SessionStorage
from agentos.tools.registry import ToolRegistry, ToolSpec
from agentos.tools.types import CallerKind, ToolContext

KEY = "agent:main:webchat:append-only"
# Over the 160k-char LOCAL tool-result budget of a 200k-token window.
BIG_OUTPUT = "LOOKUP-RESULT\n" + ("r" * 200_000)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class _ScriptedProvider:
    """Each turn: one tool call, then a text answer. Records every request."""

    provider_name = "test"

    def __init__(self, counter: dict[str, int] | None = None) -> None:
        self.model = "test/model"
        self.requests: list[dict[str, Any]] = []
        self.repair_calls_at_request: list[int] = []
        self._counter = counter

    def chat(self, messages: list[Message], tools=None, config=None) -> AsyncIterator[Any]:
        self.requests.append(
            {
                "system": getattr(config, "system", None),
                "tools": [tool.model_dump(mode="json") for tool in tools or []],
                "messages": [
                    json.dumps(message.model_dump(mode="json"), sort_keys=True)
                    for message in messages
                ],
            }
        )
        if self._counter is not None:
            self.repair_calls_at_request.append(self._counter["repair"])
        return self._stream(len(self.requests))

    async def _stream(self, call_number: int) -> AsyncIterator[Any]:
        if call_number % 2 == 1:
            tool_use_id = f"lookup-{call_number}"
            yield ProviderToolUseStart(tool_use_id=tool_use_id, tool_name="lookup")
            yield ProviderToolUseEnd(
                tool_use_id=tool_use_id,
                tool_name="lookup",
                arguments={"query": f"q{call_number}"},
            )
            yield ProviderDone(
                stop_reason="tool_use",
                input_tokens=10,
                output_tokens=1,
                thinking_blocks=[
                    ContentBlockThinking(
                        thinking=f"plan {call_number}", signature=f"sig-{call_number}"
                    ),
                    ContentBlockRedactedThinking(data=f"enc-{call_number}"),
                ],
                reasoning_content=f"plan {call_number}",
            )
            return
        yield ProviderText(text=f"answer {call_number}")
        yield ProviderDone(stop_reason="end_turn", input_tokens=10, output_tokens=1)

    async def list_models(self) -> list[Any]:
        return []


class _SelectorClone:
    current_config = SimpleNamespace(model="test/model")

    def __init__(self, provider: _ScriptedProvider) -> None:
        self.provider = provider

    def override_model(self, model: str) -> None:
        self.current_config = SimpleNamespace(model=model)

    def resolve(self) -> _ScriptedProvider:
        return self.provider


class _Selector:
    def __init__(self, provider: _ScriptedProvider) -> None:
        self.provider = provider

    def clone(self) -> _SelectorClone:
        return _SelectorClone(self.provider)


def _registry() -> ToolRegistry:
    registry = ToolRegistry()

    async def lookup(query: str) -> str:
        return BIG_OUTPUT if query == "q1" else f"small result for {query}"

    registry.register(
        ToolSpec(
            name="lookup",
            description="Look something up",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        ),
        lookup,
    )
    return registry


@pytest.fixture
async def manager(tmp_path: Path) -> AsyncIterator[SessionManager]:
    storage = SessionStorage(":memory:")
    await storage.connect()
    yield SessionManager(storage, inject_time_prefix=False, checkpoint_workspace_dir=tmp_path)
    await storage.close()


def _runner(provider: _ScriptedProvider, manager: SessionManager, tmp_path: Path) -> TurnRunner:
    return TurnRunner(
        provider_selector=_Selector(provider),
        tool_registry=_registry(),
        session_manager=manager,
        config=GatewayConfig(
            attachments=AttachmentsConfig(media_root=str(tmp_path / "media")),
            agentos_router=AgentOSRouterConfig(enabled=False),
        ),
    )


async def _turn(
    runner: TurnRunner, manager: SessionManager, text: str, tmp_path: Path
) -> list[Any]:
    # The gateway persists the user row before it runs the turn.
    await manager.append_message(KEY, "user", text)
    return [
        event
        async for event in runner.run(
            text,
            KEY,
            tool_context=ToolContext(caller_kind=CallerKind.WEB, workspace_dir=str(tmp_path)),
            history_has_persisted_user=True,
            no_memory_capture=True,
        )
    ]


def _is_prefix(shorter: list[str], longer: list[str]) -> bool:
    return longer[: len(shorter)] == shorter


# ---------------------------------------------------------------------------
# End to end: two turns, persisted and reloaded in between
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consecutive_requests_share_a_byte_identical_prefix_across_turns(
    manager: SessionManager,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter = {"repair": 0}
    real_repair = transcript_module.repair_tool_pairing

    def _counting_repair(messages: list[Message]) -> list[Message]:
        counter["repair"] += 1
        return real_repair(messages)

    monkeypatch.setattr(transcript_module, "repair_tool_pairing", _counting_repair)
    monkeypatch.setattr(agent_module, "repair_tool_pairing", _counting_repair)

    provider = _ScriptedProvider(counter)
    runner = _runner(provider, manager, tmp_path)
    await manager.create(KEY)

    first = await _turn(runner, manager, "find the first thing", tmp_path)
    second = await _turn(runner, manager, "and now the second", tmp_path)

    assert any(isinstance(event, DoneEvent) for event in first)
    assert any(isinstance(event, DoneEvent) for event in second)
    requests = provider.requests
    assert len(requests) == 4

    # System prompt and tools never change between requests.
    assert len({request["system"] for request in requests}) == 1
    assert all(request["tools"] == requests[0]["tools"] for request in requests)
    # Every request starts with the previous request, byte for byte -- within
    # a turn AND across the persist/reload between turns.
    for earlier, later in zip(requests, requests[1:], strict=False):
        assert _is_prefix(earlier["messages"], later["messages"])

    # Across turns the prefix grows by exactly the previous final answer.
    turn_one_final = json.loads(requests[2]["messages"][len(requests[1]["messages"])])
    assert turn_one_final["role"] == "assistant"
    assert turn_one_final["content"][0]["text"] == "answer 2"

    # One pairing-repair pass per request at most (load once, then window only).
    deltas = [
        after - before
        for before, after in zip(
            [0, *provider.repair_calls_at_request], provider.repair_calls_at_request, strict=False
        )
    ]
    assert all(delta <= 1 for delta in deltas), deltas


@pytest.mark.asyncio
async def test_oversized_tool_result_is_shrunk_at_append_and_never_again(
    manager: SessionManager,
    tmp_path: Path,
) -> None:
    provider = _ScriptedProvider()
    runner = _runner(provider, manager, tmp_path)
    await manager.create(KEY)

    await _turn(runner, manager, "find the first thing", tmp_path)
    await _turn(runner, manager, "and now the second", tmp_path)

    def _result(request: dict[str, Any], tool_use_id: str) -> str:
        for raw in request["messages"]:
            message = json.loads(raw)
            if isinstance(message["content"], list):
                for block in message["content"]:
                    if block.get("tool_use_id") == tool_use_id:
                        return str(block["content"])
        raise AssertionError(f"{tool_use_id} not in request")

    first_send = _result(provider.requests[1], "lookup-1")
    assert "[tool_result_projection]" in first_send
    assert BIG_OUTPUT not in first_send
    assert len(first_send) < len(BIG_OUTPUT)
    # The same bytes on every later request, in this turn and the next one.
    assert _result(provider.requests[2], "lookup-1") == first_send
    assert _result(provider.requests[3], "lookup-1") == first_send


@pytest.mark.asyncio
async def test_new_format_is_written_next_to_display_segments(
    manager: SessionManager,
    tmp_path: Path,
) -> None:
    provider = _ScriptedProvider()
    runner = _runner(provider, manager, tmp_path)
    await manager.create(KEY)

    await _turn(runner, manager, "find the first thing", tmp_path)

    rows = await manager.get_transcript(KEY)
    user_row, assistant_row = rows[-2], rows[-1]
    assert assistant_row.role == "assistant"
    # Flat display segments are still written for the UI / memory readers.
    assert [segment["type"] for segment in assistant_row.tool_calls or []] == [
        "tool_use",
        "tool_result",
        "text",
    ]
    record = assistant_row.provider_messages
    assert isinstance(record, dict)
    assert record["kind"] == PROVIDER_MESSAGES_KIND
    assert record["v"] == 1
    assert record["mode"] == "append"
    assert record["input_entry_id"] == user_row.message_id
    roles = [message["role"] for message in record["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]
    # The input message carries its timestamp block; per-block thinking keeps
    # each signature / redacted payload, in stream order, before the tool call.
    assert record["messages"][0]["content"].startswith("find the first thing\n\n")
    assert "[Runtime context for this turn]" in record["messages"][0]["content"]
    assert [block["type"] for block in record["messages"][1]["content"]] == [
        "thinking",
        "redacted_thinking",
        "tool_use",
    ]
    assert record["messages"][1]["content"][0]["signature"] == "sig-1"
    assert record["messages"][1]["content"][1]["data"] == "enc-1"


@pytest.mark.asyncio
async def test_timestamp_lives_in_the_current_turn_and_is_never_rewritten(
    manager: SessionManager,
    tmp_path: Path,
) -> None:
    provider = _ScriptedProvider()
    runner = _runner(provider, manager, tmp_path)
    await manager.create(KEY)

    await _turn(runner, manager, "find the first thing", tmp_path)
    await _turn(runner, manager, "and now the second", tmp_path)

    last = [json.loads(raw) for raw in provider.requests[-1]["messages"]]
    with_timestamp = [
        index
        for index, message in enumerate(last)
        if message["role"] == "user"
        and isinstance(message["content"], str)
        and "[Runtime context for this turn]" in message["content"]
    ]
    # One block per user turn, each inside that turn's input message.
    assert [last[index]["content"].split("\n\n")[0] for index in with_timestamp] == [
        "find the first thing",
        "and now the second",
    ]
    first_turn_input = json.loads(provider.requests[0]["messages"][0])
    assert last[0] == first_turn_input


# ---------------------------------------------------------------------------
# Loading: both formats
# ---------------------------------------------------------------------------


def _row(role: str, content: str, **fields: Any) -> TranscriptEntry:
    return TranscriptEntry(session_id="s1", session_key=KEY, role=role, content=content, **fields)


def test_legacy_rows_load_through_the_heuristic_reconstruction() -> None:
    rows = [
        _row("user", "old question"),
        _row(
            "assistant",
            "old answer",
            tool_calls=[
                {"type": "tool_use", "tool_use_id": "t1", "name": "lookup", "input": {"q": 1}},
                {"type": "tool_result", "tool_use_id": "t1", "result": "found"},
                {"type": "text", "text": "old answer"},
            ],
        ),
    ]

    loaded = build_history_from_transcript(rows)

    assert loaded.legacy_entries == 2
    assert loaded.structured_entries == 0
    assert [message.role for message in loaded.messages] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert isinstance(loaded.messages[1].content[0], ContentBlockToolUse)


def test_structured_rows_replay_verbatim_and_replace_their_input_row() -> None:
    turn = [
        Message(role="user", content="new question\n\n[Runtime context for this turn]\nnow"),
        Message(
            role="assistant",
            content=[
                ContentBlockThinking(thinking="t", signature="sig"),
                ContentBlockText(text="new answer"),
            ],
            reasoning_details=[{"type": "reasoning.encrypted", "data": "x", "index": 0}],
        ),
    ]
    user_row = _row("user", "new question")
    record = encode_transcript_record(
        TranscriptRecord(messages=turn, input_entry_id=user_row.message_id)
    )
    rows = [
        _row("user", "old question"),
        _row("assistant", "old answer"),
        user_row,
        _row("assistant", "new answer", provider_messages=record),
    ]
    legacy_seen: list[list[Message]] = []

    loaded = build_history_from_transcript(
        rows, legacy_migrate=lambda messages: legacy_seen.append(messages) or messages
    )

    assert loaded.messages == [
        Message(role="user", content="old question"),
        Message(role="assistant", content="old answer"),
        *turn,
    ]
    assert loaded.structured_entries == 1
    # Only legacy rows are migrated; the structured record is not touched.
    assert [m.content for batch in legacy_seen for m in batch] == ["old question", "old answer"]


def test_reset_record_replaces_everything_before_it() -> None:
    compacted = [
        Message(role="user", content="[Context summary]\nall of it"),
        Message(role="assistant", content="Understood. Continuing from summary."),
        Message(role="user", content="current"),
        Message(role="assistant", content="reply"),
    ]
    rows = [
        _row("user", "very old"),
        _row("assistant", "very old answer"),
        _row("user", "current"),
        _row(
            "assistant",
            "reply",
            id=42,
            provider_messages=encode_transcript_record(
                TranscriptRecord(messages=compacted, mode="reset", compaction_id="cmp-1")
            ),
        ),
        _row("user", "next"),
    ]

    loaded = build_history_from_transcript(rows)

    assert loaded.messages == [*compacted, Message(role="user", content="next")]
    assert loaded.reset_entry_id == 42
    assert loaded.last_entry_was_user is True


def test_corrupt_or_unknown_record_falls_back_to_legacy() -> None:
    rows = [
        _row("user", "q"),
        _row("assistant", "a", provider_messages={"kind": PROVIDER_MESSAGES_KIND, "v": 99}),
        _row("assistant", "b", provider_messages={"kind": "something-else", "messages": []}),
    ]

    loaded = build_history_from_transcript(rows)

    assert [message.content for message in loaded.messages] == ["q", "a", "b"]
    assert loaded.legacy_entries == 3


def test_record_codec_round_trips_every_block_kind() -> None:
    messages = [
        Message(role="user", content="hi"),
        Message(
            role="assistant",
            content=[
                ContentBlockThinking(thinking="", signature="sig-empty"),
                ContentBlockThinking(thinking="second", signature="sig-2"),
                ContentBlockRedactedThinking(data="opaque=="),
                ContentBlockText(text="calling"),
                ContentBlockToolUse(
                    id="t1",
                    name="lookup",
                    input={"b": 2, "a": [1, {"z": None}]},
                    thought_signature="ts",
                ),
            ],
            reasoning_content="joined",
            reasoning_details=[{"type": "reasoning.text", "text": "r", "index": 0}],
        ),
        Message(
            role="user",
            content=[
                ContentBlockToolResult(
                    tool_use_id="t1",
                    content="done",
                    is_error=False,
                    execution_status=normalize_legacy_execution_status(is_error=False),
                )
            ],
        ),
    ]
    record = TranscriptRecord(messages=messages, input_entry_id="u-1", compaction_id=None)

    stored = json.loads(json.dumps(encode_transcript_record(record)))
    decoded = decode_transcript_record(stored)

    assert decoded is not None
    assert decoded.messages == messages
    assert decoded.input_entry_id == "u-1"
    # Byte-identical provider payloads after the round trip.
    assert [m.model_dump(mode="json") for m in decoded.messages] == [
        m.model_dump(mode="json") for m in messages
    ]


def test_images_are_stored_as_markers_not_bytes() -> None:
    message = Message(
        role="user",
        content=[
            ContentBlockText(text="what is this"),
            ContentBlockImage(media_type="image/png", data="aGVsbG8="),
        ],
    )

    (stored,) = persistable_messages([message])

    assert stored.content == [
        ContentBlockText(text="what is this"),
        ContentBlockText(text="[historical image omitted: image/png]"),
    ]


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_storage_adds_the_column_to_an_existing_database(tmp_path: Path) -> None:
    import aiosqlite

    db_path = tmp_path / "old.db"
    async with aiosqlite.connect(db_path) as conn:
        await conn.execute(
            """
            CREATE TABLE transcript_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                session_key TEXT NOT NULL,
                message_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_calls TEXT,
                tool_call_id TEXT,
                created_at INTEGER NOT NULL,
                token_count INTEGER,
                provenance_kind TEXT,
                provenance_origin_session_id TEXT,
                provenance_source_session_key TEXT,
                provenance_source_channel TEXT,
                provenance_source_tool TEXT,
                schema_version INTEGER NOT NULL DEFAULT 1
            )
            """
        )
        await conn.execute(
            "INSERT INTO transcript_entries (session_id, session_key, message_id, role, "
            "content, created_at) VALUES ('s1', ?, 'm1', 'user', 'legacy row', 1)",
            (KEY,),
        )
        await conn.commit()

    storage = SessionStorage(str(db_path))
    await storage.connect()
    try:
        async with storage.conn.execute("PRAGMA table_info(transcript_entries)") as cur:
            columns = {row[1] for row in await cur.fetchall()}
        assert "provider_messages" in columns
        (old,) = await storage.get_transcript("s1")
        assert old.content == "legacy row"
        assert old.provider_messages is None
    finally:
        await storage.close()


@pytest.mark.asyncio
async def test_fork_points_the_record_at_the_child_copy_of_its_input_row(
    manager: SessionManager,
) -> None:
    await manager.create(KEY)
    user_row = await manager.append_message(KEY, "user", "question")
    record = encode_transcript_record(
        TranscriptRecord(
            messages=[
                Message(role="user", content="question\n\n[Runtime context for this turn]\nx"),
                Message(role="assistant", content="answer"),
            ],
            input_entry_id=user_row.message_id,
        )
    )
    await manager.append_message(KEY, "assistant", "answer", provider_messages=record)

    child_key = "agent:main:webchat:append-only-fork"
    await manager.branch(KEY, child_key, fork_transcript=True)

    child_rows = await manager.get_transcript(child_key)
    loaded = build_history_from_transcript(child_rows)
    assert child_rows[1].provider_messages["input_entry_id"] == child_rows[0].message_id
    # The input row is replayed once, from the record.
    assert [message.content for message in loaded.messages] == [
        "question\n\n[Runtime context for this turn]\nx",
        "answer",
    ]
