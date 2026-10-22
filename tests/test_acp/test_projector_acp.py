"""ACP event -> engine event projection, transcript shape and usage math."""

from __future__ import annotations

from types import SimpleNamespace

from agentos.acp.projector import (
    COST_READING_KEY,
    HARNESS_COST_SOURCE,
    MAX_ASSISTANT_TEXT_CHARS,
    AcpProjector,
    DoneEvent,
    TextDeltaEvent,
    ToolResultEvent,
    ToolUseStartEvent,
    cost_delta,
    error_event,
    format_stats,
    last_turn_record,
    stats_for_session,
    usage_update_fields,
)
from agentos.acp.types import AcpEvent, AcpTurnResult, AcpTurnTimeoutError, AcpUsage
from agentos.engine import types as engine_types
from agentos.gateway.acp_dispatch import to_engine_event


def _feed(projector: AcpProjector, *events: AcpEvent) -> list:
    out: list = []
    for event in events:
        out.extend(projector.feed(event))
    return out


def test_final_only_buffers_text_until_done() -> None:
    p = AcpProjector(stream_delivery="final_only")
    out = _feed(p, AcpEvent(kind="text_delta", text="Hel"), AcpEvent(kind="text_delta", text="lo"))
    assert out == []
    final = p.final_events(
        AcpTurnResult(stop_reason="end_turn", usage=AcpUsage(input_tokens=10, output_tokens=4)),
        model="acp:claude",
        turn_cost=0.25,
    )
    assert isinstance(final[0], TextDeltaEvent) and final[0].text == "Hello"
    done = final[-1]
    assert isinstance(done, DoneEvent)
    assert (done.text, done.input_tokens, done.output_tokens) == ("Hello", 10, 4)
    assert done.cost_usd == 0.25 and done.cost_source == HARNESS_COST_SOURCE
    assert done.model == "acp:claude"


def test_live_streams_deltas_and_done_has_no_duplicate_text() -> None:
    p = AcpProjector(stream_delivery="live")
    out = _feed(p, AcpEvent(kind="text_delta", text="a"), AcpEvent(kind="text_delta", text="b"))
    assert [e.text for e in out] == ["a", "b"]
    final = p.final_events(AcpTurnResult(stop_reason="end_turn"))
    assert len(final) == 1 and isinstance(final[0], DoneEvent)
    assert final[0].cost_source == "none"


def test_hidden_tags_produce_nothing() -> None:
    p = AcpProjector()
    out = _feed(
        p,
        AcpEvent(kind="thought_delta", text="thinking"),
        AcpEvent(kind="plan", raw={"entries": []}),
        AcpEvent(kind="status", tag="available_commands_update", raw={"x": "y" * 5000}),
        AcpEvent(kind="mode", tag="current_mode_update"),
    )
    assert out == []
    assert p.text == ""


def test_tool_call_lifecycle_maps_to_start_and_result() -> None:
    p = AcpProjector()
    out = _feed(
        p,
        AcpEvent(kind="text_delta", text="Editing."),
        AcpEvent(
            kind="tool_call",
            tool_call_id="t1",
            tool_kind="edit",
            title="Edit main.py",
            status="pending",
            raw_input={"path": "main.py"},
        ),
        AcpEvent(kind="tool_call_update", tool_call_id="t1", status="in_progress"),
        AcpEvent(
            kind="tool_call_update",
            tool_call_id="t1",
            status="completed",
            raw_output={"ok": True},
        ),
        AcpEvent(kind="text_delta", text="Done."),
    )
    assert isinstance(out[0], ToolUseStartEvent)
    # The tool row is named with its human label (the UIs title cards by name).
    assert (out[0].tool_use_id, out[0].tool_name) == ("t1", "Edit main.py")
    result = out[1]
    assert isinstance(result, ToolResultEvent)
    assert result.tool_name == "Edit main.py" and not result.is_error
    assert result.arguments == {"path": "main.py", "title": "Edit main.py", "acp_kind": "edit"}
    assert '"ok": true' in result.result
    segments = p.transcript_segments()
    assert [s["type"] for s in segments] == ["text", "tool_use", "tool_result", "text"]
    assert segments[1]["name"] == "Edit main.py"
    assert segments[1]["input"] == {"path": "main.py", "title": "Edit main.py", "acp_kind": "edit"}
    # Narration resumed after the tool call is its own paragraph.
    assert p.text == "Editing.\n\nDone."


def test_failed_tool_and_unknown_update_synthesize_start() -> None:
    p = AcpProjector()
    out = _feed(
        p,
        AcpEvent(
            kind="tool_call_update",
            tool_call_id="x9",
            tool_kind="execute",
            status="failed",
            error="exit 1",
        ),
    )
    assert isinstance(out[0], ToolUseStartEvent)
    assert isinstance(out[1], ToolResultEvent) and out[1].is_error
    assert out[1].result == "exit 1"
    # Completing twice emits nothing new.
    again = p.feed(AcpEvent(kind="tool_call_update", tool_call_id="x9", status="completed"))
    assert again == []


def test_unfinished_tool_gets_synthetic_result_in_transcript() -> None:
    p = AcpProjector()
    _feed(p, AcpEvent(kind="tool_call", tool_call_id="a", tool_kind="read"))
    segments = p.transcript_segments()
    assert segments[-1]["type"] == "tool_result"
    assert segments[-1]["tool_use_id"] == "a"


def test_plain_text_turn_has_no_segments() -> None:
    p = AcpProjector()
    _feed(p, AcpEvent(kind="text_delta", text="just text"))
    assert p.transcript_segments() == []


def test_text_is_capped_with_marker() -> None:
    p = AcpProjector(stream_delivery="live", max_text_chars=10)
    out = _feed(
        p, AcpEvent(kind="text_delta", text="x" * 8), AcpEvent(kind="text_delta", text="y" * 7)
    )
    assert "".join(e.text for e in out) == "x" * 8 + "yy"
    assert p.truncated
    assert p.text.startswith("x" * 8 + "yy")
    assert "5 more characters" in p.text
    final = p.final_events(AcpTurnResult(stop_reason="end_turn"))
    assert isinstance(final[0], TextDeltaEvent) and "truncated" in final[0].text
    assert MAX_ASSISTANT_TEXT_CHARS == 100_000


def test_usage_event_cost_is_remembered() -> None:
    p = AcpProjector()
    p.feed(AcpEvent(kind="usage", usage={"used": 1, "size": 2}, cost_usd=0.5))
    p.feed(AcpEvent(kind="usage", usage={"used": 2, "size": 2}))
    assert p.last_cost_usd == 0.5


def test_cost_delta_rules() -> None:
    assert cost_delta(None, None, session_id="s") is None
    assert cost_delta(0.4, None, session_id="s") == 0.4
    prev = {"acp_session_id": "s", "amount": 0.4}
    assert abs(cost_delta(0.65, prev, session_id="s") - 0.25) < 1e-9
    # A different ACP session starts its own running total.
    assert cost_delta(0.1, prev, session_id="t") == 0.1
    # Total restarted within the session (a resume that did not carry it over).
    assert cost_delta(0.2, prev, session_id="s") == 0.2
    # Records written before the session key: ``record:pid:acp_session_id``.
    legacy = {"process": "rec-1:4242:s", "amount": 0.4}
    assert abs(cost_delta(0.5, legacy, session_id="s") - 0.1) < 1e-9
    assert COST_READING_KEY == "acp_cost_reading"


def test_resumed_session_cost_counts_only_the_increase() -> None:
    """2026-10-19: agent:main:acp:108fc5c7 turn 1 read 1.0338; the resumed
    turn 2 (new process, same ACP session) read 1.1732 and was booked whole."""
    turn1 = {"acp_session_id": "sess-a", "amount": 1.0338}
    delta = cost_delta(1.1732, turn1, session_id="sess-a")
    assert delta is not None and abs(delta - 0.1394) < 1e-9
    # cfb95897: the resumed reading restarted below the baseline.
    assert (
        cost_delta(0.299, {"acp_session_id": "sess-b", "amount": 0.3995}, session_id="sess-b")
        == 0.299
    )


def test_usage_update_fields_adds_tokens_and_cost_only_when_known() -> None:
    row = SimpleNamespace(
        input_tokens=5,
        output_tokens=6,
        cache_read=1,
        cache_write=0,
        total_tokens=12,
        total_cost_usd=1.0,
        billed_cost_usd=1.0,
        estimated_cost_usd=1.0,
    )
    usage = AcpUsage(input_tokens=10, output_tokens=20, cached_read_tokens=3, context_used=99)
    no_cost = usage_update_fields(row, usage, turn_cost=None)
    assert no_cost["input_tokens"] == 15 and no_cost["output_tokens"] == 26
    assert no_cost["cache_read"] == 4 and no_cost["total_tokens"] == 12 + 33
    assert no_cost["context_tokens"] == 99
    assert "total_cost_usd" not in no_cost and "cost_source" not in no_cost
    with_cost = usage_update_fields(row, usage, turn_cost=0.5)
    assert with_cost["total_cost_usd"] == 1.5
    assert with_cost["cost_source"] == HARNESS_COST_SOURCE


def test_stats_line_format() -> None:
    record = last_turn_record(
        AcpTurnResult(stop_reason="end_turn", usage=AcpUsage(input_tokens=1000, output_tokens=500)),
        duration_ms=312_000,
        turn_cost=0.0123,
    )
    assert format_stats(record) == "runtime 5m12s • tokens 1500 (in 1000 / out 500) • cost $0.0123"
    no_cost = last_turn_record(
        AcpTurnResult(stop_reason="end_turn"), duration_ms=4_000, turn_cost=None
    )
    assert format_stats(no_cost) == "runtime 4s"
    row = {"origin": {"kind": "acp", "last_turn": record}}
    assert stats_for_session(row) is not None
    assert stats_for_session({"origin": {"kind": "subagent", "last_turn": record}}) is None


def test_error_event_carries_code() -> None:
    event = error_event(AcpTurnTimeoutError("ACP turn timed out after 5s"))
    assert event.code == "ACP_TURN_TIMEOUT"
    assert "timed out" in event.message


def test_projected_records_map_one_to_one_onto_engine_events() -> None:
    p = AcpProjector(stream_delivery="live")
    projected = _feed(
        p,
        AcpEvent(kind="text_delta", text="hi"),
        AcpEvent(kind="tool_call", tool_call_id="t", tool_kind="read", raw_input={"p": 1}),
        AcpEvent(kind="tool_call_update", tool_call_id="t", status="completed"),
    )
    projected += p.final_events(AcpTurnResult(stop_reason="end_turn"), turn_cost=0.1)
    projected.append(error_event(AcpTurnTimeoutError("late")))
    engine = [to_engine_event(e) for e in projected]
    assert [type(e) for e in engine] == [
        engine_types.TextDeltaEvent,
        engine_types.ToolUseStartEvent,
        engine_types.ToolResultEvent,
        engine_types.DoneEvent,
        engine_types.ErrorEvent,
    ]
    assert [e.kind for e in engine] == [e.kind for e in projected]
    assert engine[2].arguments == {"p": 1, "title": "Reading a file", "acp_kind": "read"}
    assert engine[3].cost_source == "harness" and engine[4].code == "ACP_TURN_TIMEOUT"


def test_distinct_harness_messages_are_separated() -> None:
    from agentos.acp.projector import AcpProjector
    from agentos.acp.types import AcpEvent

    p = AcpProjector(stream_delivery="final_only")
    p.feed(AcpEvent(kind="text_delta", text="I'll create the file now.", raw={"message_id": "m1"}))
    p.feed(AcpEvent(kind="text_delta", text="Done.", raw={"message_id": "m2"}))
    assert "".join(p._text_parts) == "I'll create the file now.\n\nDone."
