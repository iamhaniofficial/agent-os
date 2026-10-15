"""Schema 16: the per-turn ``timing`` breakdown in the decision log."""

from __future__ import annotations

import json
from typing import Any

from agentos.observability.decision_log import (
    SCHEMA_VERSION,
    DecisionEntry,
    load_entries,
    write_decision_entry,
)
from agentos.observability.replay import format_transcript

_TIMING: dict[str, Any] = {
    "total_ms": 4210,
    "pre_provider_ms": 640,
    "first_text_ms": 1900,
    "provider_ms": 2900,
    "tool_ms": 450,
    "approval_wait_ms": 0,
    "finalize_ms": 120,
    "iterations": 2,
    "tool_batches": 1,
    "tool_calls": 2,
    "stages": {"input": 3, "stream_consumer": 3400, "turn_finalizer": 120},
    "provider_calls": [
        {
            "request_started_ms": 640,
            "first_event_ms": 800,
            "first_text_ms": None,
            "stream_done_ms": 1500,
            "input_tokens": 9000,
            "output_tokens": 80,
        }
    ],
    "tool_batch_detail": [
        {"started_ms": 2140, "duration_ms": 450, "calls": 2, "approval_wait_ms": 0}
    ],
}


def _entry(**overrides: Any) -> DecisionEntry:
    base: dict[str, Any] = {
        "turn_id": "turn-1",
        "session_key": "agent:main:test",
        "prompt_hash": "a" * 16,
        "system_prompt_hash": "b" * 16,
        "tool_list_hash": "c" * 16,
        "tool_choice": "auto",
        "tokens_input": 1,
        "tokens_output": 2,
        "model": "fake-model",
        "provider": "fake",
        "latency_ms": 4210,
        "ts": "2026-10-15T00:00:00Z",
    }
    base.update(overrides)
    return DecisionEntry(**base)


def test_schema_version_is_16_and_timing_defaults_empty() -> None:
    assert SCHEMA_VERSION == 16
    assert _entry().timing == {}


def test_timing_round_trips_through_the_log(tmp_path) -> None:
    write_decision_entry(_entry(timing=_TIMING), log_dir=tmp_path)

    path = next(tmp_path.glob("decisions-*.jsonl"))
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == 16
    assert raw["timing"] == _TIMING
    assert load_entries(path)[0].timing == _TIMING


def test_rows_without_or_with_malformed_timing_load_as_empty(tmp_path) -> None:
    path = tmp_path / "decisions-20261015.jsonl"
    old = _entry().__dict__ | {"schema_version": 15}
    old.pop("timing")
    bad = _entry(turn_id="turn-2").__dict__ | {"timing": "not-a-dict"}
    path.write_text(
        "\n".join(json.dumps(row, default=lambda _o: {}) for row in (old, bad)) + "\n",
        encoding="utf-8",
    )

    entries = load_entries(path)

    assert [e.turn_id for e in entries] == ["turn-1", "turn-2"]
    assert [e.timing for e in entries] == [{}, {}]


def test_replay_prints_the_timing_breakdown() -> None:
    text = format_transcript(_entry(timing=_TIMING))

    assert "Timing: total=4210ms pre_provider=640ms first_text=1900ms" in text
    assert "approvals=0ms finalize=120ms iterations=2" in text
    assert "Stages: input=3ms stream_consumer=3400ms turn_finalizer=120ms" in text
    assert "call 1: start=+640ms first_event=800ms first_text=- done=1500ms" in text
    assert "tokens in=9000 out=80" in text
    assert "tool batch 1: start=+2140ms duration=450ms calls=2 approvals=0ms" in text


def test_replay_of_a_pre_16_row_has_no_timing_lines() -> None:
    assert "Timing:" not in format_transcript(_entry())
