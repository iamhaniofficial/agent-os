"""Project a normalized ACP event stream onto AgentOS engine events.

The harness speaks ACP ``session/update`` notifications; the runtime turns
them into :class:`~agentos.acp.types.AcpEvent`. This module is the last hop:
it maps those events onto the engine events the gateway already knows how to
stream (``TextDeltaEvent``, ``ToolUseStartEvent``, ``ToolResultEvent``,
``DoneEvent``, ``ErrorEvent``), builds the transcript row the normal turn
path would have written, and rolls harness usage into the session row.

``agentos.acp`` stays a leaf package (the architecture import contract keeps
it out of the engine/gateway cycle), so the projector emits ``Projected*``
records whose fields mirror the engine event dataclasses one to one;
``agentos.gateway.acp_dispatch.to_engine_event`` turns them into the real
engine events.

Shape borrowed from OpenClaw's ``acp-projector.ts`` (MIT, see
``THIRD_PARTY_NOTICES.md``): hidden-by-default thought/plan/status tags,
``final_only`` vs ``live`` delivery, an output cap with a tail marker.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import structlog

from agentos.acp.labels import harness_tool_name, tool_label
from agentos.acp.types import AcpEvent, AcpTurnResult, AcpUsage

log = structlog.get_logger("agentos.acp.projector")

#: Cap on the assistant text kept for one harness turn (UI, transcript, announce).
#: A long harness report is the work itself: at 24,000 a 25-34k-character
#: report reached the parent with its ending dropped (``full_chars`` 24,064 =
#: the cap plus the truncation marker) and the parent went to read the
#: harness's own files instead. The parent reads past the completion's
#: 12,000 characters with ``sessions_history(answer_offset=...)``.
MAX_ASSISTANT_TEXT_CHARS = 100_000
#: Cap on one tool result kept in the transcript.
MAX_TOOL_RESULT_CHARS = 4_000

#: ``cost_source`` written on session rows whose cost came from the harness.
HARNESS_COST_SOURCE = "harness"

StreamDelivery = Literal["final_only", "live"]


# Field-for-field mirrors of ``agentos.engine.types`` events; ``kind`` matches
# the engine event's ``kind`` discriminator.


@dataclass(slots=True)
class TextDeltaEvent:
    kind: ClassVar[str] = "text_delta"
    text: str = ""


@dataclass(slots=True)
class ToolUseStartEvent:
    kind: ClassVar[str] = "tool_use_start"
    tool_use_id: str = ""
    tool_name: str = ""


@dataclass(slots=True)
class ToolResultEvent:
    kind: ClassVar[str] = "tool_result"
    tool_use_id: str = ""
    tool_name: str = ""
    result: str = ""
    is_error: bool = False
    arguments: dict[str, Any] | None = None


@dataclass(slots=True)
class DoneEvent:
    kind: ClassVar[str] = "done"
    text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    iterations: int = 0
    cost_usd: float = 0.0
    cost_source: str = "none"
    model: str = ""
    reasoning_content: str | None = None


@dataclass(slots=True)
class ErrorEvent:
    kind: ClassVar[str] = "error"
    message: str = ""
    code: str = ""


EngineEvent = TextDeltaEvent | ToolUseStartEvent | ToolResultEvent | DoneEvent | ErrorEvent


def _truncation_marker(dropped: int) -> str:
    return f"\n\n[... harness output truncated: {dropped} more characters not shown]"


def _tool_name(tool_kind: str | None) -> str:
    """Fallback tool name when a step has no label at all."""
    kind = (tool_kind or "other").strip() or "other"
    return f"acp:{kind}"


def _tool_display_name(state: _ToolState) -> str:
    """The name a tool row carries: the human label ("Run git status").

    The UIs title a tool card with its ``name``; ``input.acp_kind`` keeps the
    ACP tool kind for anything that wants it.
    """
    return tool_label(
        state.title, state.tool_kind, state.raw_input, tool_name=state.tool_name
    ) or _tool_name(state.tool_kind)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _cap(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + _truncation_marker(len(text) - limit)


@dataclass(slots=True)
class _ToolState:
    tool_call_id: str
    tool_kind: str | None = None
    title: str | None = None
    raw_input: dict[str, Any] | None = None
    tool_name: str | None = None
    finished: bool = False


@dataclass(slots=True)
class AcpProjector:
    """Stateful per-turn projector. Feed events, then call :meth:`finish`."""

    stream_delivery: StreamDelivery = "final_only"
    max_text_chars: int = MAX_ASSISTANT_TEXT_CHARS
    show_thoughts: bool = False
    _text_parts: list[str] = field(default_factory=list)
    _last_message_id: str | None = None
    _text_len: int = 0
    _dropped_chars: int = 0
    _current_segment_text: list[str] = field(default_factory=list)
    _segments: list[dict[str, Any]] = field(default_factory=list)
    _tools: dict[str, _ToolState] = field(default_factory=dict)
    _tool_order: list[str] = field(default_factory=list)
    _last_cost_usd: float | None = None
    _thoughts: list[str] = field(default_factory=list)
    _tool_calls: int = 0
    #: A tool call came between the previous text run and the next one.
    _tool_since_text: bool = False

    # ------------------------------------------------------------------ feed

    def feed(self, event: AcpEvent) -> list[EngineEvent]:
        """Map one harness event to zero or more engine events."""
        kind = event.kind
        if kind == "text_delta":
            return self._on_text(event.text or "", message_id=_message_id(event))
        if kind == "tool_call":
            return self._on_tool_call(event)
        if kind == "tool_call_update":
            return self._on_tool_update(event)
        if kind == "usage":
            if event.cost_usd is not None:
                self._last_cost_usd = float(event.cost_usd)
            return []
        if kind == "thought_delta":
            if self.show_thoughts and event.text:
                self._thoughts.append(event.text)
            return []
        # plan / mode / status: hidden by default; ``done`` / ``error`` are
        # synthesized by the runtime and handled by :meth:`finish` /
        # :meth:`error_event` once ``result`` resolves.
        return []

    def _on_text(self, text: str, *, message_id: str | None = None) -> list[EngineEvent]:
        if not text:
            return []
        if self._text_parts:
            tail = "".join(self._text_parts[-2:])
            new_message = bool(
                message_id and self._last_message_id and message_id != self._last_message_id
            )
            if (new_message or self._tool_since_text) and not tail.endswith("\n\n"):
                # A new harness message, or narration resumed after a tool
                # call (Claude Code sends no message ids): a paragraph break
                # keeps it from gluing onto the previous sentence.
                text = ("\n" if tail.endswith("\n") else "\n\n") + text
        self._tool_since_text = False
        if message_id:
            self._last_message_id = message_id
        room = self.max_text_chars - self._text_len
        if room <= 0:
            self._dropped_chars += len(text)
            return []
        kept = text if len(text) <= room else text[:room]
        self._dropped_chars += len(text) - len(kept)
        self._text_parts.append(kept)
        self._current_segment_text.append(kept)
        self._text_len += len(kept)
        if self.stream_delivery == "live":
            return [TextDeltaEvent(text=kept)]
        return []

    def _flush_text_segment(self) -> None:
        if self._current_segment_text:
            self._segments.append({"type": "text", "text": "".join(self._current_segment_text)})
            self._current_segment_text = []

    def _start_tool(self, event: AcpEvent) -> tuple[_ToolState, list[EngineEvent]]:
        tool_id = event.tool_call_id or f"acp-tool-{len(self._tool_order) + 1}"
        existing = self._tools.get(tool_id)
        if existing is not None:
            return existing, []
        state = _ToolState(
            tool_call_id=tool_id,
            tool_kind=event.tool_kind,
            title=event.title,
            raw_input=dict(event.raw_input) if isinstance(event.raw_input, dict) else None,
            tool_name=_harness_tool_name(event),
        )
        self._tools[tool_id] = state
        self._tool_order.append(tool_id)
        self._tool_calls += 1
        self._tool_since_text = True
        self._flush_text_segment()
        name = _tool_display_name(state)
        self._segments.append(
            {
                "type": "tool_use",
                "tool_use_id": tool_id,
                "name": name,
                "input": self._tool_arguments(state),
            }
        )
        return state, [ToolUseStartEvent(tool_use_id=tool_id, tool_name=name)]

    @staticmethod
    def _tool_arguments(state: _ToolState) -> dict[str, Any]:
        args: dict[str, Any] = dict(state.raw_input or {})
        args.setdefault("title", _tool_display_name(state))
        args.setdefault("acp_kind", (state.tool_kind or "other").strip() or "other")
        return args

    def _merge(self, state: _ToolState, event: AcpEvent) -> None:
        if event.title:
            state.title = event.title
        if event.tool_kind:
            state.tool_kind = event.tool_kind
        if isinstance(event.raw_input, dict) and event.raw_input:
            state.raw_input = dict(event.raw_input)
        state.tool_name = state.tool_name or _harness_tool_name(event)

    def _finish_tool(self, state: _ToolState, event: AcpEvent) -> list[EngineEvent]:
        if state.finished:
            return []
        state.finished = True
        failed = event.status == "failed"
        result_text = _stringify(event.raw_output)
        if not result_text and event.error:
            result_text = event.error
        if not result_text:
            result_text = "failed" if failed else "completed"
        capped = _cap(result_text, MAX_TOOL_RESULT_CHARS)
        name = _tool_display_name(state)
        arguments = self._tool_arguments(state)
        for segment in reversed(self._segments):
            if (
                segment.get("type") == "tool_use"
                and segment.get("tool_use_id") == state.tool_call_id
            ):
                segment["name"] = name
                segment["input"] = arguments
                break
        self._segments.append(
            {
                "type": "tool_result",
                "tool_use_id": state.tool_call_id,
                "name": name,
                "result": capped,
                "is_error": failed,
            }
        )
        return [
            ToolResultEvent(
                tool_use_id=state.tool_call_id,
                tool_name=name,
                result=capped,
                is_error=failed,
                arguments=arguments,
            )
        ]

    def _on_tool_call(self, event: AcpEvent) -> list[EngineEvent]:
        state, out = self._start_tool(event)
        self._merge(state, event)
        if event.status in ("completed", "failed"):
            out.extend(self._finish_tool(state, event))
        return out

    def _on_tool_update(self, event: AcpEvent) -> list[EngineEvent]:
        out: list[EngineEvent] = []
        state = self._tools.get(event.tool_call_id or "")
        if state is None:
            state, out = self._start_tool(event)
        self._merge(state, event)
        if event.status in ("completed", "failed"):
            out.extend(self._finish_tool(state, event))
        return out

    # ---------------------------------------------------------------- finish

    @property
    def text(self) -> str:
        body = "".join(self._text_parts)
        if self._dropped_chars:
            body += _truncation_marker(self._dropped_chars)
        return body

    @property
    def truncated(self) -> bool:
        return self._dropped_chars > 0

    @property
    def tool_calls(self) -> int:
        return self._tool_calls

    @property
    def last_cost_usd(self) -> float | None:
        return self._last_cost_usd

    def transcript_segments(self) -> list[dict[str, Any]]:
        """Segments in the shape the normal turn path persists as ``tool_calls``.

        A tool the harness started but never reported as finished gets a
        synthetic result so every ``tool_use`` has its pair.
        """
        self._flush_text_segment()
        segments = [dict(s) for s in self._segments]
        for tool_id in self._tool_order:
            state = self._tools[tool_id]
            if state.finished:
                continue
            segments.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "name": _tool_display_name(state),
                    "result": "[no result reported by the harness]",
                    "is_error": False,
                }
            )
        if self._dropped_chars and segments and segments[-1].get("type") == "text":
            segments[-1]["text"] = segments[-1]["text"] + _truncation_marker(self._dropped_chars)
        elif self._dropped_chars:
            segments.append({"type": "text", "text": _truncation_marker(self._dropped_chars)})
        if all(s.get("type") == "text" for s in segments):
            # Plain text turn: the content column already carries it.
            return []
        return segments

    def final_events(
        self, result: AcpTurnResult, *, model: str = "", turn_cost: float | None = None
    ) -> list[EngineEvent]:
        """Events to emit once the turn has resolved: buffered text, then done."""
        out: list[EngineEvent] = []
        text = self.text
        if self.stream_delivery == "final_only":
            if text:
                out.append(TextDeltaEvent(text=text))
        elif self._dropped_chars:
            out.append(TextDeltaEvent(text=_truncation_marker(self._dropped_chars)))
        out.append(self.done_event(result, model=model, turn_cost=turn_cost))
        return out

    def done_event(
        self, result: AcpTurnResult, *, model: str = "", turn_cost: float | None = None
    ) -> DoneEvent:
        usage = result.usage or AcpUsage()
        return DoneEvent(
            text=self.text,
            input_tokens=int(usage.input_tokens or 0),
            output_tokens=int(usage.output_tokens or 0),
            reasoning_tokens=int(usage.thought_tokens or 0),
            cached_tokens=int(usage.cached_read_tokens or 0),
            cache_write_tokens=int(usage.cached_write_tokens or 0),
            iterations=max(1, self._tool_calls),
            cost_usd=float(turn_cost or 0.0),
            cost_source=HARNESS_COST_SOURCE if turn_cost is not None else "none",
            model=usage.model or model,
            reasoning_content="".join(self._thoughts) or None,
        )


def _harness_tool_name(event: AcpEvent) -> str | None:
    return harness_tool_name(event.raw if isinstance(event.raw, dict) else None) or None


def _message_id(event: AcpEvent) -> str | None:
    raw = event.raw if isinstance(event.raw, dict) else {}
    value = raw.get("message_id") or raw.get("messageId")
    return str(value) if value else None


def error_event(exc: BaseException) -> ErrorEvent:
    code = getattr(exc, "code", None) or "ACP_TURN_FAILED"
    message = str(exc) or code
    return ErrorEvent(message=message, code=str(code))


# --------------------------------------------------------------------------
# Usage / cost accounting
# --------------------------------------------------------------------------

#: ``origin`` key holding the last cumulative harness cost reading.
COST_READING_KEY = "acp_cost_reading"


def _baseline_session(previous: Mapping[str, Any]) -> str | None:
    session = previous.get("acp_session_id")
    if isinstance(session, str) and session:
        return session
    # Older records keyed the baseline by process: ``record:pid:acp_session_id``.
    process = previous.get("process")
    if isinstance(process, str) and process:
        tail = process.rsplit(":", 1)[-1]
        return tail or None
    return None


def cost_delta(
    reading: float | None,
    previous: Mapping[str, Any] | None,
    *,
    session_id: str | None,
) -> float | None:
    """Turn a cumulative harness cost reading into this turn's increment.

    ACP's ``usage_update.cost.amount`` is the *cumulative* cost of the ACP
    session (Claude's ``total_cost_usd``), and a resumed session may carry
    that total over into the new process. So the baseline is keyed by
    ``acp_session_id``, not by process: a reading at or above the baseline of
    the same session counts only the increase; a reading below it means the
    total restarted (a resume that did not carry it over), so the reading
    itself is the increment; a different session starts its own total.
    """
    if reading is None:
        return None
    reading = max(0.0, float(reading))
    if not isinstance(previous, Mapping) or not session_id:
        return reading
    if _baseline_session(previous) != session_id:
        return reading
    try:
        base = float(previous.get("amount") or 0.0)
    except (TypeError, ValueError):
        base = 0.0
    if reading < base:
        return reading
    return reading - base


def usage_update_fields(
    session: Any,
    usage: AcpUsage,
    *,
    turn_cost: float | None,
) -> dict[str, Any]:
    """Session-row fields after adding one harness turn's usage.

    Token counts from ``PromptResponse.usage`` are per turn (the adapters
    reset their tally when a turn starts), so they add up directly. Cost is
    only touched when the harness reported one.
    """

    def _get(name: str, default: Any = 0) -> Any:
        value = getattr(session, name, None)
        if value is None and isinstance(session, Mapping):
            value = session.get(name)
        return default if value is None else value

    inp = int(usage.input_tokens or 0)
    out = int(usage.output_tokens or 0)
    cread = int(usage.cached_read_tokens or 0)
    cwrite = int(usage.cached_write_tokens or 0)
    total = int(usage.total_tokens or 0) or (inp + out + cread + cwrite)
    fields: dict[str, Any] = {
        "input_tokens": int(_get("input_tokens")) + inp,
        "output_tokens": int(_get("output_tokens")) + out,
        "cache_read": int(_get("cache_read")) + cread,
        "cache_write": int(_get("cache_write")) + cwrite,
        "total_tokens": int(_get("total_tokens")) + total,
    }
    if usage.context_used is not None:
        fields["context_tokens"] = int(usage.context_used)
    if usage.model:
        fields["model"] = usage.model
    if turn_cost is not None:
        fields["total_cost_usd"] = float(_get("total_cost_usd", 0.0)) + turn_cost
        fields["billed_cost_usd"] = float(_get("billed_cost_usd", 0.0)) + turn_cost
        fields["estimated_cost_usd"] = float(_get("estimated_cost_usd", 0.0)) + turn_cost
        fields["cost_source"] = HARNESS_COST_SOURCE
    return fields


def last_turn_record(
    result: AcpTurnResult | None,
    *,
    duration_ms: int,
    turn_cost: float | None,
    error: str | None = None,
    error_code: str | None = None,
    tool_calls: int | None = None,
    stop_reason: str | None = None,
    stopped_by: str | None = None,
) -> dict[str, Any]:
    """The ``origin["last_turn"]`` entry the announce ``Stats:`` line reads.

    ``tool_calls`` overrides the result's count (a failed or cancelled turn
    has no result but the projector still saw its tool calls). Token counts
    are left out when nobody reported them (no result, or a cancelled turn
    whose response carried none) rather than recorded as zero.
    """
    usage = result.usage if result is not None else AcpUsage()
    if tool_calls is None:
        tool_calls = result.tool_calls if result is not None else 0
    if stop_reason is None:
        stop_reason = result.stop_reason if result is not None else "error"
    record: dict[str, Any] = {
        "duration_ms": int(duration_ms),
        "stop_reason": stop_reason,
        "tool_calls": int(tool_calls),
    }
    reported = result is not None and (
        result.stop_reason != "cancelled"
        or bool(usage.input_tokens or usage.output_tokens or usage.total_tokens)
    )
    if reported:
        record["input_tokens"] = int(usage.input_tokens or 0)
        record["output_tokens"] = int(usage.output_tokens or 0)
        record["total_tokens"] = int(usage.total_tokens or 0) or int(usage.input_tokens or 0) + int(
            usage.output_tokens or 0
        )
    if turn_cost is not None:
        record["cost_usd"] = round(float(turn_cost), 6)
    if stopped_by:
        # A stop is not a failure: no agent-facing error text, a summary the
        # UIs can show as it is ("Stopped by you after 3 tool calls (0:18).").
        record["stopped_by"] = stopped_by
        record["summary"] = stop_summary(
            stopped_by, record["tool_calls"], format_clock(record["duration_ms"])
        )
    elif error:
        record["error"] = error[:500]
    if error_code:
        record["error_code"] = error_code
    return record


def stop_summary(stopped_by: str, tool_calls: int, clock: str) -> str:
    """User-facing line for a stopped turn: "Stopped by you after 3 tool calls (0:18)."."""
    who = "you" if stopped_by == "user" else "the agent"
    return f"Stopped by {who} {tool_calls_phrase(tool_calls, clock)}."


def _format_duration(ms: int) -> str:
    seconds = max(0, int(round(ms / 1000)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def format_clock(ms: int | float) -> str:
    """``m:ss`` (``h:mm:ss`` past an hour) for a duration in milliseconds."""
    seconds = max(0, int(round(float(ms) / 1000)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def tool_calls_phrase(tool_calls: int, clock: str) -> str:
    """``after 3 tool calls (0:18)`` / ``after 1 tool call`` / ``before any tool call``."""
    if tool_calls <= 0:
        return f"before any tool call ({clock})"
    if tool_calls == 1:
        return f"after 1 tool call ({clock})"
    return f"after {tool_calls} tool calls ({clock})"


def format_stats(last_turn: Mapping[str, Any] | None) -> str | None:
    """``runtime 5m12s • tokens N (in X / out Y) • cost $Z`` (cost when known)."""
    if not isinstance(last_turn, Mapping):
        return None
    parts: list[str] = []
    duration = last_turn.get("duration_ms")
    if isinstance(duration, (int, float)):
        parts.append(f"runtime {_format_duration(int(duration))}")
    inp = int(last_turn.get("input_tokens") or 0)
    out = int(last_turn.get("output_tokens") or 0)
    total = int(last_turn.get("total_tokens") or 0) or inp + out
    if total or inp or out:
        parts.append(f"tokens {total} (in {inp} / out {out})")
    cost = last_turn.get("cost_usd")
    if isinstance(cost, (int, float)):
        parts.append(f"cost ${float(cost):.4f}")
    return " • ".join(parts) if parts else None


def stats_for_session(session_or_row: Any) -> str | None:
    """Stats line for an ACP child session row, ``None`` for anything else."""
    origin = (
        session_or_row.get("origin")
        if isinstance(session_or_row, Mapping)
        else getattr(session_or_row, "origin", None)
    )
    if not isinstance(origin, Mapping) or origin.get("kind") != "acp":
        return None
    return format_stats(origin.get("last_turn"))
