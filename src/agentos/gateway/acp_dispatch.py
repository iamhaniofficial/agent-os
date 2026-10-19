"""Task-runtime dispatch for ACP sessions.

``gateway.boot.dispatch_task_runtime_turn`` routes a ``TaskRun`` here when
the session row's ``origin.kind == "acp"``. :func:`acp_turn_stream` is an
async iterator of ordinary engine events, so the gateway streams it through
the exact path a model turn uses (``_emit_task_runtime_stream_events``):
the desktop, web UI and channels see a running session, an ``error`` event
fails the task, and the subagent terminal listener announces completion to
the parent with a ``Stats:`` line.

Every failed or refused turn also leaves a ``system`` row ``Error: ...`` in
the child transcript (the shape the engine writes for a failed model turn):
the UIs resync history right after the ``error`` event, so an error that only
lived in the event stream disappeared from the screen.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import time
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from typing import Any

import structlog

from agentos.acp import projector as _projected
from agentos.acp.manager import (
    AcpSessionManager,
    agent_display_name,
    get_acp_manager,
    origin_is_closed,
    spec_from_origin,
)
from agentos.acp.projector import (
    COST_READING_KEY,
    HARNESS_COST_SOURCE,
    AcpProjector,
    cost_delta,
    error_event,
    last_turn_record,
    usage_update_fields,
)
from agentos.acp.projector import ErrorEvent as ProjectedError
from agentos.acp.types import AcpError, AcpSessionSpec, AcpTurnError, AcpTurnResult
from agentos.engine import types as engine_types

log = structlog.get_logger("agentos.gateway.acp_dispatch")

_ENGINE_EVENTS: dict[str, type] = {
    "text_delta": engine_types.TextDeltaEvent,
    "tool_use_start": engine_types.ToolUseStartEvent,
    "tool_result": engine_types.ToolResultEvent,
    "done": engine_types.DoneEvent,
    "error": engine_types.ErrorEvent,
}


def to_engine_event(event: _projected.EngineEvent) -> Any:
    """Turn a ``agentos.acp.projector`` record into the real engine event."""
    return _ENGINE_EVENTS[event.kind](**dataclasses.asdict(event))


_END = object()

#: Default ``[acp] stream_delivery`` (child sessions stream text as it arrives).
DEFAULT_STREAM_DELIVERY = "live"
#: Cap on the transcript ``Error:`` row (stderr tails can be long).
_ERROR_ROW_MAX_CHARS = 4000


def session_origin(session: Any) -> dict[str, Any]:
    origin = (
        session.get("origin") if isinstance(session, Mapping) else getattr(session, "origin", None)
    )
    return dict(origin) if isinstance(origin, Mapping) else {}


def is_acp_session(session: Any) -> bool:
    """The one discriminator: ``origin.kind == "acp"`` on the session row."""
    if session is None:
        return False
    return session_origin(session).get("kind") == "acp"


async def _append(session_manager: Any, session_key: str, **kwargs: Any) -> Any:
    append = getattr(session_manager, "append_message", None)
    if not callable(append):
        return None
    try:
        return await append(session_key, **kwargs)
    except Exception:  # noqa: BLE001 - transcript writes must not fail the turn
        log.warning("acp.dispatch.append_failed", session_key=session_key, exc_info=True)
        return None


async def _persist_error_row(session_manager: Any, session_key: str, message: str) -> None:
    """``system`` transcript row the history renderer shows for a failed turn."""
    text = (message or "ACP turn failed").strip()
    if len(text) > _ERROR_ROW_MAX_CHARS:
        text = text[:_ERROR_ROW_MAX_CHARS] + " [...]"
    await _append(session_manager, session_key, role="system", content=f"Error: {text}")


async def _refuse(
    session_manager: Any,
    session_key: str,
    error: ProjectedError,
) -> ProjectedError:
    """Log and persist a turn refused before the harness ran; return its event.

    The refusal becomes the session's ``last_turn`` too, so a completion
    announced for it does not carry the previous turn's ``Stats:``.
    """
    log.info("acp.dispatch.turn_refused", session_key=session_key, code=error.code)
    await _persist_turn(
        session_manager=session_manager,
        session_key=session_key,
        projector=AcpProjector(),
        result=None,
        duration_ms=0,
        process_key="",
        error=AcpError(error.message, code=error.code),
    )
    return error


def _closed_message(session_key: str, origin: Mapping[str, Any]) -> str:
    name = agent_display_name(str(origin.get("agent_id") or "")) or "ACP"
    message = (
        f"This {name} session ({session_key}) was closed and takes no more turns; spawn a new one"
    )
    acp_session_id = origin.get("acp_session_id")
    if acp_session_id:
        message += f" (resume_session_id={acp_session_id} continues the harness conversation)"
    return message + "."


async def _persist_turn(
    *,
    session_manager: Any,
    session_key: str,
    projector: AcpProjector,
    result: AcpTurnResult | None,
    duration_ms: int,
    process_key: str,
    error: AcpError | None = None,
    error_row: bool = True,
) -> float | None:
    """Write transcript row, usage totals and ``origin.last_turn``.

    With ``error`` (and ``error_row``) an ``Error:`` system row follows the
    partial answer, so the failure stays visible after a history resync.
    Returns this turn's harness cost increment (``None`` when unknown).
    """
    if session_manager is None:
        return None
    try:
        row = await session_manager.get_session(session_key)
    except Exception:  # noqa: BLE001
        row = None
    if row is None:
        if error is not None and error_row:
            await _persist_error_row(session_manager, session_key, str(error))
        return None
    origin = session_origin(row)
    turn_cost: float | None = None
    if result is not None:
        reading = result.usage.cost_usd if result.usage.cost_usd is not None else None
        if reading is None:
            reading = projector.last_cost_usd
        turn_cost = cost_delta(reading, origin.get(COST_READING_KEY), process_key=process_key)
        if reading is not None:
            origin[COST_READING_KEY] = {"process": process_key, "amount": float(reading)}
    origin["last_turn"] = last_turn_record(
        result,
        duration_ms=duration_ms,
        turn_cost=turn_cost,
        error=str(error) if error is not None else None,
        error_code=getattr(error, "code", None) if error is not None else None,
        tool_calls=projector.tool_calls if result is None else None,
    )
    fields: dict[str, Any] = {"origin": origin}
    if result is not None:
        fields.update(usage_update_fields(row, result.usage, turn_cost=turn_cost))
    try:
        await session_manager.update(session_key, **fields)
    except Exception:  # noqa: BLE001
        log.warning("acp.dispatch.usage_update_failed", session_key=session_key, exc_info=True)

    text = projector.text
    segments = projector.transcript_segments()
    if text or segments:
        usage = result.usage if result is not None else None
        turn_usage = None
        if usage is not None:
            turn_usage = {
                "input_tokens": int(usage.input_tokens or 0),
                "output_tokens": int(usage.output_tokens or 0),
                "reasoning_tokens": int(usage.thought_tokens or 0),
                "cached_tokens": int(usage.cached_read_tokens or 0),
                "cache_write_tokens": int(usage.cached_write_tokens or 0),
                "cost_usd": float(turn_cost or 0.0),
                "cost_source": HARNESS_COST_SOURCE if turn_cost is not None else "none",
                "model": usage.model or f"acp:{origin.get('agent_id') or ''}",
            }
        await _append(
            session_manager,
            session_key,
            role="assistant",
            content=text,
            tool_calls=segments or None,
            turn_usage=turn_usage,
        )
    if error is not None and error_row:
        await _persist_error_row(session_manager, session_key, str(error))
    return turn_cost


def _admit_turn(
    session: Any,
    origin: Mapping[str, Any],
    *,
    session_key: str,
    acp_config: Any,
    manager: AcpSessionManager | None,
) -> AcpSessionSpec | ProjectedError:
    """The harness spec for this turn, or why the turn is refused."""
    if acp_config is not None and getattr(acp_config, "enabled", True) is False:
        return ProjectedError(
            message="ACP is disabled ([acp] enabled = false)", code="ACP_DISABLED"
        )
    if manager is None or not manager.available:
        reason = manager.unavailable_reason if manager is not None else "not configured"
        return ProjectedError(message=f"ACP unavailable: {reason}", code="ACP_UNAVAILABLE")
    try:
        spec = spec_from_origin(origin, acp_config or manager.config, session_key=session_key)
    except AcpError as exc:
        return error_event(exc)
    # Only an operator close (acp.close / subagents kill) refuses a turn; a
    # finished oneshot is idle and resumes its harness session.
    status = (
        session.get("status") if isinstance(session, Mapping) else getattr(session, "status", None)
    )
    if origin_is_closed(origin, session_status=status):
        return ProjectedError(
            message=_closed_message(session_key, origin), code="ACP_SESSION_CLOSED"
        )
    return spec


async def acp_turn_stream(
    run: Any,
    *,
    session: Any,
    config: Any,
    session_manager: Any,
    manager: AcpSessionManager | None = None,
) -> AsyncIterator[Any]:
    """Run one harness turn for ``run`` and yield engine events."""
    stream = _projected_turn_stream(
        run, session=session, config=config, session_manager=session_manager, manager=manager
    )
    try:
        async for event in stream:
            yield to_engine_event(event)
    finally:
        await stream.aclose()


async def _projected_turn_stream(
    run: Any,
    *,
    session: Any,
    config: Any,
    session_manager: Any,
    manager: AcpSessionManager | None = None,
) -> AsyncGenerator[_projected.EngineEvent, None]:
    session_key = run.session_key
    acp_config = getattr(config, "acp", None)
    manager = manager or get_acp_manager()
    origin = session_origin(session)

    # A follow-up that did not come through sessions.send (which persists the
    # user row itself) still belongs in the child transcript -- also when the
    # turn is refused below, so the error row answers a visible message.
    if not getattr(run, "persisted_user_message_id", None):
        await _append(session_manager, session_key, role="user", content=run.message)

    admitted = _admit_turn(
        session, origin, session_key=session_key, acp_config=acp_config, manager=manager
    )
    if isinstance(admitted, ProjectedError):
        yield await _refuse(session_manager, session_key, admitted)
        return
    spec = admitted
    assert manager is not None  # _admit_turn refuses without a usable manager

    delivery = (
        getattr(acp_config, "stream_delivery", DEFAULT_STREAM_DELIVERY)
        if acp_config
        else DEFAULT_STREAM_DELIVERY
    )
    projector = AcpProjector(stream_delivery="final_only" if delivery == "final_only" else "live")
    queue: asyncio.Queue[Any] = asyncio.Queue()

    def _on_event(event: Any) -> None:
        for projected in projector.feed(event):
            queue.put_nowait(projected)

    started = time.monotonic()
    task = asyncio.create_task(
        manager.run_turn(
            session_key,
            run.message,
            spec=spec,
            attachments=list(getattr(run, "attachments", None) or []) or None,
            on_event=_on_event,
        )
    )
    task.add_done_callback(lambda _t: queue.put_nowait(_END))
    try:
        while True:
            item = await queue.get()
            if item is _END:
                break
            yield item
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    duration_ms = int((time.monotonic() - started) * 1000)
    process_key = manager.process_key(session_key)
    try:
        result = task.result()
    except AcpError as exc:
        await _persist_turn(
            session_manager=session_manager,
            session_key=session_key,
            projector=projector,
            result=None,
            duration_ms=duration_ms,
            process_key=process_key,
            error=exc,
        )
        log.info("acp.dispatch.turn_failed", session_key=session_key, code=exc.code)
        yield error_event(exc)
        return
    except Exception as exc:  # noqa: BLE001 - surface as a turn error, not a crash
        wrapped = AcpTurnError(str(exc) or type(exc).__name__)
        await _persist_turn(
            session_manager=session_manager,
            session_key=session_key,
            projector=projector,
            result=None,
            duration_ms=duration_ms,
            process_key=process_key,
            error=wrapped,
        )
        log.info("acp.dispatch.turn_failed", session_key=session_key, code=wrapped.code)
        yield error_event(wrapped)
        return

    if result.stop_reason == "cancelled":
        cancelled = AcpTurnError(
            "Harness turn was cancelled by an operator or the parent agent; do not retry it",
            code="ACP_TURN_CANCELLED",
        )
        await _persist_turn(
            session_manager=session_manager,
            session_key=session_key,
            projector=projector,
            result=result,
            duration_ms=duration_ms,
            process_key=process_key,
            error=cancelled,
            # A stop the user (or parent) asked for is not a failure to report.
            error_row=False,
        )
        yield error_event(cancelled)
        return

    turn_cost = await _persist_turn(
        session_manager=session_manager,
        session_key=session_key,
        projector=projector,
        result=result,
        duration_ms=duration_ms,
        process_key=process_key,
    )
    for event in projector.final_events(result, model=f"acp:{spec.agent_id}", turn_cost=turn_cost):
        yield event
