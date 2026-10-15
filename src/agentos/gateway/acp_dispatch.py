"""Task-runtime dispatch for ACP sessions.

``gateway.boot.dispatch_task_runtime_turn`` routes a ``TaskRun`` here when
the session row's ``origin.kind == "acp"``. :func:`acp_turn_stream` is an
async iterator of ordinary engine events, so the gateway streams it through
the exact path a model turn uses (``_emit_task_runtime_stream_events``):
the desktop, web UI and channels see a running session, an ``error`` event
fails the task, and the subagent terminal listener announces completion to
the parent with a ``Stats:`` line.
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
from agentos.acp.manager import AcpSessionManager, get_acp_manager, spec_from_origin
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
from agentos.acp.types import AcpError, AcpTurnError, AcpTurnResult
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


async def _persist_turn(
    *,
    session_manager: Any,
    session_key: str,
    projector: AcpProjector,
    result: AcpTurnResult | None,
    duration_ms: int,
    process_key: str,
    error: AcpError | None = None,
) -> float | None:
    """Write transcript row, usage totals and ``origin.last_turn``.

    Returns this turn's harness cost increment (``None`` when unknown).
    """
    if session_manager is None:
        return None
    try:
        row = await session_manager.get_session(session_key)
    except Exception:  # noqa: BLE001
        row = None
    if row is None:
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
    return turn_cost


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
    if acp_config is not None and getattr(acp_config, "enabled", True) is False:
        yield ProjectedError(message="ACP is disabled ([acp] enabled = false)", code="ACP_DISABLED")
        return
    if manager is None or not manager.available:
        reason = manager.unavailable_reason if manager is not None else "not configured"
        yield ProjectedError(message=f"ACP unavailable: {reason}", code="ACP_UNAVAILABLE")
        return

    origin = session_origin(session)
    try:
        spec = spec_from_origin(origin, acp_config or manager.config, session_key=session_key)
    except AcpError as exc:
        yield error_event(exc)
        return
    if origin.get("closed"):
        yield ProjectedError(
            message=f"ACP session {session_key} was closed; spawn a new one",
            code="ACP_SESSION_CLOSED",
        )
        return

    # A follow-up that did not come through sessions.send (which persists the
    # user row itself) still belongs in the child transcript.
    if not getattr(run, "persisted_user_message_id", None):
        await _append(session_manager, session_key, role="user", content=run.message)

    delivery = getattr(acp_config, "stream_delivery", "final_only") if acp_config else "final_only"
    projector = AcpProjector(stream_delivery="live" if delivery == "live" else "final_only")
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
        yield error_event(wrapped)
        return

    if result.stop_reason == "cancelled":
        cancelled = AcpTurnError("Harness turn was cancelled", code="ACP_TURN_CANCELLED")
        await _persist_turn(
            session_manager=session_manager,
            session_key=session_key,
            projector=projector,
            result=result,
            duration_ms=duration_ms,
            process_key=process_key,
            error=cancelled,
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
