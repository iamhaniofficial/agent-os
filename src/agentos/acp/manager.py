"""``AcpSessionManager``: one per gateway, owns every live harness session.

Responsibilities (OpenClaw ``src/acp/control-plane/manager.core.ts`` and
friends, MIT, rewritten for asyncio; see ``THIRD_PARTY_NOTICES.md``):

- an actor per AgentOS session key (an ``asyncio.Lock`` plus an epoch that
  is bumped on close) so turns on one session never interleave;
- a handle cache so a ``persistent`` session keeps its harness process
  across turns, and resumes from the session row ``origin`` when it does not;
  a ``oneshot`` session closes its process after every turn but stays
  resumable (``origin.idle``): a follow-up spawns the harness again with the
  recorded ``acp_session_id``. Only an operator close (``acp.close``,
  ``subagents kill``) refuses further turns;
- admission: at most ``AcpConfig.max_concurrent`` live harness processes;
  when the cap is reached the least recently used *idle* session is closed
  (it stays resumable) and otherwise the turn waits for a slot;
- the turn wall clock (``turn_timeout_s``; silence alone is never a failure),
  cancel (from the task runtime or an operator), close, mode / config option
  changes and an orderly ``shutdown`` at gateway stop.

The runtime is injected (:class:`agentos.acp.types.AcpRuntime`); this module
never imports the concrete implementation.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import structlog

from agentos.acp.labels import (
    KIND_LABELS,
    MAX_LABEL_CHARS,
    TOOL_KINDS,
    harness_tool_name,
    tool_label,
)
from agentos.acp.registry import get_harness, normalize_agent_id
from agentos.acp.types import (
    AcpError,
    AcpEvent,
    AcpRuntime,
    AcpSessionHandle,
    AcpSessionSpec,
    AcpTurn,
    AcpTurnError,
    AcpTurnResult,
    AcpTurnTimeoutError,
    SessionMode,
)

log = structlog.get_logger("agentos.acp.manager")

EventCallback = Callable[[AcpEvent], Awaitable[None] | None]

#: Logical option name -> ACP config option ids / categories, first match wins.
OPTION_ALIASES: dict[str, tuple[str, ...]] = {
    "model": ("model",),
    "thinking": ("thinking", "effort", "reasoning_effort", "thought_level"),
    "permissions": ("approval_policy", "permission_profile", "permissions", "permission_mode"),
}

_CANCEL_GRACE_S = 5.0
_CLOSE_GRACE_S = 10.0
#: Idle (process-less) session states kept in memory for ``status`` / activity.
_MAX_IDLE_STATES = 256

#: Longest ``activity`` label (one line, ``$HOME`` shortened to ``~``).
ACTIVITY_MAX_CHARS = MAX_LABEL_CHARS
#: ``activity_kind`` while the harness streams its answer.
TEXT_ACTIVITY_KIND = "text"
TEXT_ACTIVITY_LABEL = "Writing the answer"


class AcpUnavailableError(AcpError):
    code = "ACP_UNAVAILABLE"


class AcpSessionClosedError(AcpError):
    code = "ACP_SESSION_CLOSED"


class AcpOptionUnsupportedError(AcpError):
    code = "ACP_OPTION_UNSUPPORTED"


@dataclass(slots=True)
class _SessionState:
    session_key: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    epoch: int = 0
    handle: AcpSessionHandle | None = None
    spec: AcpSessionSpec | None = None
    active_turn: AcpTurn | None = None
    turn_started_at: float | None = None
    turns: int = 0
    last_used: float = field(default_factory=time.monotonic)
    last_stop_reason: str | None = None
    last_error: str | None = None
    closed: bool = False
    #: A ``oneshot`` session between turns: no process, resumable.
    idle: bool = False
    holds_slot: bool = False
    process_key: str = ""
    #: ``run_turn`` calls holding or waiting for ``lock`` (never pruned while > 0).
    in_use: int = 0
    activity: str | None = None
    activity_kind: str | None = None
    #: This turn's harness tool calls: id -> (title, kind, raw_input, tool name).
    turn_tools: dict[str, tuple[str | None, str | None, dict[str, Any] | None, str | None]] = field(
        default_factory=dict
    )


def _cfg(config: Any, name: str, default: Any) -> Any:
    value = getattr(config, name, None) if config is not None else None
    return default if value is None else value


def agent_display_name(agent_id: str | None) -> str:
    """Registry display name (``claude`` -> ``Claude Code``); the id otherwise."""
    aid = str(agent_id or "")
    harness = get_harness(aid) if aid else None
    return harness.name if harness is not None else aid


def origin_is_closed(origin: Mapping[str, Any] | None, *, session_status: Any = None) -> bool:
    """True when an operator closed the session (``acp.close`` / ``subagents kill``).

    A ``oneshot`` session that merely finished its turn is idle, not closed.
    Rows written before that distinction existed carry ``closed`` without a
    ``closed_reason`` for both cases; for a ``oneshot`` row that was not
    killed, that legacy flag means "finished" and the session stays resumable.
    """
    if not isinstance(origin, Mapping) or not origin.get("closed"):
        return False
    if origin.get("closed_reason"):
        return True
    if origin.get("mode") != "oneshot":
        return True
    return str(getattr(session_status, "value", session_status) or "") == "killed"


def starting_label(agent_id: str | None) -> str:
    """``activity`` while the harness process starts ("Starting Claude Code…")."""
    return f"Starting {agent_display_name(agent_id) or 'the harness'}…"


def resume_notice(agent_id: str | None) -> str:
    """Text a turn starts with when the harness could not resume its session."""
    name = agent_display_name(agent_id) or "harness"
    return f"Started a new {name} session; the previous context could not be resumed."


def spec_from_origin(
    origin: Mapping[str, Any] | None,
    config: Any = None,
    *,
    session_key: str = "",
) -> AcpSessionSpec:
    """Rebuild the harness spec from an ACP session row ``origin``."""
    data = dict(origin or {})
    agent_id = normalize_agent_id(str(data.get("agent_id") or _cfg(config, "default_agent", "")))
    cwd = str(data.get("cwd") or "")
    if not agent_id or not cwd:
        raise AcpError(
            f"ACP session {session_key or '?'} has no harness agent/cwd in its origin",
            code="ACP_SESSION_INVALID",
        )
    mode: SessionMode = "oneshot" if data.get("mode") == "oneshot" else "persistent"
    resume = data.get("acp_session_id") or data.get("resume_session_id")
    try:
        turn_timeout = float(data.get("turn_timeout_s") or 0.0)
    except (TypeError, ValueError):
        turn_timeout = 0.0
    extra: dict[str, Any] = {"session_key": session_key}
    if data.get("record_id"):
        extra["record_id"] = data["record_id"]
    return AcpSessionSpec(
        agent_id=agent_id,
        cwd=cwd,
        mode=mode,
        resume_session_id=str(resume) if resume else None,
        model=data.get("model") or None,
        harness_mode=data.get("harness_mode") or None,
        label=str(data.get("label") or ""),
        turn_timeout_s=turn_timeout,
        extra=extra,
    )


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class AcpSessionManager:
    """Drives harness sessions for AgentOS session keys."""

    def __init__(
        self,
        runtime: AcpRuntime | None,
        *,
        config: Any = None,
        session_manager: Any = None,
        unavailable_reason: str | None = None,
        cancel_grace_s: float = _CANCEL_GRACE_S,
    ) -> None:
        self._runtime = runtime
        self._config = config
        self._session_manager = session_manager
        self._unavailable_reason = unavailable_reason
        self._cancel_grace_s = cancel_grace_s
        self._states: dict[str, _SessionState] = {}
        self._live = 0
        self._slot_cond: asyncio.Condition | None = None
        self._shutting_down = False
        #: session key -> [lock, holders]: serializes writes to one session row.
        self._row_locks: dict[str, list[Any]] = {}
        #: Set by the gateway wiring; ``None`` when the bridges are not built.
        self.permission_bridge: Any = None
        self.elicitation_bridge: Any = None

    # ------------------------------------------------------- bridge binding

    def bind_permissions(
        self,
        session_key: str,
        *,
        permission_mode: str | None = None,
        requester_ctx: Any = None,
        interactive: bool | None = None,
    ) -> None:
        """Tell the permission bridge which mode and operator surface this
        session has (spawn-time ``ToolContext`` decides unattended vs not)."""
        bind = getattr(self.permission_bridge, "bind_session", None)
        if callable(bind):
            bind(
                session_key,
                permission_mode=permission_mode,
                requester_ctx=requester_ctx,
                interactive=interactive,
            )

    def _forget_permissions(self, session_key: str) -> None:
        forget = getattr(self.permission_bridge, "forget_session", None)
        if callable(forget):
            forget(session_key)

    # ------------------------------------------------------------ properties

    @property
    def available(self) -> bool:
        return self._runtime is not None and not self._shutting_down

    @property
    def unavailable_reason(self) -> str | None:
        if self._runtime is None:
            return self._unavailable_reason or "ACP runtime is not installed"
        if self._shutting_down:
            return "gateway is shutting down"
        return None

    @property
    def runtime(self) -> AcpRuntime | None:
        return self._runtime

    @property
    def config(self) -> Any:
        return self._config

    @property
    def max_concurrent(self) -> int:
        try:
            return max(1, int(_cfg(self._config, "max_concurrent", 8)))
        except (TypeError, ValueError):
            return 8

    def _require_runtime(self) -> AcpRuntime:
        if self._runtime is None or self._shutting_down:
            raise AcpUnavailableError(f"ACP unavailable: {self.unavailable_reason}")
        return self._runtime

    def _state_for(self, session_key: str) -> _SessionState:
        state = self._states.get(session_key)
        if state is None:
            state = _SessionState(session_key=session_key)
            self._states[session_key] = state
        return state

    # ------------------------------------------------------------- admission

    def _cond(self) -> asyncio.Condition:
        if self._slot_cond is None:
            self._slot_cond = asyncio.Condition()
        return self._slot_cond

    def _idle_victim(self, exclude: _SessionState) -> _SessionState | None:
        candidates = [
            s
            for s in self._states.values()
            if s is not exclude
            and s.holds_slot
            and s.handle is not None
            and s.active_turn is None
            and not s.lock.locked()
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda s: s.last_used)

    async def _acquire_slot(self, state: _SessionState) -> None:
        if state.holds_slot:
            return
        cond = self._cond()
        while True:
            victim: _SessionState | None = None
            async with cond:
                if self._live < self.max_concurrent:
                    self._live += 1
                    state.holds_slot = True
                    return
                victim = self._idle_victim(state)
                if victim is None:
                    log.info(
                        "acp.manager.admission_wait",
                        session_key=state.session_key,
                        live=self._live,
                        max_concurrent=self.max_concurrent,
                    )
                    await cond.wait()
                    continue
            log.info(
                "acp.manager.evict_idle",
                session_key=victim.session_key,
                for_session_key=state.session_key,
            )
            await self._close_handle(victim, discard=False, reason="evicted_for_capacity")

    async def _release_slot(self, state: _SessionState) -> None:
        if not state.holds_slot:
            return
        cond = self._cond()
        async with cond:
            state.holds_slot = False
            self._live = max(0, self._live - 1)
            cond.notify_all()

    # --------------------------------------------------------------- origin

    async def _get_session_row(self, session_key: str) -> Any:
        sm = self._session_manager
        if sm is None:
            return None
        try:
            return await sm.get_session(session_key)
        except Exception:  # noqa: BLE001 - row lookup is best effort here
            log.debug("acp.manager.session_lookup_failed", session_key=session_key, exc_info=True)
            return None

    @contextlib.asynccontextmanager
    async def row_guard(self, session_key: str) -> AsyncIterator[None]:
        """Hold while reading and rewriting this session's row.

        ``SessionManager.update`` / ``append_message`` rewrite the whole row,
        so two writers that interleave lose one write: a turn's bookkeeping
        racing ``close`` dropped ``closed``. Every ACP writer of the row (this
        manager's origin updates, the dispatcher's end-of-turn writes) goes
        through this guard. Not reentrant.
        """
        entry = self._row_locks.get(session_key)
        if entry is None:
            entry = [asyncio.Lock(), 0]
            self._row_locks[session_key] = entry
        entry[1] += 1
        try:
            async with entry[0]:
                yield
        finally:
            entry[1] -= 1
            if entry[1] == 0 and self._row_locks.get(session_key) is entry:
                del self._row_locks[session_key]

    async def update_origin(self, session_key: str, **changes: Any) -> dict[str, Any] | None:
        """Merge ``changes`` into the session row ``origin`` (best effort)."""
        sm = self._session_manager
        if sm is None:
            return None
        async with self.row_guard(session_key):
            row = await self._get_session_row(session_key)
            if row is None:
                return None
            origin = dict(getattr(row, "origin", None) or {})
            origin.update(changes)
            try:
                await sm.update(session_key, origin=origin)
            except Exception:  # noqa: BLE001 - never fail a turn on bookkeeping
                log.warning(
                    "acp.manager.origin_update_failed", session_key=session_key, exc_info=True
                )
                return None
        return origin

    async def _spec_from_row(self, session_key: str) -> AcpSessionSpec:
        row = await self._get_session_row(session_key)
        origin = getattr(row, "origin", None) if row is not None else None
        if not isinstance(origin, Mapping) or origin.get("kind") != "acp":
            raise AcpError(
                f"Session {session_key} is not an ACP session", code="ACP_SESSION_INVALID"
            )
        if origin_is_closed(origin, session_status=getattr(row, "status", None)):
            raise AcpSessionClosedError(f"ACP session {session_key} was closed")
        return spec_from_origin(origin, self._config, session_key=session_key)

    # ---------------------------------------------------------------- turns

    async def _ensure_handle(
        self, state: _SessionState, spec: AcpSessionSpec
    ) -> tuple[AcpSessionHandle, bool]:
        """The live handle for ``spec`` and whether a requested resume was lost.

        ``resume_lost`` is True when a new harness process was started for a
        ``resume_session_id`` but came up with a different ACP session (the
        harness cannot ``session/load``, the record no longer matches, or the
        load failed): the runtime then falls back to a fresh session.
        """
        runtime = self._require_runtime()
        handle = state.handle
        if (
            handle is not None
            and handle.status not in ("closed", "unhealthy")
            and normalize_agent_id(handle.agent_id) == normalize_agent_id(spec.agent_id)
            and handle.cwd == spec.cwd
        ):
            return handle, False
        if handle is not None:
            await self._close_handle(state, discard=False, reason="stale_handle")
        await self._acquire_slot(state)
        try:
            handle = await runtime.ensure_session(spec, session_key=state.session_key)
        except BaseException:
            await self._release_slot(state)
            raise
        state.handle = handle
        state.spec = spec
        state.process_key = f"{handle.record_id}:{handle.pid or ''}:{handle.acp_session_id or ''}"
        resume_lost = bool(spec.resume_session_id) and (
            handle.acp_session_id != spec.resume_session_id
        )
        log.info(
            "acp.manager.session_ready",
            session_key=state.session_key,
            agent_id=handle.agent_id,
            record_id=handle.record_id,
            resumed=bool(spec.resume_session_id) and not resume_lost,
        )
        if resume_lost:
            log.warning(
                "acp.manager.resume_lost",
                session_key=state.session_key,
                agent_id=handle.agent_id,
                requested=spec.resume_session_id,
                started=handle.acp_session_id,
            )
        # Persist the identity as soon as it exists so a gateway crash in the
        # middle of the first turn still leaves a resumable session row. A
        # oneshot session is no longer idle once its harness is back; a legacy
        # ``closed`` flag from an earlier oneshot finish is cleared too (an
        # operator-closed row never gets this far).
        changes: dict[str, Any] = {
            "record_id": handle.record_id,
            "acp_session_id": handle.acp_session_id,
        }
        if spec.mode == "oneshot":
            changes.update(idle=False, closed=False)
        await self.update_origin(state.session_key, **changes)
        return handle, resume_lost

    async def _cancel_turn(self, state: _SessionState, turn: AcpTurn, *, reason: str) -> None:
        log.info("acp.manager.cancel_turn", session_key=state.session_key, reason=reason)
        try:
            await asyncio.wait_for(turn.cancel(), timeout=self._cancel_grace_s)
        except Exception:  # noqa: BLE001 - cancel is best effort, the runtime escalates
            log.warning("acp.manager.cancel_failed", session_key=state.session_key, exc_info=True)

    async def run_turn(
        self,
        session_key: str,
        text: str,
        *,
        spec: AcpSessionSpec | None = None,
        attachments: list[dict[str, Any]] | None = None,
        on_event: EventCallback | None = None,
    ) -> AcpTurnResult:
        state = self._state_for(session_key)
        state.in_use += 1
        try:
            return await self._run_turn(
                session_key, text, spec=spec, attachments=attachments, on_event=on_event
            )
        finally:
            state.in_use -= 1
            self._prune_idle_states()
            # The session just went idle (or away): a turn waiting for a
            # harness slot may now evict it.
            await self._notify_slot_waiters()

    def _prune_idle_states(self) -> None:
        """Bound the process-less states kept for ``status`` (oldest go first)."""
        idle = [
            s
            for s in self._states.values()
            if s.handle is None
            and s.active_turn is None
            and s.in_use == 0
            and not s.lock.locked()
            and not s.holds_slot
        ]
        overflow = len(idle) - _MAX_IDLE_STATES
        if overflow <= 0:
            return
        for state in sorted(idle, key=lambda s: s.last_used)[:overflow]:
            if self._states.get(state.session_key) is state:
                del self._states[state.session_key]

    def prepare_turn(self, session_key: str, agent_id: str | None = None) -> None:
        """A turn for this session was admitted: drop the previous turn's step
        at once, so ``status`` never shows it while the new turn gets going.

        Synchronous on purpose (the dispatcher calls it before its first
        await). A turn that is still running keeps its activity.
        """
        state = self._state_for(session_key)
        if state.active_turn is not None or state.lock.locked():
            return
        self._reset_activity(state)
        state.idle = False
        if state.handle is None:
            state.activity = starting_label(
                agent_id or (state.spec.agent_id if state.spec is not None else None)
            )

    @staticmethod
    def _reset_activity(state: _SessionState) -> None:
        state.activity = None
        state.activity_kind = None
        state.turn_tools = {}

    @staticmethod
    def _track_activity(state: _SessionState, event: AcpEvent) -> None:
        """Keep ``activity`` / ``activity_kind`` / tool count current for status."""
        kind = event.kind
        if kind in ("tool_call", "tool_call_update"):
            tool_id = event.tool_call_id or f"acp-tool-{len(state.turn_tools) + 1}"
            title, tool_kind, raw_input, tool_name = state.turn_tools.get(
                tool_id, (None, None, None, None)
            )
            title = event.title or title
            tool_kind = event.tool_kind or tool_kind
            if isinstance(event.raw_input, dict) and event.raw_input:
                raw_input = dict(event.raw_input)
            tool_name = tool_name or harness_tool_name(event.raw) or None
            state.turn_tools[tool_id] = (title, tool_kind, raw_input, tool_name)
            if event.title or event.tool_kind or event.raw_input or kind == "tool_call":
                normalized = tool_kind if tool_kind in TOOL_KINDS else "other"
                state.activity = tool_label(title, normalized, raw_input, tool_name=tool_name)
                state.activity_kind = normalized
        elif kind == "text_delta" and event.text:
            state.activity = TEXT_ACTIVITY_LABEL
            state.activity_kind = TEXT_ACTIVITY_KIND
        elif kind == "thought_delta" and event.text:
            state.activity = KIND_LABELS["think"]
            state.activity_kind = "think"

    async def _notify_slot_waiters(self) -> None:
        if self._slot_cond is None:
            return
        async with self._slot_cond:
            self._slot_cond.notify_all()

    async def _run_turn(
        self,
        session_key: str,
        text: str,
        *,
        spec: AcpSessionSpec | None = None,
        attachments: list[dict[str, Any]] | None = None,
        on_event: EventCallback | None = None,
    ) -> AcpTurnResult:
        """Run one harness turn and return its result.

        Events are delivered to ``on_event`` as they arrive. Raises
        :class:`AcpError` subclasses on failure (``AcpTurnTimeoutError`` on
        the wall clock, ``AcpTurnError`` when the harness reports an error).
        ``asyncio.CancelledError`` cancels the harness turn before it
        propagates.
        """
        runtime = self._require_runtime()
        state = self._state_for(session_key)
        epoch = state.epoch
        async with state.lock:
            if state.closed or state.epoch != epoch:
                raise AcpSessionClosedError(f"ACP session {session_key} was closed")
            if spec is None:
                spec = await self._spec_from_row(session_key)
            state.spec = spec
            state.idle = False
            self._reset_activity(state)
            # The turn's clock (``turn_elapsed_s``) includes the harness start.
            state.turn_started_at = time.monotonic()
            state.last_used = state.turn_started_at
            if state.handle is None:
                state.activity = starting_label(spec.agent_id)
            try:
                handle, resume_lost = await self._ensure_handle(state, spec)
            except BaseException:
                state.turn_started_at = None
                if spec.mode == "oneshot" and state.handle is None:
                    # Still a resumable idle session; this turn never started.
                    state.idle = True
                raise
            timeout_s = float(spec.turn_timeout_s or 0) or float(
                _cfg(self._config, "turn_timeout_s", 0) or 0
            )
            if resume_lost and on_event is not None:
                # Part of this turn's text (live stream, transcript and the
                # parent's completion): the harness starts without the context
                # the caller expects it to have.
                notice = AcpEvent(
                    kind="text_delta",
                    tag="agentos_notice",
                    text=resume_notice(spec.agent_id) + "\n\n",
                )
                try:
                    await _maybe_await(on_event(notice))
                except Exception:  # noqa: BLE001 - a sink failure must not kill the turn
                    log.warning("acp.manager.event_sink_failed", session_key=session_key)
            try:
                turn = await runtime.start_turn(
                    handle, text=text, attachments=attachments, session_key=session_key
                )
            except BaseException:
                state.turn_started_at = None
                if spec.mode == "oneshot":
                    await asyncio.shield(self._finish_oneshot(state))
                raise
            state.active_turn = turn
            log.info(
                "acp.manager.turn_start",
                session_key=session_key,
                agent_id=handle.agent_id,
                mode=spec.mode,
                timeout_s=timeout_s or None,
            )
            try:
                result = await self._drive_turn(state, turn, on_event, timeout_s, spec)
            except asyncio.CancelledError:
                await asyncio.shield(self._cancel_turn(state, turn, reason="task_cancelled"))
                state.last_stop_reason = "cancelled"
                raise
            except AcpError as exc:
                state.last_error = str(exc)
                state.last_stop_reason = "error"
                raise
            except Exception as exc:
                state.last_error = str(exc)
                state.last_stop_reason = "error"
                raise AcpTurnError(str(exc) or type(exc).__name__) from exc
            finally:
                state.active_turn = None
                state.turn_started_at = None
                state.last_used = time.monotonic()
                with contextlib.suppress(Exception):
                    await turn.close_stream()
                if spec.mode == "oneshot":
                    await asyncio.shield(self._finish_oneshot(state))
            state.turns += 1
            state.last_stop_reason = result.stop_reason
            state.last_error = result.error
            log.info(
                "acp.manager.turn_done",
                session_key=session_key,
                stop_reason=result.stop_reason,
                duration_ms=result.duration_ms,
                tool_calls=result.tool_calls,
            )
            await self.update_origin(
                session_key,
                record_id=handle.record_id,
                acp_session_id=handle.acp_session_id,
                last_stop_reason=result.stop_reason,
            )
            if result.stop_reason == "error":
                raise AcpTurnError(result.error or "harness turn failed")
            return result

    async def _drive_turn(
        self,
        state: _SessionState,
        turn: AcpTurn,
        on_event: EventCallback | None,
        timeout_s: float,
        spec: AcpSessionSpec,
    ) -> AcpTurnResult:
        async def _drain() -> AcpTurnResult:
            async for event in turn.events:
                self._track_activity(state, event)
                if on_event is None:
                    continue
                try:
                    await _maybe_await(on_event(event))
                except Exception:  # noqa: BLE001 - a sink failure must not kill the turn
                    log.warning(
                        "acp.manager.event_sink_failed",
                        session_key=state.session_key,
                        event_kind=event.kind,
                        exc_info=True,
                    )
            return await turn.result

        if timeout_s <= 0:
            return await _drain()
        try:
            async with asyncio.timeout(timeout_s):
                return await _drain()
        except TimeoutError as exc:
            await self._cancel_turn(state, turn, reason="turn_timeout")
            raise AcpTurnTimeoutError(
                f"ACP turn timed out after {int(timeout_s)}s", timeout_s=timeout_s
            ) from exc

    async def _finish_oneshot(self, state: _SessionState) -> None:
        """Close the harness process after a oneshot turn; the session stays
        resumable (``origin.idle``) so a follow-up turn spawns it again with
        the recorded ``acp_session_id``."""
        await self._close_handle(state, discard=False, reason="oneshot_done")
        if state.closed:
            return  # an operator close raced this turn; it owns the row now
        state.idle = True
        await self.update_origin(state.session_key, idle=True)

    def _forget(self, state: _SessionState) -> None:
        """Drop a closed session's in-memory state; the row origin remembers
        ``closed`` so a later turn is still refused."""
        if self._states.get(state.session_key) is state and state.handle is None:
            del self._states[state.session_key]

    async def _close_handle(self, state: _SessionState, *, discard: bool, reason: str) -> None:
        handle = state.handle
        state.handle = None
        try:
            if handle is not None and self._runtime is not None:
                await asyncio.wait_for(
                    self._runtime.close(handle, discard=discard), timeout=_CLOSE_GRACE_S
                )
                log.info(
                    "acp.manager.session_closed",
                    session_key=state.session_key,
                    record_id=handle.record_id,
                    reason=reason,
                    discard=discard,
                )
        except Exception:  # noqa: BLE001 - the runtime / reaper own the process
            log.warning(
                "acp.manager.close_failed",
                session_key=state.session_key,
                reason=reason,
                exc_info=True,
            )
        finally:
            await self._release_slot(state)

    # ------------------------------------------------------------- controls

    def process_key(self, session_key: str) -> str:
        """Identity of the harness process that served the last turn; cost
        readings are cumulative per process, so baselines key off this."""
        state = self._states.get(session_key)
        return state.process_key if state is not None else ""

    def is_turn_active(self, session_key: str) -> bool:
        state = self._states.get(session_key)
        return bool(state is not None and state.active_turn is not None)

    async def cancel(self, session_key: str, *, reason: str = "operator_cancel") -> bool:
        """Cancel the running turn, if any. Returns True when one was running."""
        state = self._states.get(session_key)
        if state is None:
            return False
        turn = state.active_turn
        if turn is not None:
            await self._cancel_turn(state, turn, reason=reason)
            return True
        if state.handle is not None and self._runtime is not None:
            with contextlib.suppress(Exception):
                await self._runtime.cancel(state.handle, reason=reason)
        return False

    async def close(
        self, session_key: str, *, discard: bool = False, reason: str = "operator_close"
    ) -> bool:
        """Cancel any turn, close the harness and mark the session closed.

        Unlike a finished ``oneshot`` turn (idle, resumable), a closed session
        refuses every later turn; ``closed_reason`` records who closed it.
        """
        state = self._state_for(session_key)
        state.closed = True
        state.idle = False
        state.epoch += 1
        had_handle = state.handle is not None
        turn = state.active_turn
        if turn is not None:
            await self._cancel_turn(state, turn, reason=reason)
        try:
            async with asyncio.timeout(_CLOSE_GRACE_S + self._cancel_grace_s):
                async with state.lock:
                    await self._close_handle(state, discard=discard, reason=reason)
        except TimeoutError:
            log.warning("acp.manager.close_lock_timeout", session_key=session_key)
            await self._close_handle(state, discard=discard, reason=reason)
        await self.update_origin(
            session_key, closed=True, closed_reason=reason or "operator_close", idle=False
        )
        self._forget(state)
        self._forget_permissions(session_key)
        return had_handle

    async def _live_handle(self, session_key: str) -> AcpSessionHandle:
        state = self._state_for(session_key)
        if state.handle is not None and state.handle.status not in ("closed", "unhealthy"):
            return state.handle
        if state.closed:
            raise AcpSessionClosedError(f"ACP session {session_key} was closed")
        async with state.lock:
            spec = await self._spec_from_row(session_key)
            handle, _resume_lost = await self._ensure_handle(state, spec)
            return handle

    async def set_mode(self, session_key: str, mode_id: str) -> dict[str, Any]:
        runtime = self._require_runtime()
        handle = await self._live_handle(session_key)
        await runtime.set_mode(handle, mode_id)
        handle.current_mode_id = mode_id
        await self.update_origin(session_key, harness_mode=mode_id)
        return {"session_key": session_key, "mode": mode_id}

    @staticmethod
    def resolve_option_id(handle: AcpSessionHandle, key: str) -> str | None:
        """Map a logical option (``model`` / ``thinking`` / ``permissions``)
        or a raw option id to the id the harness advertised."""
        options = [o for o in handle.config_options or [] if isinstance(o, Mapping)]
        candidates = OPTION_ALIASES.get(key, (key,))
        for candidate in candidates:
            for option in options:
                if str(option.get("id") or "") == candidate:
                    return candidate
        for candidate in candidates:
            for option in options:
                if str(option.get("category") or "") == candidate and option.get("id"):
                    return str(option["id"])
        return None

    async def set_option(self, session_key: str, key: str, value: Any) -> dict[str, Any]:
        runtime = self._require_runtime()
        handle = await self._live_handle(session_key)
        option_id = self.resolve_option_id(handle, key)
        if option_id is None:
            advertised = sorted(
                str(o.get("id"))
                for o in handle.config_options or []
                if isinstance(o, Mapping) and o.get("id")
            )
            raise AcpOptionUnsupportedError(
                f"Harness '{handle.agent_id}' does not advertise a '{key}' option"
                + (f" (advertised: {', '.join(advertised)})" if advertised else ""),
                advertised=advertised,
            )
        await runtime.set_config_option(handle, option_id, value)
        row = await self._get_session_row(session_key)
        origin = getattr(row, "origin", None) if row is not None else None
        options = dict((origin or {}).get("config_options") or {})
        options[option_id] = value
        changes: dict[str, Any] = {"config_options": options}
        if key == "model":
            changes["model"] = value
        await self.update_origin(session_key, **changes)
        return {"session_key": session_key, "option": option_id, "value": value}

    async def set_model(self, session_key: str, model: str) -> dict[str, Any]:
        return await self.set_option(session_key, "model", model)

    # --------------------------------------------------------------- status

    def _describe(self, state: _SessionState) -> dict[str, Any]:
        handle = state.handle
        now = time.monotonic()
        info: dict[str, Any] = {
            "session_key": state.session_key,
            "live": handle is not None,
            "active_turn": state.active_turn is not None,
            "turns": state.turns,
            "closed": state.closed,
            "idle": bool(state.idle and handle is None and state.active_turn is None),
            "epoch": state.epoch,
            "last_stop_reason": state.last_stop_reason,
            "last_error": state.last_error,
            "idle_s": round(now - state.last_used, 1),
            "activity": state.activity,
            "activity_kind": state.activity_kind,
            "turn_tool_calls": len(state.turn_tools),
        }
        if state.turn_started_at is not None:
            info["turn_elapsed_s"] = round(now - state.turn_started_at, 1)
        spec = state.spec
        if spec is not None:
            info.update(
                {
                    "agent_id": spec.agent_id,
                    "cwd": spec.cwd,
                    "mode": spec.mode,
                    "harness_mode": spec.harness_mode,
                }
            )
        if handle is not None:
            info.update(
                {
                    "agent_id": handle.agent_id,
                    "cwd": handle.cwd,
                    "record_id": handle.record_id,
                    "acp_session_id": handle.acp_session_id,
                    "pid": handle.pid,
                    "status": handle.status,
                    "current_mode_id": handle.current_mode_id,
                    "available_modes": [
                        m.get("id") for m in handle.available_modes if isinstance(m, Mapping)
                    ],
                    "config_options": [
                        o.get("id") for o in handle.config_options if isinstance(o, Mapping)
                    ],
                }
            )
        info["agent_name"] = agent_display_name(info.get("agent_id")) or None
        return info

    async def status(self, session_key: str, *, require_acp: bool = False) -> dict[str, Any]:
        """Everything known about one ACP session: live state, plus the row
        ``origin`` for sessions without a process (idle oneshot, closed,
        not yet resumed after a restart).

        ``require_acp`` raises ``ACP_SESSION_INVALID`` when the key is not an
        ACP session row (the ``acp.status`` RPC contract).
        """
        row = await self._get_session_row(session_key)
        origin = getattr(row, "origin", None) if row is not None else None
        is_acp = isinstance(origin, Mapping) and origin.get("kind") == "acp"
        if require_acp and not is_acp:
            raise AcpError(
                f"Session {session_key} is not an ACP session", code="ACP_SESSION_INVALID"
            )
        state = self._states.get(session_key)
        if state is not None and state.handle is not None and self._runtime is not None:
            try:
                state.handle = await self._runtime.status(state.handle)
            except Exception:  # noqa: BLE001 - status is informational
                log.debug("acp.manager.status_failed", session_key=session_key, exc_info=True)
        if state is not None:
            info = self._describe(state)
        else:
            info = {
                "session_key": session_key,
                "live": False,
                "active_turn": False,
                "closed": False,
                "idle": False,
                "activity": None,
                "activity_kind": None,
                "turn_tool_calls": 0,
            }
        if is_acp:
            assert isinstance(origin, Mapping)
            for key in ("agent_id", "cwd", "mode", "record_id", "acp_session_id"):
                if info.get(key) is None:
                    info[key] = origin.get(key)
            if info.get("harness_mode") is None:
                info["harness_mode"] = origin.get("harness_mode")
            session_status = getattr(row, "status", None)
            closed = bool(info.get("closed")) or origin_is_closed(
                origin, session_status=session_status
            )
            info["closed"] = closed
            busy = bool(info.get("live") or info.get("active_turn") or "turn_elapsed_s" in info)
            if state is not None:
                idle = bool(info.get("idle"))
            else:
                # After a restart only the row knows; a legacy finished oneshot
                # row says ``closed`` without a reason (see origin_is_closed).
                idle = bool(origin.get("idle")) or (
                    origin.get("mode") == "oneshot" and bool(origin.get("closed"))
                )
            info["idle"] = idle and not closed and not busy
            last_turn = origin.get("last_turn")
            info["last_turn"] = last_turn if isinstance(last_turn, Mapping) else None
            if (
                state is None
                and isinstance(last_turn, Mapping)
                and isinstance(last_turn.get("tool_calls"), int)
            ):
                # The in-memory tally is gone (restart, pruned): the last turn's
                # count is the best answer for "how many tool calls".
                info["turn_tool_calls"] = last_turn["tool_calls"]
            if origin.get("worktree"):
                info["worktree"] = origin.get("worktree")
            info["session_status"] = getattr(session_status, "value", session_status)
        # UIs key off these: always present, even when unknown.
        info["session_key"] = session_key
        info.setdefault("agent_id", None)
        for key in ("cwd", "mode", "harness_mode", "last_turn"):
            info.setdefault(key, None)
        info["agent_name"] = agent_display_name(info.get("agent_id")) or None
        info["available"] = self.available
        return info

    def list_sessions(self) -> list[dict[str, Any]]:
        """Sessions with a harness process or a turn in progress (including a
        harness that is still starting)."""
        return [
            self._describe(state)
            for state in self._states.values()
            if state.handle is not None
            or state.active_turn is not None
            or state.turn_started_at is not None
        ]

    @property
    def live_count(self) -> int:
        return self._live

    # ------------------------------------------------------------- shutdown

    async def shutdown(self) -> None:
        """Cancel running turns and close every live harness (resumable)."""
        self._shutting_down = True
        states = list(self._states.values())
        for state in states:
            turn = state.active_turn
            if turn is not None:
                await self._cancel_turn(state, turn, reason="gateway_shutdown")
        await asyncio.gather(
            *(
                self._close_handle(state, discard=False, reason="gateway_shutdown")
                for state in states
                if state.handle is not None
            ),
            return_exceptions=True,
        )
        log.info("acp.manager.shutdown", sessions=len(states))


def is_superseded_task(origin: Mapping[str, Any] | None, task_id: str | None) -> bool:
    """True when ``task_id`` was replaced by a steer (its announce is skipped)."""
    if not task_id or not isinstance(origin, Mapping):
        return False
    return task_id in (origin.get("superseded_task_ids") or [])


# --------------------------------------------------------------------------
# Process-wide accessor (gateway boot sets it; tools and dispatch read it)
# --------------------------------------------------------------------------

_manager: AcpSessionManager | None = None


def set_acp_manager(manager: AcpSessionManager | None) -> None:
    global _manager
    _manager = manager


def get_acp_manager() -> AcpSessionManager | None:
    return _manager
