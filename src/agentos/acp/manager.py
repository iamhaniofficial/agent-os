"""``AcpSessionManager``: one per gateway, owns every live harness session.

Responsibilities (OpenClaw ``src/acp/control-plane/manager.core.ts`` and
friends, MIT, rewritten for asyncio; see ``THIRD_PARTY_NOTICES.md``):

- an actor per AgentOS session key (an ``asyncio.Lock`` plus an epoch that
  is bumped on close) so turns on one session never interleave;
- a handle cache so a ``persistent`` session keeps its harness process
  across turns, and resumes from the session row ``origin`` when it does not;
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
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import structlog

from agentos.acp.registry import normalize_agent_id
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
    holds_slot: bool = False
    process_key: str = ""


def _cfg(config: Any, name: str, default: Any) -> Any:
    value = getattr(config, name, None) if config is not None else None
    return default if value is None else value


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

    async def update_origin(self, session_key: str, **changes: Any) -> dict[str, Any] | None:
        """Merge ``changes`` into the session row ``origin`` (best effort)."""
        sm = self._session_manager
        if sm is None:
            return None
        row = await self._get_session_row(session_key)
        if row is None:
            return None
        origin = dict(getattr(row, "origin", None) or {})
        origin.update(changes)
        try:
            await sm.update(session_key, origin=origin)
        except Exception:  # noqa: BLE001 - never fail a turn on bookkeeping
            log.warning("acp.manager.origin_update_failed", session_key=session_key, exc_info=True)
            return None
        return origin

    async def _spec_from_row(self, session_key: str) -> AcpSessionSpec:
        row = await self._get_session_row(session_key)
        origin = getattr(row, "origin", None) if row is not None else None
        if not isinstance(origin, Mapping) or origin.get("kind") != "acp":
            raise AcpError(
                f"Session {session_key} is not an ACP session", code="ACP_SESSION_INVALID"
            )
        if origin.get("closed"):
            raise AcpSessionClosedError(f"ACP session {session_key} was closed")
        return spec_from_origin(origin, self._config, session_key=session_key)

    # ---------------------------------------------------------------- turns

    async def _ensure_handle(self, state: _SessionState, spec: AcpSessionSpec) -> AcpSessionHandle:
        runtime = self._require_runtime()
        handle = state.handle
        if (
            handle is not None
            and handle.status not in ("closed", "unhealthy")
            and normalize_agent_id(handle.agent_id) == normalize_agent_id(spec.agent_id)
            and handle.cwd == spec.cwd
        ):
            return handle
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
        log.info(
            "acp.manager.session_ready",
            session_key=state.session_key,
            agent_id=handle.agent_id,
            record_id=handle.record_id,
            resumed=bool(spec.resume_session_id),
        )
        # Persist the identity as soon as it exists so a gateway crash in the
        # middle of the first turn still leaves a resumable session row.
        await self.update_origin(
            state.session_key,
            record_id=handle.record_id,
            acp_session_id=handle.acp_session_id,
        )
        return handle

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
        try:
            return await self._run_turn(
                session_key, text, spec=spec, attachments=attachments, on_event=on_event
            )
        finally:
            # The session just went idle (or away): a turn waiting for a
            # harness slot may now evict it.
            await self._notify_slot_waiters()

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
            handle = await self._ensure_handle(state, spec)
            timeout_s = float(spec.turn_timeout_s or 0) or float(
                _cfg(self._config, "turn_timeout_s", 0) or 0
            )
            state.turn_started_at = time.monotonic()
            state.last_used = state.turn_started_at
            turn = await runtime.start_turn(
                handle, text=text, attachments=attachments, session_key=session_key
            )
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
        state.closed = True
        state.epoch += 1
        await self._close_handle(state, discard=False, reason="oneshot_done")
        await self.update_origin(state.session_key, closed=True)
        self._forget(state)

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
        """Cancel any turn, close the harness and mark the session closed."""
        state = self._state_for(session_key)
        state.closed = True
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
        await self.update_origin(session_key, closed=True)
        self._forget(state)
        return had_handle

    async def _live_handle(self, session_key: str) -> AcpSessionHandle:
        state = self._state_for(session_key)
        if state.handle is not None and state.handle.status not in ("closed", "unhealthy"):
            return state.handle
        if state.closed:
            raise AcpSessionClosedError(f"ACP session {session_key} was closed")
        async with state.lock:
            spec = await self._spec_from_row(session_key)
            return await self._ensure_handle(state, spec)

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
            "epoch": state.epoch,
            "last_stop_reason": state.last_stop_reason,
            "last_error": state.last_error,
            "idle_s": round(now - state.last_used, 1),
        }
        if state.turn_started_at is not None:
            info["turn_elapsed_s"] = round(now - state.turn_started_at, 1)
        spec = state.spec
        if spec is not None:
            info.update({"agent_id": spec.agent_id, "cwd": spec.cwd, "mode": spec.mode})
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
        return info

    async def status(self, session_key: str) -> dict[str, Any]:
        state = self._states.get(session_key)
        if state is not None and state.handle is not None and self._runtime is not None:
            try:
                state.handle = await self._runtime.status(state.handle)
            except Exception:  # noqa: BLE001 - status is informational
                log.debug("acp.manager.status_failed", session_key=session_key, exc_info=True)
        if state is not None:
            info = self._describe(state)
        else:
            info = {"session_key": session_key, "live": False, "active_turn": False}
        row = await self._get_session_row(session_key)
        origin = getattr(row, "origin", None) if row is not None else None
        if isinstance(origin, Mapping) and origin.get("kind") == "acp":
            for key in ("agent_id", "cwd", "mode", "record_id", "acp_session_id"):
                info.setdefault(key, origin.get(key))
            info["closed"] = bool(info.get("closed") or origin.get("closed"))
            if origin.get("last_turn"):
                info["last_turn"] = origin.get("last_turn")
            if origin.get("worktree"):
                info["worktree"] = origin.get("worktree")
            info["session_status"] = getattr(row, "status", None)
        info["available"] = self.available
        return info

    def list_sessions(self) -> list[dict[str, Any]]:
        return [
            self._describe(state)
            for state in self._states.values()
            if state.handle is not None or state.active_turn is not None
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
