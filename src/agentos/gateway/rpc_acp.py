"""RPC handlers for the ACP coding-harness consumer (``acp.*``).

``acp.agents`` and a probe-less ``acp.doctor`` only read the registry and
config, so they answer on every gateway. ``acp.sessions``, ``acp.status``,
``acp.cancel``, ``acp.close`` and ``acp.doctor`` with ``probe`` need the
gateway's ``AcpSessionManager``; without one they fail with
``ACP_UNAVAILABLE``.

``acp.sessions`` lists sessions with a live harness process (each row carries
``agent_name``, ``activity``, ``activity_kind``, ``turn_tool_calls`` and,
while a turn runs, ``turn_elapsed_s``). ``acp.status {session_key}`` answers
for any ACP session -- live, idle oneshot or closed -- and fails with
``ACP_SESSION_INVALID`` for a key that is not an ACP session.

The manager is found through :func:`set_acp_manager` (boot wiring calls it)
or, failing that, an ``acp_manager`` attribute on the services the RPC
context already carries. Lookups are lazy so this module imports cleanly in
builds that do not ship the manager yet.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
from typing import Any

import structlog

from agentos.acp.inventory import (
    agents_payload,
    harness_row,
    known_agent_ids,
    to_plain,
)
from agentos.gateway.rpc import RpcContext, RpcHandlerError, get_dispatcher

_d = get_dispatcher()
log = structlog.get_logger(__name__)

ACP_UNAVAILABLE = "ACP_UNAVAILABLE"
PROBE_TIMEOUT_S = 30.0

_manager: Any = None


def set_acp_manager(manager: Any) -> None:
    """Register the gateway's ``AcpSessionManager`` (``None`` clears it)."""
    global _manager
    _manager = manager


def get_acp_manager(ctx: RpcContext | None = None) -> Any:
    if _manager is not None:
        return _manager
    if ctx is not None:
        for holder in (ctx.task_runtime, ctx.turn_runner, ctx.session_manager):
            found = getattr(holder, "acp_manager", None) if holder is not None else None
            if found is not None:
                return found
    try:
        mod = importlib.import_module("agentos.acp.manager")
    except ImportError:
        return None
    getter = getattr(mod, "get_acp_manager", None)
    if callable(getter):
        try:
            return getter()
        except Exception:  # noqa: BLE001 - a missing singleton reads as "unavailable"
            return None
    return None


def _require_manager(ctx: RpcContext) -> Any:
    manager = get_acp_manager(ctx)
    if manager is None:
        raise RpcHandlerError(
            ACP_UNAVAILABLE,
            "ACP session manager is not running on this gateway",
            details={"hint": "upgrade AgentOS or check [acp] enabled = true"},
        )
    return manager


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _params(params: dict | None) -> dict[str, Any]:
    if params is None:
        return {}
    if not isinstance(params, dict):
        raise ValueError("params must be an object")
    return params


def _session_key(params: dict[str, Any]) -> str:
    key = params.get("session_key") or params.get("key")
    if not isinstance(key, str) or not key.strip():
        raise ValueError("session_key is required")
    return key.strip()


def _map_acp_error(exc: Exception) -> RpcHandlerError:
    code = getattr(exc, "code", None)
    details = getattr(exc, "details", None)
    return RpcHandlerError(
        str(code) if isinstance(code, str) and code else "ACP_ERROR",
        str(exc) or type(exc).__name__,
        details=details if isinstance(details, dict) and details else None,
    )


@_d.method("acp.agents")
async def _handle_acp_agents(params: dict | None, ctx: RpcContext) -> dict[str, Any]:
    _params(params)
    payload = agents_payload(ctx.config)
    payload["manager"] = get_acp_manager(ctx) is not None
    return payload


@_d.method("acp.doctor")
async def _handle_acp_doctor(params: dict | None, ctx: RpcContext) -> dict[str, Any]:
    p = _params(params)
    agent = p.get("agent") or p.get("agent_id")
    probe = bool(p.get("probe", False))
    ids = [str(agent)] if agent else known_agent_ids(ctx.config)
    rows: list[dict[str, Any]] = []
    for aid in ids:
        try:
            rows.append(harness_row(aid, ctx.config))
        except KeyError as exc:
            raise KeyError(f"unknown ACP agent: {aid}") from exc
    manager = get_acp_manager(ctx)
    if probe:
        if manager is None:
            raise RpcHandlerError(ACP_UNAVAILABLE, "probe needs the ACP session manager")
        for row in rows:
            row["probe"] = await _probe(manager, row["agent_id"])
    return {"agents": rows, "manager": manager is not None}


async def _probe(manager: Any, agent_id: str) -> dict[str, Any]:
    doctor = getattr(manager, "doctor", None)
    if not callable(doctor):
        runtime = getattr(manager, "runtime", None)
        doctor = getattr(runtime, "doctor", None)
    if not callable(doctor):
        return {"probe": "error", "error": "runtime does not support doctor probes"}
    try:
        result = await asyncio.wait_for(_maybe_await(doctor(agent_id, probe=True)), PROBE_TIMEOUT_S)
    except TimeoutError:
        return {"probe": "error", "error": f"probe timed out after {int(PROBE_TIMEOUT_S)}s"}
    except Exception as exc:  # noqa: BLE001 - report, do not fail the whole doctor
        log.info("acp.doctor.probe_failed", agent_id=agent_id, error=type(exc).__name__)
        return {"probe": "error", "error": str(exc), "code": getattr(exc, "code", None)}
    plain = to_plain(result)
    return plain if isinstance(plain, dict) else {"probe": "ok", "result": plain}


@_d.method("acp.sessions")
async def _handle_acp_sessions(params: dict | None, ctx: RpcContext) -> dict[str, Any]:
    _params(params)
    manager = _require_manager(ctx)
    lister = getattr(manager, "list_sessions", None)
    if not callable(lister):
        raise RpcHandlerError(ACP_UNAVAILABLE, "ACP session manager cannot list sessions")
    rows = await _maybe_await(lister())
    sessions = [r for r in (to_plain(row) for row in rows or []) if isinstance(r, dict)]
    return {"sessions": sessions, "count": len(sessions)}


@_d.method("acp.status")
async def _handle_acp_status(params: dict | None, ctx: RpcContext) -> dict[str, Any]:
    p = _params(params)
    key = _session_key(p)
    acp_cfg = getattr(ctx.config, "acp", None)
    if acp_cfg is not None and getattr(acp_cfg, "enabled", True) is False:
        raise RpcHandlerError(ACP_UNAVAILABLE, "ACP is disabled ([acp] enabled = false)")
    manager = _require_manager(ctx)
    status = getattr(manager, "status", None)
    if not callable(status):
        raise RpcHandlerError(ACP_UNAVAILABLE, "ACP session manager cannot report status")
    try:
        result = await _call_with_optional_kw(status, key, require_acp=True)
    except Exception as exc:
        raise _map_acp_error(exc) from exc
    plain = to_plain(result)
    if not isinstance(plain, dict):
        raise RpcHandlerError("ACP_SESSION_INVALID", f"Session {key} is not an ACP session")
    return plain


async def _cancel_session_tasks(ctx: RpcContext, key: str, *, source: str) -> int:
    """Cancel the session's queued/running task-runtime turns (a harness that
    is still starting has no ACP turn to cancel yet)."""
    cancel = getattr(ctx.task_runtime, "cancel", None) if ctx.task_runtime is not None else None
    if not callable(cancel):
        return 0
    try:
        count = await _maybe_await(cancel(session_key=key, source=source, reason=source))
    except Exception:  # noqa: BLE001 - the harness cancel below still runs
        log.warning("acp.rpc.task_cancel_failed", session_key=key, exc_info=True)
        return 0
    return count if isinstance(count, int) else 0


@_d.method("acp.cancel")
async def _handle_acp_cancel(params: dict | None, ctx: RpcContext) -> dict[str, Any]:
    p = _params(params)
    key = _session_key(p)
    manager = _require_manager(ctx)
    reason = str(p.get("reason") or "operator")
    cancelled_tasks = await _cancel_session_tasks(ctx, key, source="acp_rpc_cancel")
    try:
        result = await _call_with_optional_kw(manager.cancel, key, reason=reason)
    except KeyError:
        raise
    except Exception as exc:
        raise _map_acp_error(exc) from exc
    log.info("acp.rpc.cancelled", session_key=key, tasks=cancelled_tasks)
    return {
        "session_key": key,
        "cancelled": True if result is None else bool(result or cancelled_tasks),
    }


@_d.method("acp.close")
async def _handle_acp_close(params: dict | None, ctx: RpcContext) -> dict[str, Any]:
    p = _params(params)
    key = _session_key(p)
    manager = _require_manager(ctx)
    discard = bool(p.get("discard", False))
    await _cancel_session_tasks(ctx, key, source="acp_rpc_close")
    try:
        result = await _call_with_optional_kw(manager.close, key, discard=discard)
    except KeyError:
        raise
    except Exception as exc:
        raise _map_acp_error(exc) from exc
    log.info("acp.rpc.closed", session_key=key, discard=discard)
    # ``close`` reports whether a harness process was running; the session is
    # closed either way (an idle oneshot has no process).
    return {"session_key": key, "closed": True, "was_live": result is None or bool(result)}


async def _call_with_optional_kw(fn: Any, key: str, **kwargs: Any) -> Any:
    """Call ``fn(key, **kwargs)`` passing only keywords ``fn`` accepts."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return await _maybe_await(fn(key))
    accepts_var_kw = any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values())
    passed = {k: v for k, v in kwargs.items() if accepts_var_kw or k in sig.parameters}
    return await _maybe_await(fn(key, **passed))
