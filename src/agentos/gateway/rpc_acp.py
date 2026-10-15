"""RPC handlers for the ACP coding-harness consumer (``acp.*``).

``acp.agents`` and a probe-less ``acp.doctor`` only read the registry and
config, so they answer on every gateway. ``acp.sessions``, ``acp.cancel``,
``acp.close`` and ``acp.doctor`` with ``probe`` need the gateway's
``AcpSessionManager``; without one they fail with ``ACP_UNAVAILABLE``.

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
        result = await asyncio.wait_for(_maybe_await(doctor(agent_id)), PROBE_TIMEOUT_S)
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


@_d.method("acp.cancel")
async def _handle_acp_cancel(params: dict | None, ctx: RpcContext) -> dict[str, Any]:
    p = _params(params)
    key = _session_key(p)
    manager = _require_manager(ctx)
    reason = str(p.get("reason") or "operator")
    try:
        result = await _call_with_optional_kw(manager.cancel, key, reason=reason)
    except KeyError:
        raise
    except Exception as exc:
        raise _map_acp_error(exc) from exc
    log.info("acp.rpc.cancelled", session_key=key)
    return {"session_key": key, "cancelled": True if result is None else bool(result)}


@_d.method("acp.close")
async def _handle_acp_close(params: dict | None, ctx: RpcContext) -> dict[str, Any]:
    p = _params(params)
    key = _session_key(p)
    manager = _require_manager(ctx)
    discard = bool(p.get("discard", False))
    try:
        result = await _call_with_optional_kw(manager.close, key, discard=discard)
    except KeyError:
        raise
    except Exception as exc:
        raise _map_acp_error(exc) from exc
    log.info("acp.rpc.closed", session_key=key, discard=discard)
    return {"session_key": key, "closed": True if result is None else bool(result)}


async def _call_with_optional_kw(fn: Any, key: str, **kwargs: Any) -> Any:
    """Call ``fn(key, **kwargs)`` passing only keywords ``fn`` accepts."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return await _maybe_await(fn(key))
    accepts_var_kw = any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values())
    passed = {k: v for k, v in kwargs.items() if accepts_var_kw or k in sig.parameters}
    return await _maybe_await(fn(key, **passed))
