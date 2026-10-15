"""``acp`` tool: inspect and control ACP coding-harness sessions.

Not exposed by default (operator-ish surface). Spawning goes through
``sessions_spawn(runtime="acp")``; this tool covers everything around it:
which harnesses exist, live sessions, status, cancel / close, harness mode
and config options, and a doctor probe. Output is JSON.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

import structlog

from agentos.tools.registry import tool
from agentos.tools.types import ToolError, current_tool_context

log = structlog.get_logger("agentos.tools.acp")

_ACTIONS = ("agents", "sessions", "status", "cancel", "close", "set_mode", "set_model", "doctor")
_DOCTOR_TIMEOUT_S = 30.0


def _gateway_config() -> Any:
    from agentos.tools.builtin import sessions as sessions_tool

    return sessions_tool._gateway_config


def _manager() -> Any:
    from agentos.acp.manager import get_acp_manager

    manager = get_acp_manager()
    if manager is None:
        raise ToolError("ACP unavailable: not configured on this gateway")
    return manager


def _origin(row: Any) -> dict[str, Any]:
    value = row.get("origin") if isinstance(row, Mapping) else getattr(row, "origin", None)
    return dict(value) if isinstance(value, Mapping) else {}


async def _authorized_session(session_key: str | None) -> Any:
    """The ACP session row, if the caller may control it.

    An agent may control ACP sessions it spawned or that belong to its own
    agent id; a caller without tool context (operator surfaces) may control
    any.
    """
    if not session_key:
        raise ToolError("'session_key' is required for this action")
    from agentos.session.keys import parse_agent_id
    from agentos.tools.builtin import sessions as sessions_tool

    mgr = sessions_tool._get_session_manager()
    row = await mgr.get_session(session_key)
    if row is None:
        raise ToolError(f"Session not found: {session_key}")
    origin = _origin(row)
    if origin.get("kind") != "acp":
        raise ToolError(f"Session '{session_key}' is not an ACP session")
    ctx = current_tool_context.get()
    if ctx is not None and ctx.session_key:
        spawned_by = getattr(row, "spawned_by", None)
        owner = parse_agent_id(session_key)
        if spawned_by != ctx.session_key and owner != (ctx.agent_id or ""):
            raise ToolError("ACP session belongs to another agent")
    return row


def _agents_payload() -> dict[str, Any]:
    from agentos.acp.registry import list_harnesses, resolve_launch
    from agentos.acp.spawn import _agent_allowed, acp_config_of, launch_overrides

    acp_cfg = acp_config_of(_gateway_config())
    overrides = launch_overrides(acp_cfg)
    default = getattr(acp_cfg, "default_agent", "")
    agents: list[dict[str, Any]] = []
    for harness in list_harnesses():
        launch = resolve_launch(harness.agent_id, overrides=overrides)
        agents.append(
            {
                "agent_id": harness.agent_id,
                "name": harness.name,
                "allowed": _agent_allowed(acp_cfg, harness.agent_id),
                "default": harness.agent_id == default,
                "launch_source": launch.source,
                "argv": launch.argv,
                "missing_commands": launch.missing_commands,
                "ready": not launch.missing_commands,
                "notes": harness.notes or None,
            }
        )
    known = {a["agent_id"] for a in agents}
    for agent_id, argv in overrides.items():
        if agent_id in known:
            continue
        agents.append(
            {
                "agent_id": agent_id,
                "name": agent_id,
                "allowed": _agent_allowed(acp_cfg, agent_id),
                "default": agent_id == default,
                "launch_source": "override",
                "argv": argv,
                "missing_commands": [],
                "ready": True,
                "notes": None,
            }
        )
    return {
        "action": "agents",
        "enabled": bool(getattr(acp_cfg, "enabled", True)),
        "default_agent": default,
        "permission_mode": getattr(acp_cfg, "permission_mode", None),
        "agents": agents,
    }


async def _sessions_payload() -> dict[str, Any]:
    from agentos.tools.builtin import sessions as sessions_tool

    manager = _manager()
    live = manager.list_sessions()
    ctx = current_tool_context.get()
    children: list[dict[str, Any]] = []
    if ctx is not None and ctx.session_key:
        from agentos.acp.spawn import _list_children, row_is_acp

        live_keys = {s["session_key"] for s in live}
        try:
            mgr = sessions_tool._get_session_manager()
            rows = await _list_children(mgr, ctx.session_key)
        except ToolError:
            rows = []
        for row in rows:
            if not row_is_acp(row):
                continue
            key = row.get("session_key") if isinstance(row, Mapping) else None
            origin = _origin(row)
            children.append(
                {
                    "session_key": key,
                    "status": row.get("status") if isinstance(row, Mapping) else None,
                    "acp_agent": origin.get("agent_id"),
                    "mode": origin.get("mode"),
                    "cwd": origin.get("cwd"),
                    "closed": bool(origin.get("closed")),
                    "live": key in live_keys,
                }
            )
        owner = ctx.agent_id or ""
        live = [s for s in live if str(s.get("session_key", "")).startswith(f"agent:{owner}:")]
    return {
        "action": "sessions",
        "available": manager.available,
        "live_count": manager.live_count,
        "max_concurrent": manager.max_concurrent,
        "live": live,
        "children": children,
    }


@tool(
    name="acp",
    description=(
        "Inspect and control ACP coding-harness sessions (Claude Code, Codex, Gemini CLI, ...). "
        "Actions: agents (registry + readiness), sessions (live + your children), status, "
        "cancel (running turn), close (end the harness), set_mode (harness mode id), "
        "set_model (model, or option=thinking|permissions|<id>), doctor (probe a harness). "
        "Spawn with sessions_spawn(runtime='acp')."
    ),
    params={
        "action": {"type": "string", "enum": list(_ACTIONS), "description": "What to do"},
        "session_key": {
            "type": "string",
            "description": "ACP session key (status, cancel, close, set_mode, set_model)",
        },
        "agent": {"type": "string", "description": "Harness id for doctor (e.g. claude, codex)"},
        "value": {
            "type": "string",
            "description": "Mode id for set_mode; option value for set_model",
        },
        "option": {
            "type": "string",
            "description": "set_model only: model (default), thinking, permissions, or a raw "
            "config option id",
        },
    },
    required=["action"],
    exposed_by_default=False,
)
async def acp(
    action: str,
    session_key: str | None = None,
    agent: str | None = None,
    value: str | None = None,
    option: str | None = None,
) -> str:
    from agentos.acp.types import AcpError

    if action not in _ACTIONS:
        raise ToolError(f"Invalid action: {action}. Must be one of {'|'.join(_ACTIONS)}")
    try:
        if action == "agents":
            return json.dumps(_agents_payload())
        if action == "sessions":
            return json.dumps(await _sessions_payload())
        if action == "doctor":
            from agentos.acp.registry import normalize_agent_id
            from agentos.acp.spawn import acp_config_of

            target = normalize_agent_id(
                str(agent or getattr(acp_config_of(_gateway_config()), "default_agent", "") or "")
            )
            if not target:
                raise ToolError("'agent' is required for doctor")
            runtime = _manager().runtime
            if runtime is None:
                raise ToolError(f"ACP unavailable: {_manager().unavailable_reason}")
            try:
                report = await asyncio.wait_for(runtime.doctor(target), timeout=_DOCTOR_TIMEOUT_S)
            except TimeoutError:
                report = {"agent_id": target, "probe": "error", "error": "doctor timed out (30s)"}
            return json.dumps({"action": "doctor", **dict(report)}, default=str)

        row = await _authorized_session(session_key)
        session_key = str(session_key)
        manager = _manager()
        if action == "status":
            info = await manager.status(session_key)
            info["session_status"] = getattr(row, "status", None)
            return json.dumps({"action": "status", **info}, default=str)
        if action == "cancel":
            cancelled_tasks = 0
            try:
                from agentos.tools.builtin import sessions as sessions_tool

                cancelled_tasks = await sessions_tool._get_task_runtime().cancel(
                    session_key=session_key, source="acp_tool_cancel", reason="acp_tool_cancel"
                )
            except ToolError:
                pass
            was_running = await manager.cancel(session_key, reason="acp_tool_cancel")
            return json.dumps(
                {
                    "action": "cancel",
                    "session_key": session_key,
                    "cancelled": bool(was_running or cancelled_tasks),
                }
            )
        if action == "close":
            try:
                from agentos.tools.builtin import sessions as sessions_tool

                await sessions_tool._get_task_runtime().cancel(
                    session_key=session_key, source="acp_tool_close", reason="acp_tool_close"
                )
            except ToolError:
                pass
            had_live = await manager.close(session_key, reason="acp_tool_close")
            return json.dumps(
                {
                    "action": "close",
                    "session_key": session_key,
                    "closed": True,
                    "was_live": had_live,
                }
            )
        if action == "set_mode":
            if not value:
                raise ToolError("'value' (mode id) is required for set_mode")
            return json.dumps({"action": "set_mode", **await manager.set_mode(session_key, value)})
        # set_model
        if not value:
            raise ToolError("'value' is required for set_model")
        result = await manager.set_option(session_key, (option or "model").strip(), value)
        return json.dumps({"action": "set_model", **result}, default=str)
    except AcpError as exc:
        raise ToolError(f"{exc.code}: {exc}") from exc
