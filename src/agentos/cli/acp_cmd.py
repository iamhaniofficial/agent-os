"""CLI: ``agentos acp`` -- drive external coding harnesses over ACP.

``list`` / ``doctor`` read the harness registry (through the gateway when it
is running, locally otherwise). ``run`` is the manual "telephone game"
path: it starts the harness itself, sends one prompt, streams the reply and
needs no gateway. ``sessions`` / ``cancel`` / ``close`` act on the gateway's
ACP sessions (``sessions`` falls back to the on-disk store).

The runtime (``agentos.acp.runtime``) is imported lazily so this module
loads in builds that do not ship it; commands that need it fail with a
clear "ACP runtime unavailable in this build" error instead.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import secrets
import sys
import time
from pathlib import Path
from typing import Any

import typer
from rich.table import Table

from agentos.acp.inventory import (
    acp_state_dir,
    agents_payload,
    harness_row,
    known_agent_ids,
    read_session_records,
    to_plain,
)
from agentos.acp.registry import normalize_agent_id
from agentos.acp.types import (
    AcpError,
    AcpEvent,
    AcpSessionSpec,
    AcpTurnResult,
    ElicitationRequest,
    ElicitationResponse,
    PermissionDecision,
    PermissionRequest,
)
from agentos.cli.gateway_rpc import default_gateway_token, run_gateway_sync
from agentos.cli.output import emit_error, print_json
from agentos.cli.ui import ACCENT_HEADER, console, markup_escape

app = typer.Typer(help="Drive external coding harnesses (Claude Code, Codex, ...) over ACP.")

PROBE_TIMEOUT_S = 30.0
READ_TOOL_KINDS = frozenset({"read", "search", "fetch", "think"})
PERMISSION_CHOICES = ("approve-reads", "approve-all", "deny-all")
RUNTIME_UNAVAILABLE = "ACP runtime unavailable in this build"
_FALLBACK_CODES = frozenset({"METHOD_NOT_FOUND", "ACP_UNAVAILABLE"})

_NOT_REACHED = object()

# Keywords handed to a runtime factory that takes ``**kwargs``.
_CANONICAL_RUNTIME_KWARGS = frozenset(
    {"config", "permission_bridge", "elicitation_bridge", "state_dir"}
)


# --------------------------------------------------------------------------
# Config / gateway helpers
# --------------------------------------------------------------------------


def _load_config(config_path: Path | None) -> Any:
    from agentos.gateway.config import GatewayConfig

    try:
        return GatewayConfig.load(str(config_path) if config_path else None)
    except Exception as exc:  # noqa: BLE001 - surface the loader error, keep the CLI exit clean
        emit_error(f"could not load config: {exc}", code="CONFIG_ERROR")
        raise typer.Exit(1) from exc


async def _gateway_call(method: str, params: dict[str, Any], config_path: Path | None) -> Any:
    """Call ``method`` on a running gateway; ``_NOT_REACHED`` when there is none.

    A gateway that predates ``acp.*`` (``METHOD_NOT_FOUND``) or runs without
    the ACP manager (``ACP_UNAVAILABLE``) also reads as "not reached" so the
    caller can fall back to local data. Other RPC errors propagate.
    """
    from agentos.cli import gateway_client as gateway_client_module
    from agentos.cli.gateway_rpc import _target_gateway_url

    client = gateway_client_module.GatewayClient()
    try:
        try:
            url = _target_gateway_url(gateway_url=None, config_path=config_path)
            await client.connect(url, token=default_gateway_token(config_path))
        except (SystemExit, ConnectionError, OSError):
            return _NOT_REACHED
        try:
            return await client.call(method, params)
        except gateway_client_module.GatewayRPCError as exc:
            if (exc.code or "").upper() in _FALLBACK_CODES:
                return _NOT_REACHED
            raise
    finally:
        await client.close()


def _gateway_or_local(
    method: str, params: dict[str, Any], *, config_path: Path | None, local: bool, json_output: bool
) -> Any:
    if local:
        return _NOT_REACHED
    from agentos.cli.gateway_client import GatewayRPCError
    from agentos.cli.gateway_rpc import rpc_error_exit_code

    try:
        return asyncio.run(_gateway_call(method, params, config_path))
    except GatewayRPCError as exc:
        emit_error(exc.message, json_output=json_output, code=exc.code, details=exc.data)
        raise typer.Exit(rpc_error_exit_code(exc.code)) from exc


def _unknown_agent(agent: str, config: Any, *, json_output: bool) -> typer.Exit:
    known = ", ".join(known_agent_ids(config))
    emit_error(
        f"unknown ACP agent {agent!r}; known: {known}",
        json_output=json_output,
        code="ACP_AGENT_UNKNOWN",
    )
    return typer.Exit(2)


# --------------------------------------------------------------------------
# list
# --------------------------------------------------------------------------


@app.command("list")
def acp_list(
    config_path: Path | None = typer.Option(None, "--config", help="Config file to read"),
    local: bool = typer.Option(False, "--local", help="Skip the gateway; read config locally"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """List the built-in harnesses and how each one would launch."""
    payload = _gateway_or_local(
        "acp.agents", {}, config_path=config_path, local=local, json_output=json_output
    )
    if payload is _NOT_REACHED or not isinstance(payload, dict):
        payload = agents_payload(_load_config(config_path))
    if json_output:
        print_json(payload)
        return
    rows = [r for r in payload.get("agents", []) if isinstance(r, dict)]
    table = Table(title="ACP harnesses", show_header=True, header_style=ACCENT_HEADER)
    table.add_column("Id")
    table.add_column("Name")
    table.add_column("Launch")
    table.add_column("Missing")
    table.add_column("Override")
    table.add_column("Allowed")
    for row in rows:
        aid = str(row.get("agent_id") or "")
        if row.get("default"):
            aid += " *"
        table.add_row(
            markup_escape(aid),
            markup_escape(str(row.get("name") or "")),
            markup_escape(str(row.get("source") or "")),
            markup_escape(", ".join(row.get("missing_commands") or []) or "-"),
            "yes" if row.get("override") else "",
            "yes" if row.get("allowed", True) else "no",
        )
    console.print(table)
    state = "enabled" if payload.get("enabled", True) else "disabled"
    console.print(
        f"[dim]ACP {state}; default agent {payload.get('default_agent')} (*); "
        f"permission mode {payload.get('permission_mode')}. "
        "`agentos acp doctor <id>` shows the exact argv and fixes.[/dim]"
    )


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------


@app.command("doctor")
def acp_doctor(
    agent: str | None = typer.Argument(None, help="Harness id (default: every harness)"),
    probe: bool = typer.Option(
        False, "--probe", help="Spawn the harness and run ACP initialize (30 s cap)"
    ),
    config_path: Path | None = typer.Option(None, "--config", help="Config file to read"),
    local: bool = typer.Option(False, "--local", help="Skip the gateway; check this machine"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Check that a harness can launch: CLI on PATH, adapter, optional live probe."""
    params: dict[str, Any] = {"probe": probe}
    if agent:
        params["agent"] = normalize_agent_id(agent)
    payload = _gateway_or_local(
        "acp.doctor", params, config_path=config_path, local=local, json_output=json_output
    )
    if payload is _NOT_REACHED or not isinstance(payload, dict):
        payload = _local_doctor(
            agent, probe=probe, config_path=config_path, json_output=json_output
        )
    rows = [r for r in payload.get("agents", []) if isinstance(r, dict)]
    failed = any(not _row_ok(r) for r in rows) if agent else False
    if json_output:
        print_json(payload)
    else:
        _render_doctor(rows)
    if failed:
        raise typer.Exit(1)


def _row_ok(row: dict[str, Any]) -> bool:
    if not row.get("ok", True):
        return False
    probe = row.get("probe")
    return not (isinstance(probe, dict) and probe.get("probe") == "error")


def _local_doctor(
    agent: str | None, *, probe: bool, config_path: Path | None, json_output: bool
) -> dict[str, Any]:
    config = _load_config(config_path)
    ids = [agent] if agent else known_agent_ids(config)
    rows: list[dict[str, Any]] = []
    for aid in ids:
        try:
            rows.append(harness_row(aid, config))
        except KeyError:
            raise _unknown_agent(aid, config, json_output=json_output) from None
    if probe:
        runtime = _build_runtime(config, _StaticPermissionBridge("deny-all", quiet=True))
        if runtime is None:
            emit_error(RUNTIME_UNAVAILABLE, json_output=json_output, code="ACP_UNAVAILABLE")
            raise typer.Exit(1)
        for row in rows:
            row["probe"] = asyncio.run(_probe(runtime, row["agent_id"]))
    return {"agents": rows, "manager": False, "local": True}


async def _probe(runtime: Any, agent_id: str) -> dict[str, Any]:
    try:
        result = await asyncio.wait_for(runtime.doctor(agent_id, probe=True), PROBE_TIMEOUT_S)
    except TimeoutError:
        return {"probe": "error", "error": f"probe timed out after {int(PROBE_TIMEOUT_S)}s"}
    except Exception as exc:  # noqa: BLE001 - a failed probe is a doctor finding
        return {"probe": "error", "error": str(exc), "code": getattr(exc, "code", None)}
    plain = to_plain(result)
    return plain if isinstance(plain, dict) else {"probe": "ok", "result": plain}


def _render_doctor(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        ok = _row_ok(row)
        mark = "[green]ok[/green]" if ok else "[red]problem[/red]"
        name = markup_escape(str(row.get("name") or ""))
        console.print(f"{mark}  [bold]{markup_escape(str(row.get('agent_id')))}[/bold] ({name})")
        argv = " ".join(str(a) for a in row.get("argv") or [])
        console.print(f"    launch: {markup_escape(argv)}  [dim]({row.get('source')})[/dim]")
        for problem in row.get("problems") or []:
            console.print(f"    [yellow]- {markup_escape(str(problem))}[/yellow]")
        probe = row.get("probe")
        if isinstance(probe, dict):
            status = probe.get("probe", "?")
            detail = probe.get("error") or probe.get("version") or ""
            console.print(f"    probe: {status} {markup_escape(str(detail))}".rstrip())


# --------------------------------------------------------------------------
# run (local, no gateway)
# --------------------------------------------------------------------------


class _StaticPermissionBridge:
    """Answers harness permission prompts without a human (CLI ``run`` only).

    ``approve-reads`` allows read/search/fetch/think tool calls and rejects
    everything else; never answers ``allow_always``.
    """

    def __init__(self, mode: str, *, quiet: bool = False) -> None:
        self.mode = mode
        self.quiet = quiet
        self.decisions: list[tuple[str, PermissionDecision]] = []

    async def decide(self, request: PermissionRequest) -> PermissionDecision:
        kind = (request.tool_kind or "other").lower()
        decision: PermissionDecision
        if self.mode == "approve-all":
            decision = "allow_once"
        elif self.mode == "deny-all":
            decision = "reject_once"
        else:
            decision = "allow_once" if kind in READ_TOOL_KINDS else "reject_once"
        self.decisions.append((request.title, decision))
        if not self.quiet:
            verb = "allowed" if decision == "allow_once" else "denied"
            typer.echo(f"[permission {verb}] {kind}: {request.title}", err=True)
        return decision


class _DecliningElicitationBridge:
    """Nobody can answer a harness question in a one-shot CLI turn."""

    def __init__(self, *, quiet: bool = False) -> None:
        self.quiet = quiet

    async def ask(self, request: ElicitationRequest) -> ElicitationResponse:
        if not self.quiet:
            typer.echo(f"[question declined] {request.message}", err=True)
        return ElicitationResponse(action="decline")


def _runtime_factory() -> Any:
    """Return the runtime class/factory, or ``None`` when this build lacks it."""
    try:
        mod = importlib.import_module("agentos.acp.runtime")
    except ImportError:
        return None
    for name in ("build_runtime", "create_runtime", "AcpRuntimeImpl"):
        factory = getattr(mod, name, None)
        if callable(factory):
            return factory
    return None


def _build_runtime(config: Any, bridge: Any, elicitation: Any | None = None) -> Any:
    factory = _runtime_factory()
    if factory is None:
        return None
    acp_config = getattr(config, "acp", None)
    candidates: dict[str, Any] = {
        "config": acp_config,
        "acp_config": acp_config,
        "gateway_config": config,
        "permission_bridge": bridge,
        "bridge": bridge,
        "elicitation_bridge": elicitation or _DecliningElicitationBridge(quiet=True),
        "state_dir": acp_state_dir(config),
    }
    try:
        sig = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory()
    names = {
        p.name
        for p in sig.parameters.values()
        if p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD) and p.name != "self"
    }
    if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
        names |= _CANONICAL_RUNTIME_KWARGS
    return factory(**{k: v for k, v in candidates.items() if k in names})


def _format_duration(ms: int) -> str:
    seconds = max(0, ms) / 1000
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def stats_line(agent_id: str, result: AcpTurnResult) -> str:
    usage = result.usage
    parts = [f"{agent_id} {result.stop_reason}", f"runtime {_format_duration(result.duration_ms)}"]
    total = usage.total_tokens or (usage.input_tokens + usage.output_tokens)
    if total:
        parts.append(f"tokens {total} (in {usage.input_tokens} / out {usage.output_tokens})")
    if usage.cost_usd is not None:
        parts.append(f"cost ${usage.cost_usd:.4f}")
    if result.tool_calls:
        parts.append(f"tools {result.tool_calls}")
    return "Stats: " + " • ".join(parts)


def _quiet_logs() -> None:
    """Keep info-level harness logs out of interactive output.

    ``agentos acp run`` streams the reply on stdout while the runtime logs on
    stderr; in a terminal the two interleave. Unless the operator asked for a
    level explicitly (``AGENTOS_LOG_LEVEL``), only warnings and above show.
    """
    import logging
    import os

    if os.environ.get("AGENTOS_LOG_LEVEL"):
        return
    import structlog

    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))


def _render_event(event: AcpEvent, *, verbose: bool) -> None:
    if event.kind == "text_delta" and event.text:
        sys.stdout.write(event.text)
        sys.stdout.flush()
    elif event.kind == "tool_call":
        typer.echo(f"\n[tool] {event.tool_kind or 'other'}: {event.title or ''}", err=True)
    elif event.kind == "tool_call_update" and event.status in {"completed", "failed"}:
        if verbose or event.status == "failed":
            typer.echo(f"[tool {event.status}] {event.title or event.tool_call_id or ''}", err=True)
    elif event.kind == "thought_delta" and verbose and event.text:
        typer.echo(event.text, err=True, nl=False)
    elif event.kind == "error":
        typer.echo(f"\n[error] {event.error or event.text}", err=True)


async def _run_turn(
    runtime: Any,
    spec: AcpSessionSpec,
    prompt: str,
    *,
    session_key: str,
    discard: bool,
    json_output: bool,
    verbose: bool,
) -> tuple[AcpTurnResult, Any]:
    handle = await runtime.ensure_session(spec, session_key=session_key)
    turn = None
    started = time.monotonic()
    try:
        turn = await runtime.start_turn(handle, text=prompt, session_key=session_key)
        async for event in turn.events:
            if not json_output:
                _render_event(event, verbose=verbose)
        result: AcpTurnResult = await turn.result
        if not result.duration_ms:
            result.duration_ms = int((time.monotonic() - started) * 1000)
        return result, handle
    except asyncio.CancelledError:
        if turn is not None:
            await turn.cancel()
        raise
    finally:
        try:
            await runtime.close(handle, discard=discard)
        except Exception:  # noqa: BLE001 - the turn's outcome matters more than close noise
            pass


@app.command("run")
def acp_run(
    agent: str = typer.Argument(..., help="Harness id, e.g. claude, codex, gemini"),
    prompt: str = typer.Argument(..., help="Prompt to send"),
    cwd: Path | None = typer.Option(None, "--cwd", help="Working directory (default: current)"),
    mode: str | None = typer.Option(
        None, "--mode", help="Harness session mode id (e.g. read-only, acceptEdits)"
    ),
    oneshot: bool = typer.Option(
        False, "--oneshot", help="Discard the session record after the turn (no resume)"
    ),
    resume: str | None = typer.Option(
        None, "--resume", help="ACP session id to resume (printed after an earlier run)"
    ),
    model: str | None = typer.Option(None, "--model", help="Model id, if the harness offers one"),
    permissions: str = typer.Option(
        "approve-reads",
        "--permissions",
        help="approve-reads (default) | approve-all | deny-all",
    ),
    timeout: float | None = typer.Option(
        None, "--timeout", help="Turn wall-clock limit in seconds (default: [acp] turn_timeout_s)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Also show thoughts and tool ends"),
    config_path: Path | None = typer.Option(None, "--config", help="Config file to read"),
    json_output: bool = typer.Option(False, "--json", help="Emit one JSON result, no live text"),
) -> None:
    """Run one prompt through a harness on this machine and stream the reply.

    No gateway is needed. Permission prompts are answered by a static policy:
    reads are allowed and everything else is denied unless --permissions says
    otherwise. Harness questions are declined.
    """
    if permissions not in PERMISSION_CHOICES:
        raise typer.BadParameter(
            f"--permissions must be one of {', '.join(PERMISSION_CHOICES)}",
            param_hint="--permissions",
        )
    config = _load_config(config_path)
    acp_cfg = getattr(config, "acp", None)
    if acp_cfg is not None and not acp_cfg.enabled:
        emit_error(
            "ACP is disabled ([acp] enabled = false)", json_output=json_output, code="ACP_DISABLED"
        )
        raise typer.Exit(1)
    aid = normalize_agent_id(agent)
    try:
        row = harness_row(aid, config)
    except KeyError:
        raise _unknown_agent(agent, config, json_output=json_output) from None
    if row["missing_commands"]:
        emit_error(
            "; ".join(row["problems"]) or "harness CLI missing",
            json_output=json_output,
            code="ACP_HARNESS_MISSING",
            details={"missing": row["missing_commands"]},
        )
        raise typer.Exit(1)

    bridge = _StaticPermissionBridge(permissions, quiet=json_output)
    runtime = _build_runtime(config, bridge, _DecliningElicitationBridge(quiet=json_output))
    if runtime is None:
        emit_error(RUNTIME_UNAVAILABLE, json_output=json_output, code="ACP_UNAVAILABLE")
        raise typer.Exit(1)

    workdir = (cwd or Path.cwd()).expanduser().resolve()
    if not workdir.is_dir():
        raise typer.BadParameter(f"{workdir} is not a directory", param_hint="--cwd")
    if not verbose and not json_output:
        _quiet_logs()
    turn_timeout = (
        float(timeout) if timeout is not None else float(acp_cfg.turn_timeout_s if acp_cfg else 0)
    )
    spec = AcpSessionSpec(
        agent_id=aid,
        cwd=str(workdir),
        mode="oneshot" if oneshot else "persistent",
        resume_session_id=resume,
        model=model,
        harness_mode=mode,
        label="agentos acp run",
        turn_timeout_s=turn_timeout,
    )
    session_key = f"agent:cli:acp:{secrets.token_hex(4)}"
    if not json_output:
        typer.echo(f"[acp] {aid} in {workdir} (permissions: {permissions})", err=True)
    try:
        result, handle = asyncio.run(
            _run_turn(
                runtime,
                spec,
                prompt,
                session_key=session_key,
                discard=oneshot,
                json_output=json_output,
                verbose=verbose,
            )
        )
    except KeyboardInterrupt:
        emit_error("cancelled", json_output=json_output, code="ACP_CANCELLED")
        raise typer.Exit(130) from None
    except AcpError as exc:
        emit_error(str(exc), json_output=json_output, code=exc.code, details=exc.details or None)
        raise typer.Exit(1) from exc

    if json_output:
        print_json(
            {
                "agent_id": aid,
                "session_key": session_key,
                "record_id": getattr(handle, "record_id", None),
                "acp_session_id": getattr(handle, "acp_session_id", None),
                "cwd": str(workdir),
                "permissions": permissions,
                "result": to_plain(result),
            }
        )
    else:
        if result.text and not result.text.endswith("\n"):
            sys.stdout.write("\n")
            sys.stdout.flush()
        if result.error:
            typer.echo(f"[error] {result.error}", err=True)
        typer.echo(stats_line(aid, result), err=True)
        acp_session_id = getattr(handle, "acp_session_id", None)
        if acp_session_id and not oneshot:
            typer.echo(f"[acp] resume with --resume {acp_session_id}", err=True)
    if result.stop_reason == "cancelled":
        raise typer.Exit(130)
    if result.stop_reason in {"error", "refusal"} or result.error:
        raise typer.Exit(1)


# --------------------------------------------------------------------------
# sessions / cancel / close
# --------------------------------------------------------------------------


@app.command("sessions")
def acp_sessions(
    config_path: Path | None = typer.Option(None, "--config", help="Config file to read"),
    local: bool = typer.Option(False, "--local", help="Skip the gateway; read the session store"),
    limit: int = typer.Option(50, "--limit", "-n", help="Maximum rows"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """List ACP harness sessions (live from the gateway, else the stored records)."""
    payload = _gateway_or_local(
        "acp.sessions", {}, config_path=config_path, local=local, json_output=json_output
    )
    source = "gateway"
    if payload is _NOT_REACHED or not isinstance(payload, dict):
        config = _load_config(config_path)
        rows = read_session_records(acp_state_dir(config))
        payload = {"sessions": rows, "count": len(rows)}
        source = "store"
    sessions = [r for r in payload.get("sessions", []) if isinstance(r, dict)][: max(0, limit)]
    if json_output:
        print_json({**payload, "sessions": sessions, "count": len(sessions), "source": source})
        return
    table = Table(title=f"ACP sessions ({source})", show_header=True, header_style=ACCENT_HEADER)
    table.add_column("Session key / record")
    table.add_column("Agent")
    table.add_column("Status")
    table.add_column("Mode")
    table.add_column("Cwd")
    for row in sessions:
        key = row.get("session_key") or row.get("key") or row.get("record_id") or ""
        table.add_row(
            markup_escape(str(key)),
            markup_escape(str(row.get("agent_id") or "")),
            markup_escape(str(row.get("status") or "")),
            markup_escape(str(row.get("mode") or "")),
            markup_escape(str(row.get("cwd") or "")),
        )
    console.print(table)
    if source == "store":
        console.print("[dim]Gateway not reachable; showing stored session records.[/dim]")


@app.command("cancel")
def acp_cancel(
    session_key: str = typer.Argument(..., help="ACP session key (agent:<id>:acp:<hex>)"),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Cancel the running turn of a gateway ACP session (the session stays open)."""

    async def _run(client: Any) -> Any:
        return await client.call("acp.cancel", {"session_key": session_key})

    payload = run_gateway_sync(_run, json_output=json_output)
    if json_output:
        print_json(payload)
        return
    console.print(f"Cancel sent to {markup_escape(session_key)}")


@app.command("close")
def acp_close(
    session_key: str = typer.Argument(..., help="ACP session key (agent:<id>:acp:<hex>)"),
    discard: bool = typer.Option(
        False, "--discard", help="Also delete the resume record for this session"
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON"),
) -> None:
    """Close a gateway ACP session and stop its harness process."""

    async def _run(client: Any) -> Any:
        return await client.call("acp.close", {"session_key": session_key, "discard": discard})

    payload = run_gateway_sync(_run, json_output=json_output)
    if json_output:
        print_json(payload)
        return
    console.print(f"Closed {markup_escape(session_key)}")
