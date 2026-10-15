"""Read-only views over the ACP harness registry, config and session store.

Shared by the ``agentos acp`` CLI and the ``acp.*`` gateway RPC so both
surfaces report the same rows. Nothing here spawns a process: probing a
harness (spawn + ``initialize``) is the runtime's job (``AcpRuntime.doctor``).
"""

from __future__ import annotations

import importlib
import json
import shutil
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from agentos.acp.registry import (
    HarnessDefinition,
    get_harness,
    list_harnesses,
    normalize_agent_id,
    resolve_launch,
)

Which = Callable[[str], str | None]

# Launchers that download an adapter package on first use.
_PACKAGE_LAUNCHERS = frozenset({"npx", "uvx", "bunx", "pnpx"})


def _acp_config(config: Any) -> Any:
    return getattr(config, "acp", None) if config is not None else None


def harness_overrides(config: Any) -> dict[str, list[str]]:
    """``[acp.agents.<id>]`` overrides that replace the registry argv.

    Only entries with a ``command`` replace the launch; an entry that only
    adds ``env`` keeps the registry argv.
    """
    acp = _acp_config(config)
    agents = getattr(acp, "agents", None) or {}
    out: dict[str, list[str]] = {}
    for raw_id, override in agents.items():
        command = getattr(override, "command", None)
        if not command:
            continue
        args = list(getattr(override, "args", None) or [])
        out[normalize_agent_id(str(raw_id))] = [str(command), *map(str, args)]
    return out


def is_agent_allowed(config: Any, agent_id: str) -> bool:
    acp = _acp_config(config)
    allowed = list(getattr(acp, "allowed_agents", None) or ["*"])
    if "*" in allowed:
        return True
    wanted = normalize_agent_id(agent_id)
    return any(normalize_agent_id(str(a)) == wanted for a in allowed)


def acp_state_dir(config: Any) -> Path:
    """``[acp] state_dir`` > ``<config.state_dir>/acp`` > ``~/.agentos/state/acp``."""
    acp = _acp_config(config)
    explicit = getattr(acp, "state_dir", None)
    if isinstance(explicit, str) and explicit.strip():
        return Path(explicit.strip()).expanduser()
    base = getattr(config, "state_dir", None) if config is not None else None
    if isinstance(base, str) and base.strip():
        return Path(base.strip()).expanduser() / "acp"
    from agentos.paths import state_dir

    return state_dir("acp")


def known_agent_ids(config: Any = None) -> list[str]:
    ids = [h.agent_id for h in list_harnesses()]
    for extra in harness_overrides(config):
        if extra not in ids:
            ids.append(extra)
    return ids


def harness_row(agent_id: str, config: Any = None, *, which: Which | None = None) -> dict[str, Any]:
    """One registry/doctor row for ``agent_id`` (no process is started).

    Raises ``KeyError`` for an id that is neither built in nor overridden.
    """
    which = which or shutil.which
    aid = normalize_agent_id(agent_id)
    overrides = harness_overrides(config)
    definition: HarnessDefinition | None = get_harness(aid)
    launch = resolve_launch(aid, overrides=overrides, which=which)
    acp = _acp_config(config)
    install_missing = bool(getattr(acp, "install_missing_adapters", True))
    launcher = launch.argv[0] if launch.argv else ""
    launcher_found = bool(launcher) and which(launcher) is not None
    if launch.source == "installed":
        adapter_installed = True
    elif launch.source == "override" or (definition is not None and definition.installed_argv):
        adapter_installed = launch.source == "override" and launcher_found
    else:
        # No adapter package: the harness CLI speaks ACP itself.
        adapter_installed = launcher_found and launcher not in _PACKAGE_LAUNCHERS

    problems: list[str] = []
    for cmd in launch.missing_commands:
        problems.append(f"`{cmd}` is not on PATH; install the {_name(definition, aid)} CLI")
    if not launcher_found:
        problems.append(f"launcher `{launcher}` is not on PATH")
    elif launch.source == "package-exec" and launcher in _PACKAGE_LAUNCHERS and not install_missing:
        problems.append(
            "adapter is not installed and [acp] install_missing_adapters = false; "
            f"install `{definition.preferred_bin if definition else launcher}` or allow downloads"
        )

    return {
        "agent_id": aid,
        "name": _name(definition, aid),
        "argv": list(launch.argv),
        "source": launch.source,
        "missing_commands": list(launch.missing_commands),
        "launcher_found": launcher_found,
        "adapter_installed": adapter_installed,
        "override": aid in overrides,
        "allowed": is_agent_allowed(config, aid),
        "default": normalize_agent_id(str(getattr(acp, "default_agent", "claude") or "")) == aid,
        "package": definition.package_name if definition else None,
        "notes": definition.notes if definition else "",
        "ok": not problems,
        "problems": problems,
    }


def _name(definition: HarnessDefinition | None, aid: str) -> str:
    return definition.name if definition is not None else aid


def harness_rows(config: Any = None, *, which: Which | None = None) -> list[dict[str, Any]]:
    return [harness_row(aid, config, which=which) for aid in known_agent_ids(config)]


def agents_payload(config: Any = None, *, which: Which | None = None) -> dict[str, Any]:
    acp = _acp_config(config)
    return {
        "enabled": bool(getattr(acp, "enabled", True)),
        "default_agent": str(getattr(acp, "default_agent", "claude") or "claude"),
        "permission_mode": str(getattr(acp, "permission_mode", "ask") or "ask"),
        "agents": harness_rows(config, which=which),
    }


# --------------------------------------------------------------------------
# Session records
# --------------------------------------------------------------------------


def to_plain(value: Any) -> Any:
    """Dataclass / pydantic / mapping -> JSON-safe dict (best effort)."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump()
    if isinstance(value, dict):
        return dict(value)
    return value


def read_session_records(state_dir: Path) -> list[dict[str, Any]]:
    """Read persisted session records, newest first.

    Prefers ``agentos.acp.store`` when it offers a ``list_records`` API and
    falls back to the documented on-disk layout (``sessions/<record_id>.json``)
    so the CLI keeps working against any store build.
    """
    rows = _records_via_store(state_dir)
    if rows is None:
        rows = _records_from_disk(state_dir)
    rows.sort(key=lambda r: float(r.get("last_used_at") or r.get("created_at") or 0), reverse=True)
    return rows


def _records_via_store(state_dir: Path) -> list[dict[str, Any]] | None:
    try:
        store_mod = importlib.import_module("agentos.acp.store")
    except ImportError:
        return None
    candidates: list[Any] = []
    for name in ("AcpSessionStore", "SessionStore", "AcpStore"):
        cls = getattr(store_mod, name, None)
        if cls is None:
            continue
        try:
            candidates.append(cls(state_dir))
        except Exception:  # noqa: BLE001 - unknown constructor; try the next shape
            continue
    for obj in candidates:
        lister = getattr(obj, "list_records", None)
        if not callable(lister):
            continue
        try:
            records = lister()
        except Exception:  # noqa: BLE001 - fall back to the raw layout
            return None
        if hasattr(records, "__await__"):
            # Async store APIs are read by the raw fallback instead.
            close = getattr(records, "close", None)
            if callable(close):
                close()
            return None
        return [r for r in (to_plain(rec) for rec in records) if isinstance(r, dict)]
    return None


def _records_from_disk(state_dir: Path) -> list[dict[str, Any]]:
    folder = state_dir / "sessions"
    if not folder.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in folder.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict):
            payload.setdefault("record_id", path.stem)
            rows.append(payload)
    return rows
