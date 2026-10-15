"""Persisted ACP session records (resume handles).

One JSON file per record under ``<state>/acp/sessions/<record_id>.json``
holding ``dataclasses.asdict(AcpSessionHandle)`` plus a few bookkeeping keys
(``session_key``, ``closed_at``, ``schema``). Writes are atomic. A record is
reused only when ``agent_id``, ``cwd`` and ``argv_hash`` all match (OpenClaw
rule), so a changed adapter version or launch override never resumes a
session the new binary cannot read. At most ``max_records`` records are
kept; the oldest *closed* ones go first, live ones are never evicted.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from agentos.acp.process import atomic_write_json
from agentos.acp.types import AcpSessionHandle

SCHEMA_VERSION = 1
DEFAULT_MAX_RECORDS = 4096
_HANDLE_FIELDS = frozenset(f.name for f in dataclasses.fields(AcpSessionHandle))
_RECORD_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def argv_hash(argv: list[str]) -> str:
    """Stable short hash of a launch argv (part of the reuse key)."""
    blob = json.dumps(list(argv), separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def handle_to_dict(handle: AcpSessionHandle) -> dict[str, Any]:
    return dataclasses.asdict(handle)


def handle_from_dict(data: dict[str, Any]) -> AcpSessionHandle:
    """Rebuild a handle, ignoring unknown keys (forward compatible)."""
    kwargs = {k: v for k, v in data.items() if k in _HANDLE_FIELDS}
    for list_key in ("argv", "available_modes", "config_options"):
        if kwargs.get(list_key) is None:
            kwargs.pop(list_key, None)
    if kwargs.get("agent_capabilities") is None:
        kwargs.pop("agent_capabilities", None)
    return AcpSessionHandle(**kwargs)


@dataclasses.dataclass(slots=True)
class SessionRecord:
    handle: AcpSessionHandle
    session_key: str = ""
    closed_at: float | None = None
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def closed(self) -> bool:
        return self.handle.status == "closed" or self.closed_at is not None


class AcpSessionStore:
    """JSON-file store for :class:`AcpSessionHandle` records."""

    def __init__(self, state_dir: Path, *, max_records: int = DEFAULT_MAX_RECORDS) -> None:
        self.root = Path(state_dir) / "sessions"
        self.max_records = max(1, int(max_records))

    # -- paths ---------------------------------------------------------

    def path_for(self, record_id: str) -> Path:
        if not _RECORD_ID_RE.match(record_id or ""):
            raise ValueError(f"invalid ACP record id: {record_id!r}")
        return self.root / f"{record_id}.json"

    # -- read ----------------------------------------------------------

    def _read(self, path: Path) -> SessionRecord | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        nested = data.get("handle")
        raw_handle: dict[str, Any] = nested if isinstance(nested, dict) else data
        try:
            handle = handle_from_dict(dict(raw_handle))
        except TypeError:
            return None
        closed_at = data.get("closed_at")
        return SessionRecord(
            handle=handle,
            session_key=str(data.get("session_key") or ""),
            closed_at=float(closed_at) if isinstance(closed_at, int | float) else None,
            extra=dict(data.get("extra") or {}),
        )

    def load(self, record_id: str) -> SessionRecord | None:
        try:
            path = self.path_for(record_id)
        except ValueError:
            return None
        return self._read(path)

    def get(self, record_id: str) -> AcpSessionHandle | None:
        rec = self.load(record_id)
        return rec.handle if rec else None

    def list_records(self) -> list[SessionRecord]:
        """All records, most recently used first."""
        if not self.root.is_dir():
            return []
        out: list[SessionRecord] = []
        for p in self.root.glob("*.json"):
            rec = self._read(p)
            if rec is not None:
                out.append(rec)
        out.sort(key=lambda r: r.handle.last_used_at, reverse=True)
        return out

    def find_reusable(
        self,
        agent_id: str,
        cwd: str,
        argv_hash: str,
        *,
        acp_session_id: str | None = None,
        session_key: str | None = None,
    ) -> SessionRecord | None:
        """Newest record matching the reuse key that has a harness session id.

        ``acp_session_id`` / ``session_key`` narrow the match further (the
        runtime passes the resume id it was asked for).
        """
        for rec in self.list_records():
            h = rec.handle
            if h.agent_id != agent_id or h.cwd != cwd or h.argv_hash != argv_hash:
                continue
            if not h.acp_session_id:
                continue
            if acp_session_id and acp_session_id not in (h.acp_session_id, h.record_id):
                continue
            if session_key and rec.session_key and rec.session_key != session_key:
                continue
            return rec
        return None

    # -- write ---------------------------------------------------------

    def save(
        self,
        handle: AcpSessionHandle,
        *,
        session_key: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> SessionRecord:
        prev = self.load(handle.record_id)
        rec = SessionRecord(
            handle=handle,
            session_key=session_key
            if session_key is not None
            else (prev.session_key if prev else ""),
            closed_at=(time.time() if handle.status == "closed" else None),
            extra={**(prev.extra if prev else {}), **(extra or {})},
        )
        if prev and prev.closed_at and handle.status == "closed":
            rec.closed_at = prev.closed_at
        atomic_write_json(
            self.path_for(handle.record_id),
            {
                "schema": SCHEMA_VERSION,
                "session_key": rec.session_key,
                "closed_at": rec.closed_at,
                "extra": rec.extra,
                "handle": handle_to_dict(handle),
            },
        )
        if prev is None:
            self._evict()
        return rec

    def mark_closed(self, record_id: str) -> SessionRecord | None:
        rec = self.load(record_id)
        if rec is None:
            return None
        rec.handle.status = "closed"
        rec.handle.pid = None
        rec.handle.lease_id = None
        return self.save(rec.handle)

    def delete(self, record_id: str) -> bool:
        try:
            path = self.path_for(record_id)
        except ValueError:
            return False
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False

    def _evict(self) -> None:
        if not self.root.is_dir():
            return
        records = self.list_records()
        overflow = len(records) - self.max_records
        if overflow <= 0:
            return
        closed = sorted(
            (r for r in records if r.closed),
            key=lambda r: r.closed_at or r.handle.last_used_at,
        )
        for rec in closed[:overflow]:
            with contextlib.suppress(OSError):
                self.path_for(rec.handle.record_id).unlink()
