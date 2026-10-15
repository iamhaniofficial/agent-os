"""ACP session store: JSON records <-> AcpSessionHandle, reuse key, eviction."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from agentos.acp.store import AcpSessionStore, argv_hash, handle_from_dict, handle_to_dict
from agentos.acp.types import AcpSessionHandle


def _handle(record_id: str, **kw) -> AcpSessionHandle:
    argv = kw.pop("argv", ["claude-agent-acp"])
    return AcpSessionHandle(
        record_id=record_id,
        agent_id=kw.pop("agent_id", "claude"),
        cwd=kw.pop("cwd", "/work"),
        argv=argv,
        argv_hash=kw.pop("argv_hash", argv_hash(argv)),
        acp_session_id=kw.pop("acp_session_id", f"sess-{record_id}"),
        **kw,
    )


def test_roundtrip_preserves_every_field(tmp_path: Path) -> None:
    store = AcpSessionStore(tmp_path)
    h = _handle(
        "r1",
        pid=123,
        lease_id="l1",
        mode="oneshot",
        current_mode_id="plan",
        available_modes=[{"id": "plan", "name": "Plan"}],
        config_options=[{"id": "model", "currentValue": "x"}],
        agent_capabilities={"loadSession": True},
        status="idle",
    )
    store.save(h, session_key="agent:main:acp:deadbeef")
    rec = store.load("r1")
    assert rec is not None
    assert rec.handle == h
    assert rec.session_key == "agent:main:acp:deadbeef"
    assert handle_from_dict(handle_to_dict(h)) == h
    # atomic write leaves no temp files behind
    assert sorted(p.name for p in (tmp_path / "sessions").iterdir()) == ["r1.json"]


def test_unknown_keys_are_ignored(tmp_path: Path) -> None:
    store = AcpSessionStore(tmp_path)
    store.save(_handle("r1"))
    path = tmp_path / "sessions" / "r1.json"
    data = json.loads(path.read_text())
    data["handle"]["future_field"] = 1
    data["something_new"] = True
    path.write_text(json.dumps(data))
    assert store.get("r1") is not None
    assert handle_from_dict({**handle_to_dict(_handle("x")), "nope": 1}).record_id == "x"


def test_find_reusable_requires_agent_cwd_and_argv_hash(tmp_path: Path) -> None:
    store = AcpSessionStore(tmp_path)
    h = _handle("r1")
    store.save(h, session_key="k1")
    assert store.find_reusable("claude", "/work", h.argv_hash) is not None
    assert store.find_reusable("codex", "/work", h.argv_hash) is None
    assert store.find_reusable("claude", "/other", h.argv_hash) is None
    assert store.find_reusable("claude", "/work", argv_hash(["claude-agent-acp", "--x"])) is None
    # narrowing by resume id (acp session id or record id) and session key
    assert store.find_reusable("claude", "/work", h.argv_hash, acp_session_id="sess-r1")
    assert store.find_reusable("claude", "/work", h.argv_hash, acp_session_id="r1")
    assert store.find_reusable("claude", "/work", h.argv_hash, acp_session_id="nope") is None
    assert store.find_reusable("claude", "/work", h.argv_hash, session_key="k1")
    assert store.find_reusable("claude", "/work", h.argv_hash, session_key="k2") is None


def test_find_reusable_skips_records_without_session_id(tmp_path: Path) -> None:
    store = AcpSessionStore(tmp_path)
    h = _handle("r1", acp_session_id=None)
    store.save(h)
    assert store.find_reusable("claude", "/work", h.argv_hash) is None


def test_find_reusable_prefers_most_recent(tmp_path: Path) -> None:
    store = AcpSessionStore(tmp_path)
    old = _handle("old", last_used_at=time.time() - 100)
    new = _handle("new", last_used_at=time.time())
    store.save(old)
    store.save(new)
    rec = store.find_reusable("claude", "/work", new.argv_hash)
    assert rec is not None and rec.handle.record_id == "new"
    assert [r.handle.record_id for r in store.list_records()] == ["new", "old"]


def test_mark_closed_and_delete(tmp_path: Path) -> None:
    store = AcpSessionStore(tmp_path)
    store.save(_handle("r1", pid=5, lease_id="l", status="idle"))
    rec = store.mark_closed("r1")
    assert rec is not None and rec.closed and rec.closed_at
    got = store.get("r1")
    assert got is not None and got.status == "closed" and got.pid is None and got.lease_id is None
    assert store.mark_closed("missing") is None
    assert store.delete("r1") is True
    assert store.delete("r1") is False
    assert store.get("r1") is None


def test_eviction_drops_oldest_closed_never_live(tmp_path: Path) -> None:
    store = AcpSessionStore(tmp_path, max_records=3)
    now = time.time()
    store.save(_handle("live1", status="idle", last_used_at=now - 50))
    store.save(_handle("c_old", status="closed", last_used_at=now - 40))
    store.save(_handle("c_new", status="closed", last_used_at=now - 30))
    store.save(_handle("live2", status="idle", last_used_at=now - 20))
    ids = {r.handle.record_id for r in store.list_records()}
    assert ids == {"live1", "c_new", "live2"}
    # with only live records over the cap nothing is evicted
    store.save(_handle("live3", status="busy"))
    store.save(_handle("live4", status="idle"))
    ids = {r.handle.record_id for r in store.list_records()}
    assert {"live1", "live2", "live3", "live4"} <= ids


def test_rejects_path_traversal_ids(tmp_path: Path) -> None:
    store = AcpSessionStore(tmp_path)
    with pytest.raises(ValueError):
        store.path_for("../escape")
    assert store.load("../escape") is None
    assert store.delete("../escape") is False


def test_corrupt_files_are_skipped(tmp_path: Path) -> None:
    store = AcpSessionStore(tmp_path)
    store.save(_handle("ok"))
    (tmp_path / "sessions" / "bad.json").write_text("{not json")
    (tmp_path / "sessions" / "list.json").write_text("[1, 2]")
    assert [r.handle.record_id for r in store.list_records()] == ["ok"]


def test_argv_hash_is_order_sensitive_and_stable() -> None:
    assert argv_hash(["a", "b"]) == argv_hash(["a", "b"])
    assert argv_hash(["a", "b"]) != argv_hash(["b", "a"])
    assert len(argv_hash(["x"])) == 16
