"""Tests for GatewayPidLock: PID file placement (AC-C1) and lock lifetime (#2119)."""

from __future__ import annotations

import json
import os
import signal
from pathlib import Path

import pytest

from agentos.gateway import pidlock
from agentos.gateway.pidlock import GatewayPidLock


def test_pid_file_in_state_dir_not_parent(tmp_path: Path) -> None:
    """AC-C1-1/AC-C1-2: PID file must land in state_dir, not state_dir.parent."""
    state_dir = tmp_path / "state"
    lock = GatewayPidLock(state_dir)
    lock.acquire()
    try:
        # PID file must be inside state_dir
        assert (state_dir / "gateway.pid").exists(), f"gateway.pid not found in {state_dir}"
        # PID file must NOT be in the parent directory
        assert not (tmp_path / "gateway.pid").exists(), (
            f"gateway.pid incorrectly written to parent {tmp_path}"
        )
    finally:
        lock.release()


def test_release_removes_pid_file_but_keeps_lock_anchor(tmp_path: Path) -> None:
    lock = GatewayPidLock(tmp_path)
    lock.acquire()
    lock.release()

    assert not (tmp_path / "gateway.pid").exists()
    assert (tmp_path / "gateway.pid.lock").exists()

    # Idempotent.
    lock.release()
    assert (tmp_path / "gateway.pid.lock").exists()


def test_second_gateway_is_refused_while_first_holds_the_lock(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first = GatewayPidLock(tmp_path)
    first.acquire()
    try:
        second = GatewayPidLock(tmp_path)
        with pytest.raises(SystemExit) as excinfo:
            second.acquire()
        assert excinfo.value.code == 1
        assert f"pid={os.getpid()}" in capsys.readouterr().err
        # The loser must not have touched the winner's pid file.
        assert json.loads((tmp_path / "gateway.pid").read_bytes())["pid"] == os.getpid()
        assert second.pid is None
    finally:
        first.release()


def test_lock_survives_release_of_an_earlier_holder(tmp_path: Path) -> None:
    """The restart race from #2119.

    A holds the lock; B has already opened the anchor; A releases; B locks the
    (still open) inode; C starts. With the anchor unlinked in ``release()`` C
    would create a fresh inode, win its own independent lock, and run
    alongside B. The anchor must persist so C contends on B's inode.
    """
    lock_path = tmp_path / "gateway.pid.lock"

    a = GatewayPidLock(tmp_path)
    a.acquire()

    b_fh = open(str(lock_path), "a+b")
    try:
        a.release()
        assert pidlock._try_lock(b_fh), "B should win once A has released"

        c = GatewayPidLock(tmp_path)
        with pytest.raises(SystemExit) as excinfo:
            c.acquire()
        assert excinfo.value.code == 1
        assert c.pid is None
    finally:
        pidlock._unlock(b_fh)
        b_fh.close()


def test_acquire_takes_the_lock_before_reconciling_the_pid_file(tmp_path: Path) -> None:
    """A leftover pid file naming a *live* pid no longer blocks or gets probed.

    Whoever holds the lock is the gateway; a pid file found under a freshly
    won lock is stale by construction, even when the pid it names happens to
    be alive (pid reuse, or the reporter's false-negative liveness probe).
    """
    pid_path = tmp_path / "gateway.pid"
    pid_path.write_bytes(json.dumps({"pid": os.getpid(), "start_ts": "old"}).encode())

    lock = GatewayPidLock(tmp_path)
    lock.acquire()
    try:
        written = json.loads(pid_path.read_bytes())
        assert written["pid"] == os.getpid()
        assert written["start_ts"] != "old"
        assert lock.pid == os.getpid()
    finally:
        lock.release()


def test_stale_pid_file_is_reported_and_overwritten(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    pid_path = tmp_path / "gateway.pid"
    pid_path.write_bytes(json.dumps({"pid": 999_999_999, "start_ts": "old"}).encode())

    lock = GatewayPidLock(tmp_path)
    with caplog.at_level("WARNING", logger="agentos.gateway.pidlock"):
        lock.acquire()
    try:
        assert any(r.msg == "gateway.pidlock.stale_overwritten" for r in caplog.records)
        assert json.loads(pid_path.read_bytes())["pid"] == os.getpid()
    finally:
        lock.release()


def test_acquire_does_not_install_signal_handlers(tmp_path: Path) -> None:
    """uvicorn owns SIGTERM/SIGINT; a handler here was dead code that would
    hard-kill the process if it ever ran (#2136)."""
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    lock = GatewayPidLock(tmp_path)
    lock.acquire()
    try:
        after = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
        assert after == before
    finally:
        lock.release()
