"""`_sandbox_request_for` must resolve a relative ``workdir`` (issue #1510).

The denial ledger fingerprints a `SandboxRequest`, and `cwd` is part of that
fingerprint. Dropping a relative `workdir` made every denial under a
subdirectory hash as a denial at the workspace root.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentos.sandbox.config import SandboxSettings
from agentos.sandbox.integration import configure_runtime, reset_runtime
from agentos.tools.builtin import shell
from agentos.tools.types import CallerKind, ToolContext, current_tool_context


@pytest.fixture(autouse=True)
def _runtime(tmp_path: Path):
    reset_runtime()
    configure_runtime(
        SandboxSettings(sandbox=True, backend="noop", security_grading=False),
        workspace=tmp_path,
    )
    token = current_tool_context.set(
        ToolContext(
            caller_kind=CallerKind.CLI,
            session_key="agent:main:test",
            workspace_dir=str(tmp_path),
        )
    )
    yield
    current_tool_context.reset(token)
    reset_runtime()


def _cwd_for(workdir: str | None) -> Path:
    built = shell._sandbox_request_for("exec_command", "rm -rf build", workdir)
    assert built is not None
    request, _policy, _session_id = built
    return request.cwd


def test_relative_workdir_resolves_against_the_workspace(tmp_path: Path) -> None:
    (tmp_path / "subproject").mkdir()

    assert _cwd_for("subproject") == (tmp_path / "subproject").resolve()


def test_dot_relative_workdir_resolves_against_the_workspace(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()

    assert _cwd_for("./tests") == (tmp_path / "tests").resolve()


def test_absolute_workdir_is_still_honoured(tmp_path: Path) -> None:
    other = tmp_path / "elsewhere"
    other.mkdir()

    assert _cwd_for(str(other)) == other.resolve()


def test_missing_workdir_falls_back_to_the_workspace(tmp_path: Path) -> None:
    assert _cwd_for(None) == tmp_path.resolve()


def test_denials_in_different_subdirectories_fingerprint_differently(tmp_path: Path) -> None:
    from agentos.sandbox.integration import action_fingerprint

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()

    built_a = shell._sandbox_request_for("exec_command", "rm -rf build", "a")
    built_b = shell._sandbox_request_for("exec_command", "rm -rf build", "b")
    assert built_a is not None and built_b is not None

    assert action_fingerprint(built_a[0]) != action_fingerprint(built_b[0])


@pytest.mark.asyncio
async def test_exec_command_records_the_denial_against_the_subdirectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: a denial under subproject/ must not hash as one at the root."""
    from agentos.sandbox.integration import action_fingerprint, get_runtime

    (tmp_path / "subproject").mkdir()

    async def _denied(**_kwargs: object) -> dict[str, object]:
        return {"status": "approval_denied"}

    monkeypatch.setattr(shell, "_check_exec_approval", _denied)

    await shell.exec_command("rm -rf build", workdir="subproject")

    runtime = get_runtime()
    assert runtime is not None
    recorded, _reason = await runtime.ledger.last_denial("agent:main:test")

    built = shell._sandbox_request_for("exec_command", "rm -rf build", "subproject")
    assert built is not None
    assert recorded == action_fingerprint(built[0])

    root = shell._sandbox_request_for("exec_command", "rm -rf build", None)
    assert root is not None
    assert recorded != action_fingerprint(root[0])
